"""
Collect gradient field using finite difference method (parallelized).
This serves as a ground truth comparison for the autograd-based gradients.
"""

import os
import time

import isaacgym
import numpy as np
import torch
from legged_gym.envs import *

from mups_codesign.config import CodesignConfig
from mups_codesign.data_logger import DataLogger
from mups_codesign.design_objective import DesignObjective
from mups_codesign.design_space import DesignSpace
from mups_codesign.mups_robot import MupsRobot
from mups_codesign.optim_helper import (rollout_control_loop,
                                        setup_isaac_env_and_policy)

np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=6, sci_mode=False)


def evaluate_objective_batch(
    env,
    control_policy,
    srb_env,
    design_objective_calculator,
    design_space,
    param_batch,
    n_control_iter,
):
    """
    Evaluate the objective for a batch of design parameters in parallel.
    
    Args:
        param_batch: (num_envs, 2) tensor of design parameter values
    
    Returns:
        objective_values: (num_envs,) numpy array of objective values
    """
    param_names = design_space.active_param_names

    # Set design parameters for all environments
    env.set_design_params(param_batch.detach())  # (num_envs, 2)
    srb_env.set_design_params(param_names, param_batch.detach())

    with torch.no_grad():
        env.reset()

        # Rollout and compute objective for all environments
        total_design_objective, _ = rollout_control_loop(
            env,
            control_policy,
            srb_env,
            design_objective_calculator,
            n_control_iter,
            headless=env.headless,
            modify_priv_obs=False,
            modify_cur_obs=False,
        )

    return total_design_objective.cpu().numpy()  # (num_envs,)


def compute_gradient_field_fd_parallel(
    env,
    control_policy,
    srb_env,
    design_objective_calculator,
    design_space,
    param_grid,
    n_control_iter,
    epsilon=1e-3,
    method="forward",
):
    """
    Compute gradient field using finite difference with parallelized evaluations.
    
    Args:
        param_grid: (num_envs, 2) tensor of design parameter values
        epsilon: perturbation size
        method: "central" or "forward"
    
    Returns:
        objective_values: (num_envs,) numpy array of objective values at center points
        grad_values: (num_envs, 2) numpy array of gradients
    """
    num_envs = param_grid.shape[0]
    num_params = param_grid.shape[1]
    
    # Evaluate center points
    print("  Evaluating center points...")
    obj_center = evaluate_objective_batch(
        env, control_policy, srb_env, design_objective_calculator,
        design_space, param_grid, n_control_iter
    )
    
    grad_values = np.zeros((num_envs, num_params))
    
    if method == "central":
        # Central difference: (f(x+h) - f(x-h)) / (2h)
        for k in range(num_params):
            # Create perturbation for parameter k
            delta = torch.zeros_like(param_grid)
            delta[:, k] = epsilon
            if k == 0:
                delta[:, k] = epsilon * 50000  # scale first param differently for hopper ups_ks
            
            print(f"  Evaluating +epsilon on param {k}...")
            param_plus = param_grid + delta
            obj_plus = evaluate_objective_batch(
                env, control_policy, srb_env, design_objective_calculator,
                design_space, param_plus, n_control_iter
            )
            
            print(f"  Evaluating -epsilon on param {k}...")
            param_minus = param_grid - delta
            obj_minus = evaluate_objective_batch(
                env, control_policy, srb_env, design_objective_calculator,
                design_space, param_minus, n_control_iter
            )
            
            grad_values[:, k] = (obj_plus - obj_minus) / (2 * epsilon)
    else:
        # Forward difference: (f(x+h) - f(x)) / h
        for k in range(num_params):
            # Create perturbation for parameter k
            delta = torch.zeros_like(param_grid)
            delta[:, k] = epsilon
            
            print(f"  Evaluating +epsilon on param {k}...")
            param_plus = param_grid + delta
            obj_plus = evaluate_objective_batch(
                env, control_policy, srb_env, design_objective_calculator,
                design_space, param_plus, n_control_iter
            )
            
            grad_values[:, k] = (obj_plus - obj_center) / epsilon
    
    return obj_center, grad_values


if __name__ == "__main__":
    # Grid resolution
    num_grid = 32
    num_envs = num_grid * num_grid

    # Finite difference settings
    FD_EPSILON = 1e-2  # Perturbation size
    FD_METHOD = "central"  # "central" or "forward"

    # Initialize codesign config
    objective_weights = {
        "heating_energy": 1.0,
        "mechanical_energy": 0.0,
        "height_tracking_error": 5.0,
    }
    design_config = CodesignConfig(
        num_envs=num_envs,  # Parallel environments for all grid points
        device="cuda",
        n_design_iter=1,
        n_control_iter=100,
        active_param_names=("ups_ks", "ups_l0"),
        objective_weights=objective_weights,
    )

    # Setup isaacgym environment and control policy
    env, control_policy = setup_isaac_env_and_policy(design_config)

    # Initialize codesign modules
    srb_env = MupsRobot(design_config)
    design_objective_calculator = DesignObjective(design_config)
    design_space = DesignSpace(design_config, requires_grad=False)  # No grad needed for FD
    logger = DataLogger(design_config.log_dir)

    if len(design_space.active_param_names) != 2:
        raise ValueError("Need exactly two active design parameters for 2D gradient field")

    # Generate a 2D grid of design parameters
    design_param_bounds = design_space.active_param_bounds.cpu().numpy()  # (2, 2)
    param1_span = torch.linspace(
        design_param_bounds[0, 0],
        design_param_bounds[0, 1],
        num_grid,
        device=design_config.device,
        dtype=design_config.dtype,
    )
    param2_span = torch.linspace(
        design_param_bounds[1, 0],
        design_param_bounds[1, 1],
        num_grid,
        device=design_config.device,
        dtype=design_config.dtype,
    )
    param1_grid, param2_grid = torch.meshgrid(param1_span, param2_span, indexing="xy")

    # Flatten grid to (num_envs, 2) for parallel processing
    param_grid = torch.stack(
        [param1_grid.reshape(-1), param2_grid.reshape(-1)],
        dim=-1,
    )  # (num_envs, 2)

    grid_param_names = design_space.active_param_names[:2]
    N_CONTROL_ITER = design_config.n_control_iter

    # Number of batched evaluations needed
    num_evals = 5 if FD_METHOD == "central" else 3
    print(f"Computing gradients via finite difference on {num_grid}x{num_grid} grid in parallel ({num_envs} envs)...")
    print(f"FD method: {FD_METHOD}, epsilon: {FD_EPSILON}")
    print(f"Number of batched evaluations: {num_evals}")
    print(f"Parameter 1 ({grid_param_names[0]}): [{design_param_bounds[0, 0]:.4f}, {design_param_bounds[0, 1]:.4f}]")
    print(f"Parameter 2 ({grid_param_names[1]}): [{design_param_bounds[1, 0]:.4f}, {design_param_bounds[1, 1]:.4f}]")

    start_time = time.time()

    # Compute all gradients with parallel batched evaluations
    objective_values, grad_values = compute_gradient_field_fd_parallel(
        env,
        control_policy,
        srb_env,
        design_objective_calculator,
        design_space,
        param_grid,
        N_CONTROL_ITER,
        epsilon=FD_EPSILON,
        method=FD_METHOD,
    )

    elapsed = time.time() - start_time
    print(f"Gradient computation (FD) completed in {elapsed:.1f}s")

    # Reshape results back to grid format
    objective_grid = objective_values.reshape(num_grid, num_grid)
    grad1_grid = grad_values[:, 0].reshape(num_grid, num_grid)
    grad2_grid = grad_values[:, 1].reshape(num_grid, num_grid)

    # Convert grids to numpy for saving
    param1_grid_np = param1_grid.cpu().numpy()
    param2_grid_np = param2_grid.cpu().numpy()

    print(f"Objective range: [{objective_grid.min():.4f}, {objective_grid.max():.4f}]")
    print(f"Grad1 range: [{grad1_grid.min():.6f}, {grad1_grid.max():.6f}]")
    print(f"Grad2 range: [{grad2_grid.min():.6f}, {grad2_grid.max():.6f}]")

    # Save results
    policy_id = design_config.policy_id
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    gradient_field_dir = os.path.join(design_config.log_dir, "gradient_fields")
    os.makedirs(gradient_field_dir, exist_ok=True)
    filename = f"hopper_{policy_id}_gradient_field_fd_{timestamp}.npz"
    output_path = os.path.join(gradient_field_dir, filename)

    np.savez(
        output_path,
        param1_grid=param1_grid_np,
        param2_grid=param2_grid_np,
        objective_grid=objective_grid,
        grad1_grid=grad1_grid,
        grad2_grid=grad2_grid,
        param_names=np.array(design_space.active_param_names),
        grid_param_names=np.array(grid_param_names),
        policy_id=np.array([policy_id or ""]),
        task=np.array(["hopper"]),
        fd_epsilon=np.array([FD_EPSILON]),
        fd_method=np.array([FD_METHOD]),
    )
    print(f"Gradient field (FD) data saved to: {output_path}")

    # Close logger
    logger.close()
