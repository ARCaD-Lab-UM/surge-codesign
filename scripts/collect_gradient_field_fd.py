"""
Collect gradient field using finite difference method.
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


def evaluate_objective_at_point(
    env,
    control_policy,
    srb_env,
    design_objective_calculator,
    design_space,
    param_values,
    n_control_iter,
):
    """
    Evaluate the objective at a single design parameter point (no gradient).
    
    Args:
        param_values: (2,) tensor of design parameter values (raw, not normalized)
    
    Returns:
        objective_value: scalar objective value
    """
    param_names = design_space.active_param_names

    # Set design parameters (no grad needed)
    env.set_design_params(param_values.detach()[None, :])  # (1, num_params)
    srb_env.set_design_params(param_names, param_values.detach()[None, :])

    with torch.no_grad():
        env.reset()

        # Rollout and compute objective
        total_design_objective, _ = rollout_control_loop(
            env,
            control_policy,
            srb_env,
            None,  # no normalized params needed for FD
            design_objective_calculator,
            n_control_iter,
            headless=env.headless,
            modify_priv_obs=False,
            modify_cur_obs=False,
        )

    objective_value = total_design_objective.mean().item()
    return objective_value


def compute_gradient_fd(
    env,
    control_policy,
    srb_env,
    design_objective_calculator,
    design_space,
    param_values,
    n_control_iter,
    epsilon=1e-3,
    method="central",
):
    """
    Compute gradient using finite difference.
    
    Args:
        param_values: (2,) tensor of design parameter values (raw, not normalized)
        epsilon: perturbation size (relative to parameter scale)
        method: "central" for central difference, "forward" for forward difference
    
    Returns:
        objective_value: scalar objective value at the center point
        grad_values: (2,) numpy array of gradients
    """
    num_params = param_values.shape[0]
    grad_values = np.zeros(num_params)
    
    # Get objective at center point
    objective_center = evaluate_objective_at_point(
        env, control_policy, srb_env, design_objective_calculator,
        design_space, param_values, n_control_iter
    )
    
    # Compute gradient for each parameter
    for k in range(num_params):
        # Create perturbation
        # delta = torch.zeros_like(param_values)
        # Use relative epsilon scaled by parameter magnitude
        # param_scale = abs(param_values[k].item()) if param_values[k].item() != 0 else 1.0
        # delta[k] = epsilon * param_scale
        
        if method == "central":
            # Central difference: (f(x+h) - f(x-h)) / (2h)
            param_plus = param_values + epsilon
            param_minus = param_values - epsilon
            
            obj_plus = evaluate_objective_at_point(
                env, control_policy, srb_env, design_objective_calculator,
                design_space, param_plus, n_control_iter
            )
            obj_minus = evaluate_objective_at_point(
                env, control_policy, srb_env, design_objective_calculator,
                design_space, param_minus, n_control_iter
            )
            
            grad_values[k] = (obj_plus - obj_minus) / (2 * epsilon)
        else:
            # Forward difference: (f(x+h) - f(x)) / h
            param_plus = param_values + epsilon
            
            obj_plus = evaluate_objective_at_point(
                env, control_policy, srb_env, design_objective_calculator,
                design_space, param_plus, n_control_iter
            )
            
            grad_values[k] = (obj_plus - objective_center) / epsilon
    
    return objective_center, grad_values


if __name__ == "__main__":
    # Initialize codesign config
    objective_weights = {
        "heating_energy": 1.0,
    }
    design_config = CodesignConfig(
        num_envs=1,  # Single environment for gradient computation
        device="cuda",
        n_design_iter=1,
        n_control_iter=100,
        active_param_names=("ups_ks", "ups_l0"),
        objective_weights=objective_weights,
    )

    # Finite difference settings
    FD_EPSILON = 1e-3  # Relative perturbation size
    FD_METHOD = "forward"  # "central" or "forward"

    # Setup isaacgym environment and control policy
    env, control_policy = setup_isaac_env_and_policy(design_config)

    # Initialize codesign modules
    srb_env = MupsRobot(design_config)
    design_objective_calculator = DesignObjective(design_config)
    design_space = DesignSpace(design_config, requires_grad=False)  # No grad needed for FD
    logger = DataLogger(design_config.log_dir)

    # Generate a 2D grid of design parameters
    num_grid = 16  # Grid resolution (16x16 = 256 gradient evaluations)
    
    if len(design_space.active_param_names) != 2:
        raise ValueError("Need exactly two active design parameters for 2D gradient field")

    design_param_bounds = design_space.active_param_bounds.cpu().numpy()  # (2, 2)
    param1_span = np.linspace(
        design_param_bounds[0, 0],
        design_param_bounds[0, 1],
        num_grid,
    )
    param2_span = np.linspace(
        design_param_bounds[1, 0],
        design_param_bounds[1, 1],
        num_grid,
    )
    param1_grid, param2_grid = np.meshgrid(param1_span, param2_span, indexing="xy")

    # Storage for objectives and gradients
    objective_grid = np.zeros((num_grid, num_grid))
    grad1_grid = np.zeros((num_grid, num_grid))
    grad2_grid = np.zeros((num_grid, num_grid))

    grid_param_names = design_space.active_param_names[:2]
    N_CONTROL_ITER = design_config.n_control_iter

    print(f"Computing gradients via finite difference on {num_grid}x{num_grid} grid...")
    print(f"FD method: {FD_METHOD}, epsilon: {FD_EPSILON}")
    print(f"Parameter 1 ({grid_param_names[0]}): [{design_param_bounds[0, 0]:.4f}, {design_param_bounds[0, 1]:.4f}]")
    print(f"Parameter 2 ({grid_param_names[1]}): [{design_param_bounds[1, 0]:.4f}, {design_param_bounds[1, 1]:.4f}]")

    # Note: Central difference requires 1 + 2*num_params evaluations per grid point
    # For 2 params with central diff: 5 evals per point (center + 2 per param)
    evals_per_point = 1 + (4 if FD_METHOD == "central" else 2)
    print(f"Evaluations per grid point: {evals_per_point}")

    total_points = num_grid * num_grid
    start_time = time.time()

    # Loop through each grid point and compute gradient via FD
    for i in range(num_grid):
        for j in range(num_grid):
            point_idx = i * num_grid + j
            param_values = torch.tensor(
                [param1_grid[i, j], param2_grid[i, j]],
                dtype=design_config.dtype,
                device=design_config.device,
            )

            objective_value, grad_values = compute_gradient_fd(
                env,
                control_policy,
                srb_env,
                design_objective_calculator,
                design_space,
                param_values,
                N_CONTROL_ITER,
                epsilon=FD_EPSILON,
                method=FD_METHOD,
            )

            objective_grid[i, j] = objective_value
            grad1_grid[i, j] = grad_values[0]
            grad2_grid[i, j] = grad_values[1]

            # Progress update
            if (point_idx + 1) % 10 == 0 or point_idx == total_points - 1:
                elapsed = time.time() - start_time
                eta = elapsed / (point_idx + 1) * (total_points - point_idx - 1)
                print(f"  [{point_idx + 1}/{total_points}] "
                      f"obj={objective_value:.4f}, grad=({grad_values[0]:.6f}, {grad_values[1]:.6f}) "
                      f"ETA: {eta:.1f}s")

    print(f"Gradient computation (FD) completed in {time.time() - start_time:.1f}s")

    # Save results
    policy_id = design_config.policy_id
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    gradient_field_dir = os.path.join(design_config.log_dir, "gradient_fields")
    os.makedirs(gradient_field_dir, exist_ok=True)
    filename = f"hopper_{policy_id}_gradient_field_fd_{timestamp}.npz"
    output_path = os.path.join(gradient_field_dir, filename)

    np.savez(
        output_path,
        param1_grid=param1_grid,
        param2_grid=param2_grid,
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
