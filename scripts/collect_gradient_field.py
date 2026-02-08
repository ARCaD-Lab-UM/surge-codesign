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


def compute_gradient_field_parallel(
    env,
    control_policy,
    srb_env,
    design_objective_calculator,
    design_space,
    param_grid,
    n_control_iter,
):
    """
    Compute gradients for all grid points in parallel using batched environments.
    
    Args:
        param_grid: (num_envs, 2) tensor of design parameter values (raw, not normalized)
    
    Returns:
        objective_values: (num_envs,) numpy array of objective values
        grad_values: (num_envs, 2) numpy array of gradients
    """
    param_names = design_space.active_param_names
    param_scales = design_space.active_param_scales  # (2,)

    # Create a leaf tensor for gradient computation: (num_envs, 2)
    normalized_params = (param_grid / param_scales).clone().detach().requires_grad_(True)
    scaled_params = normalized_params * param_scales

    # Set design parameters for all environments in parallel
    env.set_design_params(scaled_params.detach())  # (num_envs, 2)
    srb_env.set_design_params(param_names, scaled_params)  # keep grad

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
        modify_cur_obs=False,
    )  # (num_envs,)

    # Compute gradients: since each env's objective only depends on its own params,
    # summing the objectives gives correct per-sample gradients
    loss = total_design_objective.sum()
    loss.backward()

    objective_values = total_design_objective.detach().cpu().numpy()
    grad_values = normalized_params.grad.cpu().numpy()  # (num_envs, 2)

    return objective_values, grad_values


if __name__ == "__main__":
    # Grid resolution
    num_grid = 32
    num_envs = num_grid * num_grid

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
    design_space = DesignSpace(design_config, requires_grad=True)
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

    print(f"Computing gradients on {num_grid}x{num_grid} grid in parallel ({num_envs} envs)...")
    print(f"Parameter 1 ({grid_param_names[0]}): [{design_param_bounds[0, 0]:.4f}, {design_param_bounds[0, 1]:.4f}]")
    print(f"Parameter 2 ({grid_param_names[1]}): [{design_param_bounds[1, 0]:.4f}, {design_param_bounds[1, 1]:.4f}]")

    start_time = time.time()

    # Compute all gradients in one parallel pass
    objective_values, grad_values = compute_gradient_field_parallel(
        env,
        control_policy,
        srb_env,
        design_objective_calculator,
        design_space,
        param_grid,
        N_CONTROL_ITER,
    )

    elapsed = time.time() - start_time
    print(f"Gradient computation completed in {elapsed:.1f}s")

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
    filename = f"hopper_{policy_id}_gradient_field_{timestamp}.npz"
    output_path = os.path.join(gradient_field_dir, filename)

    np.savez(
        output_path,
        param1_grid=param1_grid_np,
        param2_grid=param2_grid_np,
        objective_grid=objective_grid,
        grad1_grid=grad1_grid,
        grad2_grid=grad2_grid,
        grid_param_names=np.array(grid_param_names),
        policy_id=np.array([policy_id or ""]),
        task=np.array(["hopper"]),
    )
    print(f"Gradient field data saved to: {output_path}")

    # Close logger
    logger.close()
