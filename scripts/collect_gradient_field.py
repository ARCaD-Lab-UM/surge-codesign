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


def compute_gradient_at_point(
    env,
    control_policy,
    srb_env,
    design_objective_calculator,
    design_space,
    param_values,
    n_control_iter,
):
    """
    Compute the gradient of the objective w.r.t. design parameters at a single point.
    
    Args:
        param_values: (2,) tensor of design parameter values (raw, not normalized)
    
    Returns:
        objective_value: scalar objective value
        grad_values: (2,) numpy array of gradients
    """
    param_names = design_space.active_param_names
    param_scales = design_space.active_param_scales

    # Create a leaf tensor for gradient computation
    normalized_params = (param_values / param_scales).clone().detach().requires_grad_(True)
    scaled_params = normalized_params * param_scales

    # Set design parameters
    env.set_design_params(scaled_params.detach()[None, :])  # (1, num_params)
    srb_env.set_design_params(param_names, scaled_params[None, :])  # keep grad

    with torch.no_grad():
        env.reset()

    # Rollout and compute objective
    total_design_objective, _ = rollout_control_loop(
        env,
        control_policy,
        srb_env,
        normalized_params,
        design_objective_calculator,
        n_control_iter,
        headless=env.headless,
        modify_cur_obs=False,
    )

    # Compute gradient
    loss = total_design_objective.mean()
    loss.backward()

    objective_value = loss.item()
    grad_values = normalized_params.grad.cpu().numpy()

    return objective_value, grad_values


if __name__ == "__main__":
    # Initialize codesign config
    objective_weights = {
        "heating_energy": 1.0,
        "height_tracking_error": 0.0,
    }
    design_config = CodesignConfig(
        num_envs=1,  # Single environment for gradient computation
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

    print(f"Computing gradients on {num_grid}x{num_grid} grid...")
    print(f"Parameter 1 ({grid_param_names[0]}): [{design_param_bounds[0, 0]:.4f}, {design_param_bounds[0, 1]:.4f}]")
    print(f"Parameter 2 ({grid_param_names[1]}): [{design_param_bounds[1, 0]:.4f}, {design_param_bounds[1, 1]:.4f}]")

    total_points = num_grid * num_grid
    start_time = time.time()

    # Loop through each grid point and compute gradient
    for i in range(num_grid):
        for j in range(num_grid):
            point_idx = i * num_grid + j
            param_values = torch.tensor(
                [param1_grid[i, j], param2_grid[i, j]],
                dtype=design_config.dtype,
                device=design_config.device,
            )

            objective_value, grad_values = compute_gradient_at_point(
                env,
                control_policy,
                srb_env,
                design_objective_calculator,
                design_space,
                param_values,
                N_CONTROL_ITER,
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

    print(f"Gradient computation completed in {time.time() - start_time:.1f}s")

    # Save results
    policy_id = design_config.policy_id
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    gradient_field_dir = os.path.join(design_config.log_dir, "gradient_fields")
    os.makedirs(gradient_field_dir, exist_ok=True)
    filename = f"hopper_{policy_id}_gradient_field_{timestamp}.npz"
    output_path = os.path.join(gradient_field_dir, filename)

    np.savez(
        output_path,
        param1_grid=param1_grid,
        param2_grid=param2_grid,
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
