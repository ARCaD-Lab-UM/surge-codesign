import os
import time

import isaacgym
import numpy as np
import torch
from legged_gym.envs import *

from surge_codesign.config import CodesignConfig
from surge_codesign.data_logger import DataLogger
from surge_codesign.design_objective import DesignObjective
from surge_codesign.design_space import DesignSpace
from surge_codesign.mups_robot import MupsRobot
from surge_codesign.optim_helper import rollout_control_loop, setup_isaac_env_and_policy
from surge_codesign.vis_helper import plot_contour


np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=6, sci_mode=False)


def _set_default_camera(env):
    iso_pitch = np.deg2rad(25)
    iso_yaw = np.deg2rad(45)
    iso_dist = 2.0
    cam_dir_vec = np.array([
        -np.cos(iso_pitch) * np.cos(iso_yaw),
        -np.cos(iso_pitch) * np.sin(iso_yaw),
        np.sin(iso_pitch),
    ])
    robot_pos = env.root_states[0, :3].cpu().numpy()
    camera_pos = robot_pos + iso_dist * cam_dir_vec
    env.set_camera(camera_pos, robot_pos)


if __name__ == "__main__":
    #* Initialize codesign config
    design_config = CodesignConfig(
        num_envs=4096, 
        device="cuda",
        n_control_iter=100,
        active_param_names=["ups_ks", "ups_l0"],
    )

    # Setup isaacgym environment and control policy
    env, control_policy = setup_isaac_env_and_policy(design_config)

    #* Initialize codesign modules
    srb_env = MupsRobot(design_config)
    design_objective_calculator = DesignObjective(design_config)
    design_space = DesignSpace(design_config, requires_grad=False)
    logger = DataLogger(design_config.log_dir)

    # Generate a design landscape grid
    num_grid = int(np.sqrt(env.num_envs))
    if num_grid * num_grid != env.num_envs:
        raise ValueError("For grid design, num_envs should be a perfect square")

    if len(design_space.active_param_names) != 2:
        raise ValueError("Need two active design parameters to build a 2D grid")

    design_param_bounds = design_space.active_param_bounds.cpu().numpy()  # (2, 2)
    param1_span = torch.linspace(
        design_param_bounds[0, 0],
        design_param_bounds[0, 1],
        num_grid,
        device=env.device,
    )
    param2_span = torch.linspace(
        design_param_bounds[1, 0],
        design_param_bounds[1, 1],
        num_grid,
        device=env.device,
    )
    param1_grid, param2_grid = torch.meshgrid(param1_span, param2_span, indexing="xy")

    grid_points = torch.stack(
        [param1_grid.reshape(-1), param2_grid.reshape(-1)],
        dim=-1,
    )  # (num_envs, 2)
    base_params = design_space.active_param_values.detach()
    grid_param_names = design_space.active_param_names[:2]

    design_param_grid = base_params.repeat(env.num_envs, 1) # (num_envs, num_params)
    design_param_grid[:, 0] = grid_points[:, 0]
    design_param_grid[:, 1] = grid_points[:, 1]

    if not env.headless:
        _set_default_camera(env)


    # Set design parameters to the grid
    env.set_design_params({name: val for name, val in zip(grid_param_names, design_param_grid.T.detach())})  # (2, num_envs)
    srb_env.set_design_params(grid_param_names, design_param_grid[:, :2])

    # Rollout control to evaluate design objective
    with torch.no_grad():
        env.reset()
        total_design_objective, _ = rollout_control_loop(
            env,
            control_policy,
            srb_env,
            design_objective_calculator,
            design_config.n_control_iter,
            headless=env.headless,
            modify_priv_obs=False,
            # logger=logger
        )

    objective_grid = total_design_objective.reshape(num_grid, num_grid).cpu().numpy()

    policy_id = design_config.policy_id
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    landscape_dir = os.path.join(design_config.log_dir, "landscapes")
    os.makedirs(landscape_dir, exist_ok=True)
    filename = f"hopper_{policy_id}_landscape_{timestamp}.npz"
    output_path = os.path.join(landscape_dir, filename)

    np.savez(
        output_path,
        param1_grid=param1_grid.cpu().numpy(),
        param2_grid=param2_grid.cpu().numpy(),
        objective_grid=objective_grid,
        param_names=np.array(design_space.active_param_names),
        grid_param_names=np.array(grid_param_names),
        policy_id=np.array([policy_id or ""]),
        task=np.array(["hopper"]),
    )

    print(f"Landscape saved to: {output_path}")

    contour_path = output_path.replace(".npz", "_contour.png")
    plot_contour(
        param1_grid.cpu().numpy(),
        param2_grid.cpu().numpy(),
        objective_grid,
        grid_param_names,
        save_path=contour_path,
        show=False,
    )
    print(f"Saved contour plot to: {contour_path}")

    # Close logger
    logger.close()