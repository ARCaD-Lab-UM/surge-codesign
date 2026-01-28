import isaacgym

import os
import pdb
import glob
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

from mups_codesign.mups_robot import MupsRobot
from mups_codesign.isaac_env.hopper_standalone import HopperStandalone
from mups_codesign.isaac_env.hopper_standalone_config import HopperStandaloneCfg, HopperStandaloneCfgPPO

# Set print precision
np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=6, sci_mode=False)


if __name__ == '__main__':
    # Parse arguments
    args = get_args()

    task_registry.register(
        "hopper",
        HopperStandalone,
        HopperStandaloneCfg(),
        HopperStandaloneCfgPPO()
    )

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Override some parameters for testing
    env_cfg.env.num_envs = 4096
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_center_of_mass = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.zero_command = False
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]

    # Make environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Initialize design parameterized robot model
    robot = MupsRobot(num_env=env.num_envs, device=env.device)

    # Load control policy in inference mode
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    control_policy = ppo_runner.get_inference_policy(device=env.device)


    # Generate a design landscape grid
    num_grid = np.sqrt(env.num_envs).astype(int)
    assert num_grid**2 == env.num_envs, "For grid design, num_envs should be a perfect square"

    design_param_range = robot.design_param_bound * robot.design_param_scale.unsqueeze(-1)

    param1_span = torch.linspace(
        design_param_range[0, 0], 
        design_param_range[0, 1], 
        num_grid,
        device=env.device)
    
    param2_span = torch.linspace(
        design_param_range[1, 0], 
        design_param_range[1, 1], 
        num_grid,
        device=env.device)
    
    param1_grid, param2_grid = torch.meshgrid(param1_span, param2_span, indexing='xy')

    design_param_grid = torch.stack([
        param1_grid.reshape(-1),
        param2_grid.reshape(-1)
    ], dim=-1)  # (num_envs, 2)

    total_design_objective = torch.zeros(env.num_envs, device=env.device)

    N_CONTROL_ITER = 100
    ISO_PITCH = np.deg2rad(25)        # Pitch (up/down) angle
    ISO_YAW   = np.deg2rad(45)        # Yaw (rotation) angle
    ISO_DIST  = 2.0                   # metres from the robot

    # Build camera direction vector
    cam_dir_vec   = np.array([
        -np.cos(ISO_PITCH) * np.cos(ISO_YAW),
        -np.cos(ISO_PITCH) * np.sin(ISO_YAW),
        np.sin(ISO_PITCH)
    ])

    # Set custom camera
    robot_pos = env.root_states[0, :3].cpu().numpy()
    camera_pos = robot_pos + ISO_DIST * cam_dir_vec
    env.set_camera(camera_pos, robot_pos)

    # Set design parameters to the grid
    env.set_design_params(design_param_grid)

    # Initialize state buffers
    with torch.no_grad():
        env.reset()
        obs = env.get_observations()
        privileged_obs = env.get_privileged_observations()
        critic_obs = env.get_critic_observations()
        estimated_obs = env.get_estimated_observations()
        scan_obs = env.get_scan_observations()
        isaac_state = env.root_states.clone()
        dof_state = torch.hstack([env.dof_pos, env.dof_vel]).clone()

    # Task iterations
    for control_iter in range(N_CONTROL_ITER):
        time_start = time.time()

        with torch.no_grad():
            # Step control policy
            actions = control_policy(obs, privileged_obs, estimated_obs, scan_obs, adaptation_mode=False)

            # Step SRB dynamics
            srb_state, srb_torque, design_objective, _ = robot.step_srb_dynamics(
                isaac_state.clone(),
                dof_state.clone(),
                actions,
                design_param_grid.T,
            )

            # Step isaacgym dynamics
            obs, privileged_obs, critic_obs, estimated_obs, scan_obs, rews, dones, infos = env.step(actions)
            isaac_state = env.root_states.clone()
            dof_state = torch.hstack([env.dof_pos, env.dof_vel]).clone()

            # Update debug visualization
            env.draw_debug_vis_srb(srb_state)


        # Accumulate design objective
        # design_objective = robot.calc_design_objective(srb_state, dof_state, env.torques, design_param_grid.T)
        total_design_objective = total_design_objective + design_objective

        # Handle real time rendering
        if not args.headless:
            # Block rendering to wall clock
            time_elapsed = time.time() - time_start
            time_until_next_step = env.dt - time_elapsed
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    design_objective_grid = total_design_objective.reshape(num_grid, num_grid).cpu().numpy()

    # Plot design landscape
    plt.figure(figsize=(8, 6))
    plt.contourf(param1_grid.cpu().numpy(),
                 param2_grid.cpu().numpy(),
                 design_objective_grid,
                 levels=20,
                 cmap='jet')
    plt.colorbar(label='Design Objective)')
    plt.xlabel('Spring Stiffness Ks')
    plt.ylabel('Rest Length L0')
    plt.title("Design Landscape")

    # Plot grid optimum
    min_idx_flat = np.argmin(design_objective_grid)
    min_idx = np.unravel_index(min_idx_flat, (num_grid, num_grid))
    opt_param1 = param1_span[min_idx[1]].cpu().item()
    opt_param2 = param2_span[min_idx[0]].cpu().item()
    opt_objective = design_objective_grid[min_idx[0], min_idx[1]]

    # Find the latest optimization log file
    log_pattern = "design_optimization_logs_*.npz"
    log_files = glob.glob(log_pattern)
    if log_files:
        latest_log_file = max(log_files, key=os.path.getctime)
        print(f"Loading optimization logs from: {latest_log_file}")
        
        # Load the optimization data
        log_data = np.load(latest_log_file)
        f_best_log = log_data['f_best_log']
        x_best_log = log_data['x_best_log']
        
        # Plot the optimization trajectory
        plt.scatter(x_best_log[:, 0], x_best_log[:, 1],
                    color="black", marker="^", s=100, label='Optimization Path')
        plt.scatter(x_best_log[0, 0], x_best_log[0, 1], edgecolor="black", 
                    color='cyan', marker='s', s=100, label='Start')
        plt.scatter(x_best_log[-1, 0], x_best_log[-1, 1], edgecolor="black",
                    color='magenta', marker='*', s=150, label='Final Best')
        
        print(f"Optimization Final Best at Ks={x_best_log[-1, 0]:.4f}, L0={x_best_log[-1, 1]:.4f} with Objective={f_best_log[-1]:.4f}")
    else:
        print("No optimization log files found")

    # Show plot
    plt.legend()
    plt.tight_layout()
    if log_files:
        plt.savefig(latest_log_file.replace(".npz", "_landscape.png"), dpi=300)
    plt.show()
