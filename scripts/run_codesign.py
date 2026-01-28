"""
Thin CLI that selects optimizer and calls optim.py. All isaacgym imports and logic isolated here.
"""

import isaacgym

import os
import pdb
import sys
import time
from collections import defaultdict
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

from mups_codesign.mups_robot import MupsRobot
from mups_codesign.isaac_env.hopper_standalone import HopperStandalone
from mups_codesign.isaac_env.hopper_standalone_config import HopperStandaloneCfg, HopperStandaloneCfgPPO


torch.autograd.set_detect_anomaly(True)

# Set print precision
np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=6, sci_mode=False)

# Fix manual seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


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
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
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

    # Initialize design optimization
    N_DESIGN_ITER = 100
    N_CONTROL_ITER = 100

    print(f"Design Iterations: {N_DESIGN_ITER}, Control Iterations: {N_CONTROL_ITER}")

    initial_design_params = torch.tensor([3000, 0.1], device=env.device)
    print(f"Initial Design Parameters: {initial_design_params}")

    design_params_normalized = nn.Parameter(
        initial_design_params / robot.design_param_scale, requires_grad=True
    ) # (num_params, )

    # First-order optimizer
    optimizer = optim.Adam([design_params_normalized], lr=5e-2)

    f_best_log = []
    x_best_log = [initial_design_params.cpu().numpy()]

    # Design iterations
    for design_iter in tqdm(range(N_DESIGN_ITER), desc="Design Iteration", ncols=80, file=sys.stdout):
        print("") # Flush a newline after tqdm progress bar

        total_design_objective = torch.zeros(env.num_envs, device=env.device)
        objective_term_sums = defaultdict(float)

        # Initialize environment and its buffers
        with torch.no_grad():
            env.reset()
            obs = env.get_observations()
            privileged_obs = env.get_privileged_observations()
            critic_obs = env.get_critic_observations()
            estimated_obs = env.get_estimated_observations()
            scan_obs = env.get_scan_observations()
            isaac_state = env.root_states.clone()
            dof_state = torch.hstack([env.dof_pos, env.dof_vel]).clone()
            # next_state = isaac_state.clone()

        # Optimized design parameters (requires grad)
        design_params_opt = design_params_normalized * robot.design_param_scale  # (num_params, )
        design_params_opt_detached = design_params_opt.detach()

        # Set design parameters for each environment
        env.set_design_params(design_params_opt_detached[None, :]) # (num_envs, num_params)

        # Task iterations
        for control_iter in range(N_CONTROL_ITER):
            time_start = time.time()

            # Step control policy
            #* Use the normalized design parameters as the privi_obs has to be clipped
            modified_privileged_obs = torch.cat(
                (
                    privileged_obs[:, :-2],
                    design_params_normalized.unsqueeze(0).expand(env.num_envs, -1),
                ),
                dim=-1,
            )

            actions = control_policy(obs, modified_privileged_obs, estimated_obs, scan_obs, adaptation_mode=False)

            # Step SRB dynamics
            srb_state, srb_torque, design_objective, objective_terms = robot.step_srb_dynamics(
                isaac_state,         #! non-diff, critical fix
                dof_state.clone(),   # non-diff
                actions,             # diff
                design_params_opt    # diff
            )

            # Update design objective sum
            total_design_objective = total_design_objective + design_objective
            
            # Update logging
            for name, value in objective_terms.items():
                objective_term_sums[name] += value.mean().item()

            # Now we step isaacgym dynamics
            with torch.no_grad():
                obs, privileged_obs, critic_obs, estimated_obs, scan_obs, rews, dones, infos = env.step(actions)
                isaac_state = env.root_states.clone()
                dof_state = torch.hstack([env.dof_pos, env.dof_vel]).clone()

            # Naive state alignment
            next_state = isaac_state + 0.9 * (srb_state - srb_state.detach())

            # Handle real time rendering
            if not args.headless:
                # Block rendering to wall clock
                time_elapsed = time.time() - time_start
                time_until_next_step = env.dt - time_elapsed
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        # Backprop
        optimizer.zero_grad()
        loss = total_design_objective.mean()
        loss.backward()

        # Clip gradient
        grad_norm_before_clipping = nn.utils.clip_grad_norm_(design_params_normalized, max_norm=1.0)
        print(f"Grad before clipping: {grad_norm_before_clipping:.4f}")

        # Step optimizer
        optimizer.step()

        design_params_normalized.data.clamp_(robot.design_param_bound[:, 0], robot.design_param_bound[:, 1])

        # Update optmization logs
        f_best = loss.item()
        x_best = (design_params_normalized.detach() * robot.design_param_scale).cpu().numpy()
        f_best_log.append(f_best)
        x_best_log.append(x_best)

        # Print design iteration summary
        print(f"Design Iteration {design_iter + 1}/{N_DESIGN_ITER}, Iteration loss: {f_best:.4f}")
        print(f"New Design Parameters: {x_best}")
        if objective_term_sums:
            averaged_terms = {name: value / N_CONTROL_ITER for name, value in objective_term_sums.items()}
            term_summary = ", ".join(f"{name}: {val:.4f}" for name, val in averaged_terms.items())
            print(f"Mean objective components -> {term_summary}")

    print("Design Optimization Completed.")

    # Save intermediate design parameters and objective logs to file
    log_file = f"design_optimization_logs_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(log_file, 
             f_best_log=np.array(f_best_log), 
             x_best_log=np.array(x_best_log))
    print(f"Optimization logs saved to: {log_file}")
    
