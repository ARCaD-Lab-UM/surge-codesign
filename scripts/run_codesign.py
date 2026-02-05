"""
Thin CLI that selects optimizer and calls optim.py. All isaacgym imports and logic isolated here.
"""

import isaacgym

import pdb
import sys
import time
from dataclasses import asdict
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

from mups_codesign.mups_robot import MupsRobot
from mups_codesign.config import CodesignConfig
from mups_codesign.data_logger import DataLogger
from mups_codesign.design_space import DesignSpace
from mups_codesign.design_objective import DesignObjective
from mups_codesign.optim_helper import rollout_control_loop

from mups_codesign.isaac_env.hopper import HopperRobot
from mups_codesign.isaac_env.hopper_config import HopperCfg, HopperCfgPPO


torch.autograd.set_detect_anomaly(True)

# Set print precision
np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=6, sci_mode=False)


if __name__ == '__main__':
    # Parse isaacgym arguments
    args = get_args()
    args.task = "hopper"
    task_registry.register(
        "hopper",
        HopperRobot,
        HopperCfg(),
        HopperCfgPPO()
    )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    #* Initialize codesign config
    design_config = CodesignConfig(
        num_envs=10, 
        device=args.sim_device,
        n_design_iter=200,
        n_control_iter=100,
        learning_rate=2e-3,
        raw_init_param_values=(6000, 0.11),
    )

    # Override config from codesign config
    env_cfg.env.num_envs = design_config.num_envs
    env_cfg.seed = design_config.seed
    train_cfg.seed = design_config.seed

    # Override env_cfg for evaluation
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.noise.add_noise = False
    env_cfg.commands.zero_command = False
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_center_of_mass = False
    env_cfg.domain_rand.randomize_kp_kd = False

    # Override train_cfg to load pre-trained policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = design_config.policy_id

    # Make isaacgym environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Load control policy in inference mode
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    control_policy = ppo_runner.get_inference_policy(device=env.device)

    # Freeze policy parameters
    for param in ppo_runner.alg.actor_critic.parameters():
        param.requires_grad = False

    # Codesign parameters
    N_DESIGN_ITER = design_config.n_design_iter
    N_CONTROL_ITER = design_config.n_control_iter
    LEARN_RATE = design_config.learning_rate
    print(f"Design Iterations: {N_DESIGN_ITER}, Control Iterations: {N_CONTROL_ITER}")
    print(f"Initial Design Parameters: {design_config.raw_init_param_values}")

    #* Initialize codesign modules
    srb_env = MupsRobot(design_config)
    design_objective_calculator = DesignObjective(design_config)
    design_space = DesignSpace(design_config)
    logger = DataLogger(root_dir=design_config.log_dir)

    design_params_normalized = design_space.active_normalized_param_values #* this is the leaf of computation graph, no need to rebuild

    # First-order optimizer
    optimizer = optim.Adam([design_params_normalized], lr=LEARN_RATE)

    logger.log_metadata({
        "args": vars(args),
        "design_config": asdict(design_config),
        "param_names": list(design_space.active_param_names),
        "n_design_iter": N_DESIGN_ITER,
        "n_control_iter": N_CONTROL_ITER,
        "learn_rate": LEARN_RATE,
    })
    best_loss = float("inf")
    best_params = None

    # Design iterations
    for design_iter in tqdm(range(N_DESIGN_ITER), desc="Design Iteration", ncols=80, file=sys.stdout):
        print("") # Flush a newline after tqdm progress bar
        iter_start = time.time()

        # Retrieve design variables in shape (num_params,)
        param_names = design_space.active_param_names
        param_values = design_space.active_param_values #* this is not the leaf of computation graph, so it has to be rebuilt every iteration
        param_values_detached = design_space.detached_active_param_values
        params_eval = param_values_detached.cpu().numpy()
        params_normalized_eval = design_params_normalized.detach().cpu().numpy()

        # Set design parameters for each environment
        env.set_design_params(param_values_detached[None, :]) # (num_envs, num_params)
        srb_env.set_design_params(param_names, param_values[None, :]) # keep grad
        with torch.no_grad():
            env.reset()

        total_design_objective, objective_term_sums = rollout_control_loop(
            env,
            control_policy,
            srb_env,
            design_params_normalized,
            design_objective_calculator,
            N_CONTROL_ITER,
            headless=args.headless
        )

        # Backprop
        optimizer.zero_grad()
        loss = total_design_objective.mean()
        loss.backward()

        # Record and clip gradients
        grad_values = None
        if design_params_normalized.grad is not None:
            grad_values = design_params_normalized.grad.detach().cpu().numpy()

        grad_norm_before_clipping = nn.utils.clip_grad_norm_(design_params_normalized, max_norm=1.0)
        print(f"Grad before clipping: {grad_norm_before_clipping:.4f}")

        # Step optimizer
        optimizer.step()

        design_space.project_active_params_into_bounds()

        # Update optmization logs
        f_best = loss.item()
        x_best = design_space.active_param_values.detach().cpu().numpy()
        if f_best < best_loss:
            best_loss = f_best
            best_params = params_eval.copy()

        # Print design iteration summary
        print(f"Design Iteration {design_iter + 1}/{N_DESIGN_ITER}, Iteration loss: {f_best:.4f}")
        print(f"New Design Parameters: {x_best}")
        term_summary = ", ".join(f"{name}: {val:.4f}" for name, val in objective_term_sums.items())
        print(f"Mean objective components -> {term_summary}")

        iter_time_s = time.time() - iter_start
        logger.log_iteration(
            iteration=design_iter,
            objective_total=f_best,
            objective_terms=objective_term_sums,
            params_value=params_eval,
            params_normalized=params_normalized_eval,
            grad_norm=float(grad_norm_before_clipping),
            grad_terms=grad_values,
            best_loss=best_loss,
            best_params=best_params,
            extra={
                "lr": LEARN_RATE, 
                "iter_time_s": iter_time_s
                },
        )

    print("Design Optimization Completed.")

    # Close logger
    logger.close()
    print(f"Logs saved to {logger.run_dir}")
