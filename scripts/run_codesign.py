import isaacgym

import os
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

from mups_codesign.mups_robot import MupsRobot
from mups_codesign.config import CodesignConfig
from mups_codesign.data_logger import DataLogger
from mups_codesign.design_space import DesignSpace
from mups_codesign.design_objective import DesignObjective
from mups_codesign.optim_helper import rollout_control_loop, setup_isaac_env_and_policy, parse_seed
from mups_codesign.vis_helper import save_ad_graph


torch.autograd.set_detect_anomaly(True)

# Set print precision
np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=6, sci_mode=False)


if __name__ == '__main__':
    #* Initialize codesign config
    seed_override = parse_seed()
    design_config = CodesignConfig(
        **({'seed': seed_override} if seed_override is not None else {}),
        num_envs=10, 
        device="cuda",
        n_design_iter=int(os.environ.get('N_DESIGN_ITER', '50')),
        n_control_iter=int(os.environ.get('N_CONTROL_ITER', '100')),
        learning_rate=1e-2,
        raw_init_param_values=(7000, 0.15, 0.1, 0.02),
    )

    # Setup isaacgym environment and control policy
    env, control_policy = setup_isaac_env_and_policy(design_config)

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
        param_values_detached = design_space.detached_active_param_values # (num_params, )
        params_eval = param_values_detached.cpu().numpy() # (num_params, )
        params_normalized_eval = design_params_normalized.detach().cpu().numpy()

        # Set design parameters for each environment
        env.set_design_params({name: val for name, val in zip(param_names, param_values_detached)}) # (num_params, )
        srb_env.set_design_params(param_names, param_values.unsqueeze(0).expand(design_config.num_envs, -1)) # (num_envs, num_params)
        with torch.no_grad():
            env.reset()

        total_design_objective, objective_term_sums = rollout_control_loop(
            env,
            control_policy,
            srb_env,
            design_objective_calculator,
            N_CONTROL_ITER,
            headless=env.headless
        )

        # Backprop
        optimizer.zero_grad()
        loss = total_design_objective.mean()
        loss.backward()

        # save_ad_graph(loss, {"param:": design_space.active_normalized_param_values})

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
