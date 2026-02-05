"""
Optimization helper for control loop and design iterations
"""

import pdb
import time
from collections import defaultdict

import torch
from torch import nn

from mups_codesign.data_logger import DataLogger
from mups_codesign.design_objective import DesignObjective
from mups_codesign.isaac_env.hopper import HopperRobot
from mups_codesign.mups_robot import MupsRobot


def rollout_control_loop(
    env: HopperRobot,
    control_policy: nn.Sequential,
    srb_env: MupsRobot,
    param_values_normalized: nn.Parameter,
    objective_calculator: DesignObjective,
    num_steps: int,
    headless: bool,
    modify_priv_obs: bool=True,
    logger: DataLogger = None,
):
    total_design_objective = torch.zeros(env.num_envs, device=env.device)
    objective_term_sums = defaultdict(float)

    # Task iterations
    for i in range(num_steps):
        time_start = time.time()

        # Get observations and states
        with torch.no_grad():
            obs = env.get_observations()
            privileged_obs = env.get_privileged_observations()
            estimated_obs = env.get_estimated_observations()
            scan_obs = env.get_scan_observations()
            isaac_state = env.root_states.clone()
            dof_state = torch.hstack([env.dof_pos, env.dof_vel]).clone()

        # Step control policy with design in privileged observation
        #* Use the normalized design parameters as the privi_obs has to be clipped
        modified_privileged_obs = privileged_obs.clone()
        if modify_priv_obs:
            # TODO: after the full policy is trained, we should always pass a full set of design params
            # TODO: and handle it outside this function
            modified_privileged_obs[:, -2:] = param_values_normalized[:2].unsqueeze(0)

        actions = control_policy(obs, modified_privileged_obs, estimated_obs, scan_obs, adaptation_mode=False)

        # Step isaacgym dynamics
        with torch.no_grad():
            env.step(actions)

        # Step SRB dynamics
        srb_state, motor_torque, info = srb_env.step_srb_dynamics(
            isaac_state,    #! non-diff, critical fix
            dof_state,      # non-diff
            actions,        # diff
        )

        if logger is not None:
            logger.log_control_step(
                i, 
                {
                    "srb_state": srb_state,
                    "dof_state": dof_state,
                    "motor_torque": motor_torque,
                    "info": info,
                }
            )

        #* No need for state alignment
        next_state = isaac_state + 1.0 * (srb_state - srb_state.detach())
        #* isaac_state: non-diff
        #* srb_state:   diff but slightly different value from isaac_state
        #* next_state:  diff and same value as isaac_state

        # Compute design objective
        design_objective, objective_terms = objective_calculator.calc_objective(
            next_state,     # diff
            dof_state,      # non-diff
            motor_torque    # diff
        )

        # Update design objective sum
        total_design_objective = total_design_objective + design_objective

        # Update logging
        for name, value in objective_terms.items():
            objective_term_sums[name] += value.mean().item()

        # Handle real time rendering
        if not headless:
            # Block rendering to wall clock
            time_elapsed = time.time() - time_start
            time_until_next_step = env.dt - time_elapsed
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    return total_design_objective, objective_term_sums
