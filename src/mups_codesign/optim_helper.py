"""
Optimization helper for control loop and design iterations
"""

import time
from collections import defaultdict

import torch

from mups_codesign.design_objective import DesignObjective
from mups_codesign.design_space import DesignSpace
from mups_codesign.mups_robot import MupsRobot


def rollout_control_loop(
    env,
    control_policy,
    srb_env: MupsRobot,
    design_space: DesignSpace,
    objective_calculator: DesignObjective,
    num_steps: int,
    headless: bool,
):
    total_design_objective = torch.zeros(env.num_envs, device=env.device)
    objective_term_sums = defaultdict(float)

    # Retrieve design variables in shape (num_params,)
    param_names = design_space.active_param_names
    param_values = design_space.active_param_values
    param_values_detached = design_space.detached_active_param_values
    param_values_normalized = design_space.active_normalized_param_values

    # Initialize environment and its buffers
    with torch.no_grad():
        env.reset()
        obs = env.get_observations()
        privileged_obs = env.get_privileged_observations()
        estimated_obs = env.get_estimated_observations()
        scan_obs = env.get_scan_observations()
        isaac_state = env.root_states.clone()
        dof_state = torch.hstack([env.dof_pos, env.dof_vel]).clone()

    # Set design parameters for each environment
    env.set_design_params(param_values_detached[None, :]) # (num_envs, num_params)
    srb_env.set_design_params(param_names, param_values[None, :]) # keep grad

    # Task iterations
    for _ in range(num_steps):
        time_start = time.time()

        # Step control policy with design in privileged observation
        #* Use the normalized design parameters as the privi_obs has to be clipped
        # TODO: double check this part
        modified_privileged_obs = torch.cat(
            (
                privileged_obs[:, :-2],
                param_values_normalized.unsqueeze(0).expand(env.num_envs, -1),
            ),
            dim=-1,
        )
        actions = control_policy(obs, modified_privileged_obs, estimated_obs, scan_obs, adaptation_mode=False)

        # Step SRB dynamics and compute design objective
        srb_state, motor_torque = srb_env.step_srb_dynamics(
            isaac_state,    #! non-diff, critical fix
            dof_state,      # non-diff
            actions,        # diff
        )
        design_objective, objective_terms = objective_calculator.calc_objective(
            srb_state,      # diff
            dof_state,      # non-diff
            motor_torque    # diff
        )

        # Now we step isaacgym dynamics
        with torch.no_grad():
            obs, privileged_obs, _, estimated_obs, scan_obs, _, _, _ = env.step(actions)
            isaac_state = env.root_states.clone()
            dof_state = torch.hstack([env.dof_pos, env.dof_vel]).clone()

        #* No need for state alignment
        # next_state = isaac_state + 0.9 * (srb_state - srb_state.detach())

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
