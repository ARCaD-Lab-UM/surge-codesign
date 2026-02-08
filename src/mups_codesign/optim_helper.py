"""
Optimization helper for control loop and design iterations
"""

import pdb
import time
from collections import defaultdict

import torch
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.task_registry import TaskRegistry
from torch import nn

from mups_codesign.config import CodesignConfig
from mups_codesign.data_logger import DataLogger
from mups_codesign.design_objective import DesignObjective
from mups_codesign.design_space import DesignSpace
from mups_codesign.isaac_env.hopper import HopperRobot
from mups_codesign.isaac_env.hopper_config import HopperCfg, HopperCfgPPO
from mups_codesign.mups_robot import MupsRobot


def setup_isaac_env_and_policy(design_config: CodesignConfig):
    # Parse isaacgym arguments
    args = get_args()
    args.task = "hopper"
    args.headless = True
    args.load_run = design_config.policy_id
    task_registry.register(
        "hopper",
        HopperRobot,
        HopperCfg(),
        HopperCfgPPO()
    )

    # Fetch config for env and policy
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Override config from codesign config
    env_cfg.env.num_envs = design_config.num_envs
    env_cfg.seed = design_config.seed
    train_cfg.seed = design_config.seed

    # Override env_cfg for evaluation
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.noise.add_noise = False
    env_cfg.commands.zero_command = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_center_of_mass = False
    env_cfg.domain_rand.randomize_kp_kd = False

    # Override train_cfg to load pre-trained policy
    train_cfg.runner.resume = True

    # Make isaacgym environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Load control policy in inference mode
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    control_policy = ppo_runner.get_inference_policy(device=env.device)

    # Freeze policy parameters
    for param in ppo_runner.alg.actor_critic.parameters():
        param.requires_grad = False

    return env, control_policy


def rollout_control_loop(
    env: HopperRobot,
    control_policy: nn.Sequential,
    srb_env: MupsRobot,
    objective_calculator: DesignObjective,
    num_steps: int,
    headless: bool,
    modify_priv_obs: bool=True,
    modify_cur_obs: bool=False,
    logger: DataLogger = None,
):
    total_design_objective = torch.zeros(env.num_envs, device=env.device)
    objective_term_sums = defaultdict(float)

    with torch.no_grad():
        obs = env.get_observations()
        privileged_obs = env.get_privileged_observations()
        estimated_obs = env.get_estimated_observations()
        scan_obs = env.get_scan_observations()
        isaac_state = env.root_states.clone()
        dof_state = torch.hstack([env.dof_pos, env.dof_vel]).clone()
        next_state = env.root_states.clone()

    # Task iterations
    for i in range(num_steps):
        time_start = time.time()

        # Step control policy with design in privileged observation
        #* Use the normalized design parameters as the privi_obs has to be clipped
        modified_privileged_obs = privileged_obs.clone()
        if modify_priv_obs:
            modified_privileged_obs[:, -DesignSpace.PARAM_NUMS:] = srb_env.normalized_design_params

        # TODO: verify this is actually helpful
        # Fill obs with aligned next_state to carry gradients from SRB
        partial_diff_obs_from_srb = torch.cat(
            (
                next_state[:, 2:3] * env.obs_scales.xyz_pos, # height
                next_state[:, 7:10] * env.obs_scales.lin_vel, # lin vel
                next_state[:, 10:13] * env.obs_scales.ang_vel, # ang vel
                quat_rotate_inverse(next_state[:, 3:7], env.gravity_vec), # projected gravity
            ),
            dim=-1
        ) # (num_envs, 10)
        modified_obs = obs.clone()
        if modify_cur_obs:
            modified_obs[:, -env.num_proprio:-env.num_proprio+10] = partial_diff_obs_from_srb

        actions = control_policy(modified_obs, modified_privileged_obs, estimated_obs, scan_obs, adaptation_mode=False)

        # Step SRB dynamics
        srb_state, motor_torque, info = srb_env.step_srb_dynamics(
            isaac_state,    #! non-diff, critical fix
            dof_state,      # non-diff
            actions,        # diff
        )

        # Step isaacgym dynamics
        with torch.no_grad():
            env.step(actions)
            obs = env.get_observations()
            privileged_obs = env.get_privileged_observations()
            estimated_obs = env.get_estimated_observations()
            scan_obs = env.get_scan_observations()
            isaac_state = env.root_states.clone()
            dof_state = torch.hstack([env.dof_pos, env.dof_vel]).clone()

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

        #* State alignment
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
