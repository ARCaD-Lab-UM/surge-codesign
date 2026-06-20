"""
Optimization helper for control loop and design iterations
"""

import os
import pdb
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import HopperRobot
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.task_registry import TaskRegistry
from torch import nn

from surge_codesign.config import CodesignConfig
from surge_codesign.data_logger import DataLogger
from surge_codesign.design_objective import DesignObjective
from surge_codesign.design_space import DesignSpace
from surge_codesign.mups_robot import MupsRobot


def parse_seed():
    """Extract --seed from sys.argv (isaacgym's parser also defines it)."""
    for i, arg in enumerate(sys.argv):
        if arg == '--seed' and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return None  # use CodesignConfig default


def setup_isaac_env_and_policy(design_config: CodesignConfig):
    # Parse isaacgym arguments
    args = get_args()
    args.task = "hopper"
    args.headless = True
    args.load_run = design_config.policy_id

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
    env_cfg.domain_rand.randomize_design = False  # codesign sets designs explicitly

    # Override train_cfg to load pre-trained policy
    train_cfg.runner.resume = True

    # Make isaacgym environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Load control policy in inference mode from the shipped checkpoints directory
    # (<policy_root>/<policy_id>/model_*.pt), resolved relative to the repo root.
    policy_log_root = design_config.policy_root
    if not os.path.isabs(policy_log_root):
        policy_log_root = os.path.join(LEGGED_GYM_ROOT_DIR, policy_log_root)
    ppo_runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=policy_log_root
    )
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


def evaluate_population(
    candidates_normalized: list,
    env,
    control_policy,
    srb_env: MupsRobot,
    design_objective_calculator: DesignObjective,
    design_space: DesignSpace,
    design_config: CodesignConfig,
):
    """
    Evaluate all CMA-ES candidates in parallel (one candidate per IsaacGym environment).
    Uses no_grad for pure fitness evaluation in true (non-differentiable) dynamics.

    Args:
        candidates_normalized: List of normalized design parameters, each shape (num_params,)

    Returns:
        fitness_values: List of objective values (one per candidate)
        objective_terms_list: List of objective term dicts (one per candidate)
        candidates_raw: Raw (un-normalized) candidate parameters, shape (pop_size, num_params)
    """
    pop_size = len(candidates_normalized)

    # Convert all candidates to tensor: (pop_size, num_params)
    candidates_tensor = torch.tensor(
        np.array(candidates_normalized),
        dtype=design_config.dtype,
        device=design_config.device
    )
    # Convert normalized -> raw values
    candidates_raw = candidates_tensor * design_space.active_param_scales  # (pop_size, num_params)

    # Set design parameters: each environment gets a different candidate
    param_names = design_space.active_param_names
    env.set_design_params({name: val for name, val in zip(param_names, candidates_raw.T.detach())})  # (num_params, pop_size)
    srb_env.set_design_params(param_names, candidates_raw)  # (pop_size, num_params)

    with torch.no_grad():
        env.reset()
        total_design_objective, objective_term_sums = rollout_control_loop(
            env,
            control_policy,
            srb_env,
            design_objective_calculator,
            design_config.n_control_iter,
            headless=env.headless,
            modify_priv_obs=False
        )

    # total_design_objective shape: (num_envs,) = (pop_size,)
    fitness_values = total_design_objective.cpu().numpy().tolist()
    # Reuse aggregate objective terms for each candidate (per-candidate breakdown not available)
    objective_terms_list = [objective_term_sums for _ in range(pop_size)]

    return fitness_values, objective_terms_list, candidates_raw.cpu().numpy()


def compute_surrogate_gradient(
    normalized_params: np.ndarray,
    env,
    control_policy,
    srb_env: MupsRobot,
    design_objective_calculator: DesignObjective,
    design_space: DesignSpace,
    design_config: CodesignConfig,
):
    """
    Compute gradient of the surrogate (SRB) objective at a single design point.
    Uses the differentiable surrogate dynamics with state-alignment trick to
    backpropagate through the non-differentiable IsaacGym simulator.

    Args:
        normalized_params: Design parameters in normalized space, shape (num_params,)

    Returns:
        gradient: Gradient in normalized space, shape (num_params,)
        loss_value: Scalar surrogate loss at this point
    """
    # Update design_space's internal parameter (this is the leaf for autograd)
    with torch.no_grad():
        design_space.active_normalized_param_values.copy_(
            torch.tensor(normalized_params, dtype=design_config.dtype, device=design_config.device)
        )
    design_space.project_active_params_into_bounds()  # Ensure params are within bounds

    # Enable grad for the normalized parameter leaf
    design_space.active_normalized_param_values.requires_grad_(True)

    # Get param values (creates computation graph through design_space)
    param_names = design_space.active_param_names
    param_values = design_space.active_param_values  # (num_params,) — differentiable
    param_values_detached = design_space.detached_active_param_values  # for IsaacGym (non-diff)

    # Set design params: IsaacGym uses detached values, SRB uses differentiable values
    env.set_design_params({name: val for name, val in zip(param_names, param_values_detached.detach())})
    srb_env.set_design_params(
        param_names,
        param_values.unsqueeze(0).expand(design_config.num_envs, -1)
    )

    with torch.no_grad():
        env.reset()

    # Rollout with gradient tracking through surrogate (modify_priv_obs=True)
    total_design_objective, _ = rollout_control_loop(
        env,
        control_policy,
        srb_env,
        design_objective_calculator,
        design_config.n_control_iter,
        headless=env.headless,
        modify_priv_obs=True
    )

    # Backpropagate through surrogate dynamics
    loss = total_design_objective.mean()
    loss.backward()

    gradient = None
    if design_space.active_normalized_param_values.grad is not None:
        gradient = design_space.active_normalized_param_values.grad.detach().cpu().numpy().copy()
    else:
        raise ValueError("Gradient is None from AD — check if computation graph is connected")

    # Clean up: zero grad and disable grad tracking for next call
    design_space.active_normalized_param_values.grad = None
    design_space.active_normalized_param_values.requires_grad_(False)

    return gradient, loss.item()
