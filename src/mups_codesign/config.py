"""
Dataclass configs for: design space, optimizer (GD + CMA-ES), logging.
"""

from dataclasses import dataclass, field
from typing import Dict

import torch


def default_objective_weights():
    return {
        "heating_energy": 1.0,
        "mechanical_energy": 0.0,
        "height_tracking_error": 5.0,
    }


@dataclass
class CodesignConfig:
    # General config
    seed: int = 12  # NOTE: seed=0 is treated as "no seed" by pycma (falsy check), use non-zero seed instead
    num_envs: int = 1024
    device: str = "cuda:0"      # this will be overwritten by isaacgym env.device
    dtype: str = torch.float32  # this only used internally for codesign modules

    # RL policy config
    policy_id: str = "rainbow_v7"    # trained with ks, l0, l2, l4 and all in privileged obs
    policy_root: str = "checkpoints" # use "logs/<exp_name>" for newly trained policies

    # Optimizer config
    learning_rate: float = None
    n_design_iter: int = None
    n_control_iter: int = None

    # Design space config

    # 4D codesign config
    active_param_names: tuple = ("ups_ks", "ups_l0", "ups_l2", "ups_l4") # if None, use all parameters in design space
    raw_init_param_values: tuple = (8000, 0.11, 0.13, 0.02)  # if None, use default param values

    # 2D hardware sweep config
    # active_param_names: tuple = ("ups_ks", "ups_l0")
    # raw_init_param_values: tuple = (8000, 0.1)

    # MUPS spring config
    softplus_beta: float = 1.0
    softplus_threshold: float = 20.0

    # Design objective config
    use_log1p: bool = True
    dt: float = 0.02  # Time step
    objective_weights: Dict[str, float] = field(
        default_factory=default_objective_weights
    )
    hw_height_offset: float = 0.0  # Height offset to account for IMU position in hardware (added to desired height)

    # Logging config
    log_dir: str = "logs/"
