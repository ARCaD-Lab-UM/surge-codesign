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
        "height_tracking_error": 0.0,
    }


@dataclass
class CodesignConfig:
    # General config
    num_envs: int = 1024 # TODO: this should not be overwritten by isaacgym num_envs
    device: str = "cuda:0"
    dtype: str = torch.float32

    # Design space config
    active_param_names: tuple = ("ups_ks", "ups_l0")
    raw_init_param_values: tuple = None  # if None, use default param values

    # MUPS spring config
    softplus_beta: float = 1.0
    softplus_threshold: float = 20.0

    # Design objective config
    use_log1p: bool = True
    dt: float = 0.02  # Time step
    objective_weights: Dict[str, float] = field(
        default_factory=default_objective_weights
    )

    # Logging config
    log_dir: str = "logs/"
