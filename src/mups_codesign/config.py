"""
Dataclass configs for: design space, optimizer (GD + CMA-ES), logging.
"""

import torch

from dataclasses import dataclass

@dataclass
class CodesignConfig:
    # General config
    num_envs: int = 1024
    device: str = "cuda:0"
    dtype: str = torch.float32

    # Design space config
    active_dim: int = 2  # Number of active design parameters

    # MUPS spring config
    softplus_beta: float = 1.0
    softplus_threshold: float = 20.0

    # Design objective config
    use_log1p: bool = True
    dt: float = 0.02  # Time step

    # Logging config
    log_dir: str = "logs/"
