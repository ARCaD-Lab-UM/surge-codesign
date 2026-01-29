"""
Single source of truth for design parameters: names, defaults, bounds, scaling, transforms (log/linear), constraints.
"""

import torch
import torch.nn as nn
from torch import Tensor

from mups_codesign.config import CodesignConfig


class DesignSpace:
    def __init__(self, config: CodesignConfig, init_param_values: Tensor=None, requires_grad: bool=True):

        # Available design parameters
        self.param_names = [
            "ups_ks",  # Spring stiffness (N/m)
            "ups_l0",  # Spring rest length (m)
            "ups_l2",  # Parallel linkage length (m)
            "ups_l4",  # Parallel linkage offset from knee joint (m)
        ]

        assert config.active_dim >= 1, "Active design dimension must be at least 1."
        assert config.active_dim <= len(self.param_names), "Active design dimension cannot exceed total number of design parameters."

        # Configurable parameters
        self.device = config.device
        self.dtype = config.dtype
        self.active_dim = config.active_dim

        self.default_param_values = torch.tensor(
            [
                4115,   # ups_ks
                0.138,  # ups_l0
                0.1,    # ups_l2
                0.02,   # ups_l4
            ],
            dtype=self.dtype,
            device=self.device
        )

        if init_param_values is None:
            init_param_values = self.default_param_values[:self.active_dim]

        assert init_param_values.shape == (self.active_dim,), "Initial parameters shape mismatch."

        self.normalized_param_bounds = torch.tensor(
            [
                [0.2, 2.2],
                [0.7, 1.2],
                [0.8, 1.4],
                [1.0, 2.0],
            ],
            dtype=self.dtype,
            device=self.device
        )
        self.param_bounds = self.normalized_param_bounds * self.default_param_values.unsqueeze(1)
        self.param_scales = self.default_param_values.clone()

        # Active design parameters
        converted_init_param_values = init_param_values.to(self.device, self.dtype)
        self.active_normalized_param_values = nn.Parameter(
            converted_init_param_values / self.active_param_scales,
            requires_grad=requires_grad
        )  # (active_dim, )

    @property
    def active_param_names(self):
        return self.param_names[:self.active_dim]

    @property
    def active_param_bounds(self):
        return self.param_bounds[:self.active_dim, :]

    @property
    def active_param_scales(self):
        return self.param_scales[:self.active_dim]

    @property
    def active_param_values(self):
        return self.active_normalized_param_values * self.active_param_scales

    @property
    def active_normalized_param_bounds(self):
        return self.normalized_param_bounds[:self.active_dim, :]

    def project_active_params_into_bounds(self):
        with torch.no_grad():
            self.active_normalized_param_values.clamp_(
                self.active_normalized_param_bounds[:, 0],
                self.active_normalized_param_bounds[:, 1]
            )
