"""
Single source of truth for design parameters: names, defaults, bounds, scaling, transforms (log/linear), constraints.
"""

import torch
import torch.nn as nn

from mups_codesign.config import CodesignConfig


class DesignSpace:
    PARAM_NUMS = 4
    PARAM_NAMES = (
        "ups_ks",  # Spring stiffness (N/m)
        "ups_l0",  # Spring rest length (m)
        "ups_l2",  # Parallel linkage length (m)
        "ups_l4",  # Parallel linkage offset from knee joint (m)
    )
    PARAM_VALUES = (
        4115,   # ups_ks
        0.138,  # ups_l0
        0.1,    # ups_l2
        0.02,   # ups_l4
    )

    def __init__(self, config: CodesignConfig, requires_grad: bool=True):

        # Configurable parameters
        self.device = config.device
        self.dtype = config.dtype

        # Available design parameter names and default values
        self.param_names = self.PARAM_NAMES
        self.default_param_values = torch.tensor(
            self.PARAM_VALUES,
            dtype=self.dtype,
            device=self.device
        )  # (num_params, )

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

        # Parse active design parameters
        self.active_param_names = config.active_param_names
        self.active_param_indices = [self.param_names.index(name) for name in self.active_param_names]

        # Parse initial design parameter values
        init_param_values = self.default_param_values[self.active_param_indices].clone()
        raw_init_param_values = config.raw_init_param_values # tuple or None
        if raw_init_param_values is not None:
            assert len(raw_init_param_values) == self.num_active_params, \
                f"init_param_values shape {len(raw_init_param_values)} does not match active design dimension {self.num_active_params}"
            init_param_values = torch.tensor(
                raw_init_param_values,
                dtype=self.dtype,
                device=self.device
            )

        self.active_normalized_param_values = nn.Parameter(
            init_param_values / self.active_param_scales,
            requires_grad=requires_grad
        )  # (len(active_param_names), )

    @property
    def num_active_params(self):
        return len(self.active_param_names)

    @property
    def active_param_scales(self):
        return self.param_scales[self.active_param_indices]

    @property
    def active_param_values(self):
        return self.active_normalized_param_values * self.active_param_scales
    
    @property
    def detached_active_param_values(self):
        return self.active_param_values.detach()

    @property
    def active_param_bounds(self):
        return self.param_bounds[self.active_param_indices, :]

    @property
    def active_normalized_param_bounds(self):
        return self.normalized_param_bounds[self.active_param_indices, :]

    def project_active_params_into_bounds(self):
        with torch.no_grad():
            self.active_normalized_param_values.clamp_(
                self.active_normalized_param_bounds[:, 0],
                self.active_normalized_param_bounds[:, 1]
            )

    def sample_active_param_values(self):
        """Sample active design parameters uniformly within bounds."""
        rand_vals = torch.rand(
            (self.num_active_params,),
            dtype=self.dtype,
            device=self.device
        )
        lower_bounds = self.active_param_bounds[:, 0]
        upper_bounds = self.active_param_bounds[:, 1]
        sampled_params = lower_bounds + rand_vals * (upper_bounds - lower_bounds)
        return sampled_params
