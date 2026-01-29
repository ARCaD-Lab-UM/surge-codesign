"""
Single source of truth for design parameters: names, defaults, bounds, scaling, transforms (log/linear), constraints.
"""

import torch


class DesignSpace:
    def __init__(self, active_dim=2, device="cpu", dtype=torch.float32):

        # Available design parameters
        self.param_names = [
            "ups_ks",  # Spring stiffness (N/m)
            "ups_l0",  # Spring rest length (m)
            "ups_l2",  # Parallel linkage length (m)
            "ups_l4",  # Parallel linkage offset from knee joint (m)
        ]

        # Configurable parameters
        self.device = device
        self.dtype = dtype
        self.active_dim = active_dim

        assert active_dim >= 1, "Active design dimension must be at least 1."
        assert active_dim <= len(self.param_names), "Active design dimension cannot exceed total number of design parameters."

        self.default_params = torch.tensor(
            [
                4115,   # ups_ks
                0.138,  # ups_l0
                0.1,    # ups_l2
                0.02,   # ups_l4
            ],
            dtype=self.dtype,
            device=self.device
        )

        self.param_bounds = torch.tensor(
            [
                [0.2, 2.2],
                [0.7, 1.2],
                [0.8, 1.4],
                [1.0, 2.0],
            ],
            dtype=self.dtype,
            device=self.device
        ) * self.default_params.unsqueeze(1)

        self.param_scales = self.default_params.clone()

    def get_active_param_names(self):
        return self.param_names[:self.active_dim]

    def get_active_param_bounds(self):
        return self.param_bounds[:self.active_dim, :]
    
    def get_active_param_scales(self):
        return self.param_scales[:self.active_dim]
