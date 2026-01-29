"""
Implementation of UPS spring torque calculations and default parameters.
The calculation is parallelized over multiple environments.
"""

import pdb
import torch

from mups_codesign.config import CodesignConfig
from mups_codesign.design_space import DesignSpace


class MupsSpring:
    def __init__(self, config: CodesignConfig):

        # Configurable parameters
        self.num_envs = config.num_envs
        self.device = config.device
        self.dtype = config.dtype
        self.softplus_beta = config.softplus_beta
        self.softplus_threshold = config.softplus_threshold

        # Fixed parameters
        self.ups_l1 = 0.03  # Parallelogram short side (m)
        self.ups_l3 = 0.22  # Parallelogram long side (m)
        self.ups_l5 = 0.01  # Orthogonal offset from slider (m)
        self.ups_l6 = 0.003 # Parallel offset from slider (m)

        # Fetch default changeable parameters and build the param dict
        names = DesignSpace.PARAM_NAMES # tuple of str
        values = DesignSpace.PARAM_VALUES # tuple of float
        assert len(names) == len(values), "DesignSpace PARAM_NAMES/PARAM_VALUES length mismatch"
        self.design_param_dict = {name: value for name, value in zip(names, values)}

        # Vectorize param dict to tensor in shape (num_envs, num_params)
        for key in self.design_param_dict.keys():
            self.design_param_dict[key] = torch.full(
                (self.num_envs,),
                self.design_param_dict[key],
                dtype=self.dtype,
                device=self.device
            )

    def update_design_param_dict(self, param_names, param_values, print_info=False):
        """Set UPS spring parameters from design optimization.

        Args:
            param_names (list of str): List of parameter names to set.
            param_values (tensor): (num_envs, num_params) Tensor of parameter values.
        """
        for i, name in enumerate(param_names):
            if name in self.design_param_dict:
                self.design_param_dict[name] = param_values[:, i]
                if print_info:
                    print(f"Set {name} to {self.design_param_dict[name]}")
            else:
                raise ValueError(f"Unknown UPS parameter name: {name}")

    def calc_spring_torque(self, dof_pos):
        """Calculate spring torque and store it in knee index of torque array.

        Args:
            dof_pos (tensor): (num_env, 2)
            design_param (tensor): (num_env, num_design)

        Returns:
            spring_torque: (num_env, 2)
        """

        # Build spring torque based on knee joint angle
        spring_torque = torch.zeros((self.num_envs, 2), device=self.device, dtype=self.dtype)
        knee_angle = -dof_pos[:, 1] - torch.pi / 2.0

        # Retrieve spring parameters from design
        ks = self.design_param_dict["ups_ks"]
        l0 = self.design_param_dict["ups_l0"]
        l1 = self.ups_l1
        l2 = self.design_param_dict["ups_l2"]
        l3 = self.ups_l3
        l4 = self.design_param_dict["ups_l4"]
        l5 = self.ups_l5
        l6 = self.ups_l6

        t2 = torch.cos(knee_angle)
        t3 = torch.sin(knee_angle)
        t4 = l1 + l4
        t5 = torch.square(l2)
        t6 = t2 * t4
        t7 = -t6
        t8 = l5 + t7
        t9 = torch.square(t8)
        t10 = -t9
        t11 = t5 + t10
        softplus = torch.nn.functional.softplus(
            l0 * 1.0e3
            - l3 * 1.0e3
            + l6 * 1.0e3
            + t3 * t4 * 1.0e3
            + torch.sqrt(t11) * 1.0e3,
            beta=self.softplus_beta,
            threshold=self.softplus_threshold
        )
        tau = (ks * softplus * (t6 - t3 * t4 * t8 * 1.0 / torch.sqrt(t11))) / 1.0e+3

        try:
            assert torch.isfinite(tau).all(), "NaN or inf in spring torque calculation"
        except:
            pdb.set_trace()

        spring_torque[:, 1] = tau

        return spring_torque
