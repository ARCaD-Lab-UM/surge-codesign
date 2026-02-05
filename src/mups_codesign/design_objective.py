"""
One function that returns objective, components from SRB + policy rollout. No more ad-hoc objective in multiple places.
"""

import torch

from mups_codesign.config import CodesignConfig


class DesignObjective:
    def __init__(self, config: CodesignConfig):

        # Configurable parameters
        self.num_envs = config.num_envs
        self.device = config.device
        self.dtype = config.dtype
        self.use_log1p = config.use_log1p

        # Energy related parameters
        self.dt = config.dt
        self.motor_resistance = 0.17
        self.motor_torque_constant = 0.945

        # Task related parameters
        self.desired_base_height = 0.5

        # Multi-objective weights
        self.objective_weights = {
            "heating_energy": 1.0,
            "mechanical_energy": 0.0,
            "height_tracking_error": 1.0,
        }

    def _calc_energy_consumption(self, dof_state, motor_torque):
        """Calculate energy related objectives

        Args:
            dof_state (tensor): (num_envs, 4)
            motor_torque (tensor): (num_envs, 2)
        """

        # Energy objective with SRB states only
        # total_power = self.foot_force[:, 0] * srb_state[:, 7] + \
        #               self.foot_force[:, 1] * srb_state[:, 9]  # (num_env, )
        # energy = total_power * self.dt  # (num_env, )

        dof_vel = dof_state[:, 2:4]

        mech_power = (motor_torque * dof_vel).sum(dim=-1) # (num_env, )
        positive_mech_power = mech_power.clamp(min=0.0)
        mechanical_energy = positive_mech_power * self.dt
        if self.use_log1p:
            mechanical_energy = torch.log1p(mechanical_energy)

        heat_power = motor_torque.square().sum(dim=-1) * self.motor_resistance / (self.motor_torque_constant**2) # (num_env, )
        heating_energy = heat_power * self.dt
        if self.use_log1p:
            heating_energy = torch.log1p(heating_energy)

        energy_components = {
            "mechanical_energy": mechanical_energy,
            "heating_energy": heating_energy,
        }


        return energy_components

    def _calc_tracking_error(self, srb_state):
        """Calculate tracking error objective

        Args:
            srb_state (tensor): (num_envs, 13)
        """
        base_height = srb_state[:, 2]
        height_error = (base_height - self.desired_base_height).square() # (num_env, )

        tracking_components = {
            "height_tracking_error": height_error,
        }

        return tracking_components

    def calc_objective(self, srb_state, dof_state, motor_torque):
        """Calculate design objective

        Args:
            srb_state (tensor): (num_envs, 13)
            dof_state (tensor): (num_envs, 4)
            motor_torque (tensor): (num_envs, 2)
        """

        # Energy components
        energy_components = self._calc_energy_consumption(dof_state, motor_torque)

        # Tracking components
        tracking_components = self._calc_tracking_error(srb_state)

        # Combine all components
        all_components = {**energy_components, **tracking_components}

        # Weighted sum to get final design objective
        design_objective = torch.zeros((self.num_envs,), device=self.device, dtype=self.dtype)
        for name, component in all_components.items():
            weight = self.objective_weights[name]
            design_objective += weight * component

        return design_objective, all_components
