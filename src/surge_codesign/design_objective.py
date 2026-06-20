import torch

from surge_codesign.config import CodesignConfig


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
        self.desired_base_height = 0.5 + config.hw_height_offset # IMU offset

        # Multi-objective weights
        self.objective_weights = config.objective_weights

    def _calc_mechanical_energy(self, dof_state, motor_torque, **kwargs):
        """Calculate mechanical energy objective

        Args:
            dof_state (tensor): (num_envs, 4)
            motor_torque (tensor): (num_envs, 2)
        """
        dof_vel = dof_state[:, 2:4]

        mech_power = (motor_torque * dof_vel).sum(dim=-1)  # (num_env, )
        positive_mech_power = mech_power.clamp(min=0.0)
        mechanical_energy = positive_mech_power * self.dt
        if self.use_log1p:
            mechanical_energy = torch.log1p(mechanical_energy)

        return mechanical_energy

    def _calc_heating_energy(self, motor_torque, **kwargs):
        """Calculate heating energy objective

        Args:
            motor_torque (tensor): (num_envs, 2)
        """
        heat_power = motor_torque.square().sum(dim=-1) * self.motor_resistance / (self.motor_torque_constant**2)  # (num_env, )
        heating_energy = heat_power * self.dt
        if self.use_log1p:
            heating_energy = torch.log1p(heating_energy)

        return heating_energy

    def _calc_height_tracking_error(self, srb_state, **kwargs):
        """Calculate height tracking error objective

        Args:
            srb_state (tensor): (num_envs, 13)
        """
        base_height = srb_state[:, 2]
        height_error = (base_height - self.desired_base_height).square()  # (num_env, )

        return height_error

    def calc_objective(self, srb_state, dof_state, motor_torque):
        """Calculate design objective

        Args:
            srb_state (tensor): (num_envs, 13)
            dof_state (tensor): (num_envs, 4)
            motor_torque (tensor): (num_envs, 2)
        """

        # Mapping from objective name to calc function
        objective_funcs = {
            "mechanical_energy": self._calc_mechanical_energy,
            "heating_energy": self._calc_heating_energy,
            "height_tracking_error": self._calc_height_tracking_error,
        }

        # Compute only the objectives that have non-zero weights
        all_components = {}
        for name, weight in self.objective_weights.items():
            if weight == 0:
                continue
            calc_func = objective_funcs[name]
            all_components[name] = calc_func(
                srb_state=srb_state,
                dof_state=dof_state,
                motor_torque=motor_torque,
            )

        # Weighted sum to get final design objective
        design_objective = torch.zeros((self.num_envs,), device=self.device, dtype=self.dtype)
        for name, component in all_components.items():
            weight = self.objective_weights[name]
            design_objective += weight * component

        return design_objective, all_components
