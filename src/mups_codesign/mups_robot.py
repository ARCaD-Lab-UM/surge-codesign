"""Include MUPS robot dynamics and kinematics."""

import pdb

import torch

from mups_codesign.config import CodesignConfig
from mups_codesign.design_space import DesignSpace
from mups_codesign.mups_spring import MupsSpring


class MupsRobot:
    def __init__(self, config: CodesignConfig):
        # Basic parameters
        self.num_env = config.num_envs
        self.device = config.device
        self.dtype = config.dtype
        self.dt = config.dt

        # Physical parameters
        self.mass = 2.5
        self.gravity = 9.81
        self.inertia = 0.05
        self.l1 = 0.22
        self.l2 = 0.22

        # RL policy parameters
        self.kp = 40
        self.kd = 1
        self.default_dof_pos = torch.tensor([[0.9, -1.8]], device=self.device, dtype=self.dtype)
        self.action_scale = 0.25
        self.action_limit = 6.0
        self.torque_limit = 35.0

        self.mups_spring = MupsSpring(config)
        self.design_param_scales = torch.tensor(DesignSpace.PARAM_VALUES, device=self.device, dtype=self.dtype).unsqueeze(0)  # (1, num_params)
        self.design_param_values = torch.stack(list(self.mups_spring.design_param_dict.values()), dim=1)  # (num_envs, num_params)
        self.normalized_design_params = self.design_param_values / self.design_param_scales  # (num_envs, num_params)

    def set_design_params(self, param_names, param_values):
        # param_values shape: (num_envs, active_dim) or (1, active_dim)
        self.mups_spring.update_design_param_dict(param_names, param_values, print_info=False)

        # Since python dict preserve insertion order, we can convert it to tensor in the correct order
        self.design_param_values = torch.stack(list(self.mups_spring.design_param_dict.values()), dim=1)  # (num_envs, num_params)
        self.normalized_design_params = self.design_param_values / self.design_param_scales  # (num_envs, num_params)

    def _get_rot_mat_y(self, theta):
        c = torch.cos(theta)
        s = torch.sin(theta)
        rot_mat = torch.zeros((self.num_env, 3, 3), device=self.device, dtype=self.dtype)
        rot_mat[:, 0, 0] = c
        rot_mat[:, 0, 1] = 0
        rot_mat[:, 0, 2] = s
        rot_mat[:, 1, 0] = 0
        rot_mat[:, 1, 1] = 1
        rot_mat[:, 1, 2] = 0
        rot_mat[:, 2, 0] = -s
        rot_mat[:, 2, 1] = 0
        rot_mat[:, 2, 2] = c
        return rot_mat

    def _calc_joint_pd_torques(self, dof_pos, dof_vel, action):
        # Clip and scale actions
        clipped_action = torch.clip(action, -self.action_limit, self.action_limit)
        scaled_action = clipped_action * self.action_scale

        # Calculate torques from joint PD
        torque = self.kp * (scaled_action + self.default_dof_pos - dof_pos) - self.kd * dof_vel
        clipped_torque = torch.clip(torque, -self.torque_limit, self.torque_limit)

        return clipped_torque

    def _calc_foot_position(self, pos, pitch, dof_pos):
        q1 = dof_pos[:, 0]  # hip
        q2 = dof_pos[:, 1]  # knee
        
        # Calculate foot position
        foot_pos_body = torch.zeros((self.num_env, 3), device=self.device, dtype=self.dtype)
        foot_pos_body[:, 0] = - self.l1 * torch.sin(q1) - self.l2 * torch.sin(q1 + q2)
        foot_pos_body[:, 2] = - self.l1 * torch.cos(q1) - self.l2 * torch.cos(q1 + q2)

        rot_mat = self._get_rot_mat_y(pitch)
        foot_pos_world = pos + torch.bmm(rot_mat, foot_pos_body.unsqueeze(-1)).squeeze(-1)

        return foot_pos_world

    def _calc_foot_jacobian(self, dof_pos):
        q1 = dof_pos[:, 0]  # hip
        q2 = dof_pos[:, 1]  # knee

        foot_jac = torch.zeros((self.num_env, 2, 2), device=self.device, dtype=self.dtype)
        foot_jac[:, 0, 0] = - self.l1 * torch.cos(q1) - self.l2 * torch.cos(q1 + q2)
        foot_jac[:, 0, 1] = - self.l2 * torch.cos(q1 + q2)
        foot_jac[:, 1, 0] = self.l1 * torch.sin(q1) + self.l2 * torch.sin(q1 + q2)
        foot_jac[:, 1, 1] = self.l2 * torch.sin(q1 + q2)

        return foot_jac

    def step_srb_dynamics(self, root_state, dof_state, action):
        # TODO vz and wy is inaccurate by a lot

        # Parse inputs
        pos = root_state[:, :3]  # (x, y, z)
        quat = root_state[:, 3:7]  # (x, y, z, w)
        lin_vel = root_state[:, 7:10]  # (vx, vy, vz)
        ang_vel = root_state[:, 10:13]  # (wx, wy, wz)
        dof_pos = dof_state[:, :2]  # (hip, knee)
        dof_vel = dof_state[:, 2:4]  # (hip, knee)

        pitch = torch.arcsin(2.0 * (quat[:, 3] * quat[:, 1]))

        motor_torque = self._calc_joint_pd_torques(dof_pos, dof_vel, action)

        # * Caution: spring torque depends on design parameters
        spring_torque = self.mups_spring.calc_spring_torque(dof_pos)

        joint_torque = motor_torque + spring_torque
        clipped_torque = torch.clip(joint_torque, -self.torque_limit, self.torque_limit)

        foot_pos = self._calc_foot_position(pos, pitch, dof_pos) # (num_envs, 3)
        foot_jac = self._calc_foot_jacobian(dof_pos) # (num_envs, 2, 2)

        is_contact = (foot_pos[:, 2] <= 0.008).unsqueeze(-1) # (num_envs, 1), True if in contact

        foot_force = - torch.linalg.solve(foot_jac.transpose(1, 2), clipped_torque.unsqueeze(-1)).squeeze(-1) * is_contact # (num_envs, 2)
        self.foot_force = foot_force

        # Update 2D SRB dynamics
        com_acc = torch.zeros((self.num_env, 3), device=self.device, dtype=self.dtype)
        com_acc[:, 0] = foot_force[:, 0] / self.mass
        com_acc[:, 1] = 0.0
        com_acc[:, 2] = (foot_force[:, 1] - self.mass * self.gravity) / self.mass

        r_vec = foot_pos - pos
        moment = (r_vec[:, 2] * foot_force[:, 0] - r_vec[:, 0] * foot_force[:, 1])
        ang_acc = moment / self.inertia

        # Create new tensors instead of modifying in-place
        new_lin_vel = lin_vel + com_acc * self.dt
        new_ang_vel = ang_vel.clone()
        new_ang_vel[:, 1] = ang_vel[:, 1] + ang_acc * self.dt  # only pitch
        new_pos = pos + new_lin_vel * self.dt
        new_pitch = pitch + new_ang_vel[:, 1] * self.dt

        cp = torch.cos(new_pitch / 2.0)
        sp = torch.sin(new_pitch / 2.0)
        
        # Create new quaternion instead of modifying in-place
        new_quat = quat.clone()
        new_quat[:, 1] = sp
        new_quat[:, 3] = cp

        srb_state = torch.hstack([new_pos, new_quat, new_lin_vel, new_ang_vel])
        assert srb_state.shape == root_state.shape, "SRB state shape mismatch"

        info = {
            "foot_pos": foot_pos,
            "is_contact": is_contact,
            "foot_force": foot_force,
            "com_acc": com_acc,
            "ang_acc": ang_acc,
        }

        return srb_state, motor_torque, info
