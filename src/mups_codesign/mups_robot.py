import pdb
import torch
import numpy as np


class MupsRobot:
    def __init__(self, num_env, device):
        # Basic parameters
        self.num_env = num_env
        self.device = device

        # Physical parameters
        self.mass = 2.5
        self.gravity = 9.81
        self.inertia = 0.05

        self.l1 = 0.22
        self.l2 = 0.22
        self.dt = 0.02

        self.action_scale = 0.25
        self.action_limit = 6.0
        self.torque_limit = 35.0
        self.default_dof_pos = torch.tensor([[0.9, -1.8]], device=device)

        # Motor parameters
        self.kp = 40
        self.kd = 1
        self.motor_resistance = 0.17
        self.motor_torque_constant = 0.945

        # Task parameters
        self.desired_base_height = 0.5

        # Design parameters
        self.ups_ks = 4115  # spring coefficient
        self.ups_l0 = 0.138 # spring resting length
        self.ups_l1 = 0.03  # Parallelogram short side
        self.ups_l2 = 0.1   # PEA link
        self.ups_l3 = 0.22  # Parallelogram long side
        self.ups_l4 = 0.02  # PEA link offset from knee joint
        self.ups_l5 = 0.01  # orthogonal offset from slider
        self.ups_l6 = 0.003 # parallel offset from slider

        self.design_param_scale = torch.tensor(
            [
                self.ups_ks, 
                self.ups_l0,
            ], 
            device=device
        )

        self.design_param_bound = torch.tensor(
            [
                [0.2, 2.2],
                [0.7, 1.2],
            ],
            device=device
        )

        # Objective weighting and compression controls
        self.cost_weights = {
            "positive_mech": 1.0,
            "heat": 1.0,
            "height_error": 1.0,
        }
        self.compress_power_cost = True


    def calc_design_objective(self, srb_state, dof_state, motor_torque, design_param):
        """Calculate design objective

        Args:
            srb_state (num_env, 13) # diff
            dof_state (num_env, 6)  # non-diff
        """
        base_height = srb_state[:, 2]
        dof_vel = dof_state[:, 2:4]

        # Energy objective with full states
        mech_power = (motor_torque * dof_vel).sum(dim=-1) # (num_env, )
        positive_mech_power = mech_power.clamp(min=0.0)
        heat_power = motor_torque.square().sum(dim=-1) * self.motor_resistance / (self.motor_torque_constant**2) # (num_env, )
        if self.compress_power_cost:
            positive_mech_cost = torch.log1p(positive_mech_power)
            heat_cost = torch.log1p(heat_power)
        else:
            positive_mech_cost = positive_mech_power
            heat_cost = heat_power

        energy_cost = (
            self.cost_weights["positive_mech"] * positive_mech_cost
            + self.cost_weights["heat"] * heat_cost
        )  # (num_env, )

        # Energy objective with SRB states only
        # total_power = self.foot_force[:, 0] * srb_state[:, 7] + \
        #               self.foot_force[:, 1] * srb_state[:, 9]  # (num_env, )
        # energy = total_power * self.dt  # (num_env, )

        # Tracking objective
        height_error = (base_height - self.desired_base_height).square()  # (num_env, )
        tracking_cost = self.cost_weights["height_error"] * height_error

        # Final design objective (num_env, )
        design_objective = energy_cost * self.dt + tracking_cost
        # design_objective = height_error

        objective_components = {
            "positive_mech_cost": positive_mech_cost.detach(),
            "heat_cost": heat_cost.detach(),
            "height_error": height_error.detach(),
        }

        return design_objective, objective_components


    def calc_ups_spring_torque(self, dof_pos, design_param):
        spring_torque = torch.zeros((self.num_env, 2), device=self.device)
        q_knee = -dof_pos[:, 1] - torch.pi / 2.0

        Ks = design_param[0]
        l0 = design_param[1]
        l1 = self.ups_l1
        l2 = self.ups_l2
        l3 = self.ups_l3
        l4 = self.ups_l4
        l5 = self.ups_l5
        l6 = self.ups_l6
        t2 = torch.cos(q_knee)
        t3 = torch.sin(q_knee)
        t4 = l1 + l4
        t5 = l2**2
        t6 = t2 * t4
        t7 = - t6
        t8 = l5 + t7
        t9 = t8**2
        t10 = - t9
        t11 = t5 + t10
        softplus = torch.nn.functional.softplus(l0 * 1.0e+3-l3 * 1.0e+3+l6 * 1.0e+3+t3 * t4 * 1.0e+3+torch.sqrt(t11) * 1.0e+3)
        tau = (Ks * softplus * (t6 - t3 * t4 * t8 * 1.0 / torch.sqrt(t11))) / 1.0e+3

        try:
            assert torch.isfinite(tau).all(), "NaN or inf in spring torque calculation"
        except:
            pdb.set_trace()

        spring_torque[:, 1] = tau

        return spring_torque


    def get_rot_mat_y(self, theta):
        c = torch.cos(theta)
        s = torch.sin(theta)
        rot_mat = torch.zeros((self.num_env, 3, 3), device=self.device)
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


    def calc_joint_pd_torques(self, dof_pos, dof_vel, action):
        # Clip and scale actions
        clipped_action = torch.clip(action, -self.action_limit, self.action_limit)
        scaled_action = clipped_action * self.action_scale

        # Calculate torques from joint PD
        torque = self.kp * (scaled_action + self.default_dof_pos - dof_pos) - self.kd * dof_vel
        clipped_torque = torch.clip(torque, -self.torque_limit, self.torque_limit)

        return clipped_torque


    def calc_foot_position(self, pos, pitch, dof_pos):
        q1 = dof_pos[:, 0]  # hip
        q2 = dof_pos[:, 1]  # knee
        
        # Calculate foot position
        foot_pos_body = torch.zeros((self.num_env, 3), device=self.device)
        foot_pos_body[:, 0] = - self.l1 * torch.sin(q1) - self.l2 * torch.sin(q1 + q2)
        foot_pos_body[:, 2] = - self.l1 * torch.cos(q1) - self.l2 * torch.cos(q1 + q2)

        rot_mat = self.get_rot_mat_y(pitch)
        foot_pos_world = pos + torch.bmm(rot_mat, foot_pos_body.unsqueeze(-1)).squeeze(-1)

        return foot_pos_world


    def calc_foot_jacobian(self, dof_pos):
        q1 = dof_pos[:, 0]  # hip
        q2 = dof_pos[:, 1]  # knee

        foot_jac = torch.zeros((self.num_env, 2, 2), device=self.device)
        foot_jac[:, 0, 0] = - self.l1 * torch.cos(q1) - self.l2 * torch.cos(q1 + q2)
        foot_jac[:, 0, 1] = - self.l2 * torch.cos(q1 + q2)
        foot_jac[:, 1, 0] = self.l1 * torch.sin(q1) + self.l2 * torch.sin(q1 + q2)
        foot_jac[:, 1, 1] = self.l2 * torch.sin(q1 + q2)

        return foot_jac


    def step_srb_dynamics(self, root_state, dof_state, action, design_param):
        # TODO vz and wy is inaccurate by a lot

        # Parse inputs
        pos = root_state[:, :3]  # (x, y, z)
        quat = root_state[:, 3:7]  # (x, y, z, w)
        lin_vel = root_state[:, 7:10]  # (vx, vy, vz)
        ang_vel = root_state[:, 10:13]  # (wx, wy, wz)
        dof_pos = dof_state[:, :2]  # (hip, knee)
        dof_vel = dof_state[:, 2:4]  # (hip, knee)

        pitch = torch.arcsin(2.0 * (quat[:, 3] * quat[:, 1]))

        motor_torque = self.calc_joint_pd_torques(dof_pos, dof_vel, action)
        spring_torque = self.calc_ups_spring_torque(dof_pos, design_param)

        joint_torque = motor_torque + spring_torque
        clipped_torque = torch.clip(joint_torque, -self.torque_limit, self.torque_limit)

        foot_pos = self.calc_foot_position(pos, pitch, dof_pos) # (num_envs, 3)
        foot_jac = self.calc_foot_jacobian(dof_pos) # (num_envs, 2, 2)

        is_contact = (foot_pos[:, 2] <= 0.008).unsqueeze(-1) # (num_envs, 1), True if in contact

        foot_force = - torch.linalg.solve(foot_jac.transpose(1, 2), clipped_torque.unsqueeze(-1)).squeeze(-1) * is_contact # (num_envs, 2)
        self.foot_force = foot_force

        # Update 2D SRB dynamics
        com_acc = torch.zeros((self.num_env, 3), device=self.device)
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

        cp = torch.cos(new_pitch / 2.0) # noooo hank dont abbreviate torch.cos(pitch)
        sp = torch.sin(new_pitch / 2.0)
        
        # Create new quaternion instead of modifying in-place
        new_quat = quat.clone()
        new_quat[:, 1] = sp
        new_quat[:, 3] = cp

        srb_state = torch.hstack([new_pos, new_quat, new_lin_vel, new_ang_vel])

        # assert srb_state.shape == root_state.shape, "SRB state shape mismatch"

        design_objective, objective_components = self.calc_design_objective(srb_state, dof_state, motor_torque, design_param)

        return srb_state, motor_torque, design_objective, objective_components
