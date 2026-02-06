import pdb
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import quat_to_euler

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import numpy as np
import os

class HopperRobot(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.set_design_params()
        self.debug_viz = True

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure
        """
        # Build noise vector
        noise_vec = torch.zeros(self.cfg.env.num_proprio, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # ==========================================================================
        noise_vec[0:1] = noise_scales.xyz_pos * noise_level * self.obs_scales.xyz_pos
        noise_vec[1:4] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[4:7] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[7:10] = noise_scales.gravity * noise_level
        noise_vec[10:13] = 0. # commands
        noise_vec[13 : 13+cfg.env.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[13+cfg.env.num_actions : 13+cfg.env.num_actions*2] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[13+cfg.env.num_actions*2 : 13+cfg.env.num_actions*3] = 0. # previous actions
        # ==========================================================================
        return noise_vec

    def compute_observations(self):
        """ Computes observations for the robot. Overloaded to include unique observations
        """
        sin_phase = torch.sin(2*np.pi*self.phase)
        cos_phase = torch.cos(2*np.pi*self.phase)
        phase_features = torch.stack([sin_phase, cos_phase], dim=1)

        # CUR OBS    
        cur_obs_buf = torch.cat((self.root_states[:, 2:3] * self.obs_scales.xyz_pos,                 # (1,) z
                                 self.base_lin_vel * self.obs_scales.lin_vel,                        # (3,) linear vel
                                 self.base_ang_vel  * self.obs_scales.ang_vel,                       # (3,)
                                 self.projected_gravity,                                             # (3,)
                                 self.commands[:, :3] * self.commands_scale,                         # (3,)  
                                 (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,    # (2,)
                                 self.dof_vel * self.obs_scales.dof_vel,                             # (2,)
                                 self.actions                                                        # (2,) last actions
                                ),dim=-1)                                                          
        
        # PHASE FEATURES
        cur_obs_buf = torch.cat([cur_obs_buf, phase_features], dim=1)

        # NOISE
        if self.add_noise:
            cur_obs_buf += (2 * torch.rand_like(cur_obs_buf) - 1) * self.noise_scale_vec
        
        # HISTORY OBS (concatenate)
        self.obs_buf = torch.cat([
            self.obs_history_buf.view(self.num_envs, -1),  # Flattened history
            cur_obs_buf                                    # Current observation
        ], dim=-1)

        # PRIVILEGED OBS
        self.privileged_obs_buf = torch.cat((
            self.privileged_mass_params,        # 4
            self.privileged_friction_coeffs,    # 1
            self.kp_kd_multipliers[0] - 1,      # 2
            self.kp_kd_multipliers[1] - 1,      # 2
            self.design_params[:, :2] * self.design_params_scale[:, :2]), #ks, l0
            dim=-1
        )
        
        # ESTIMATED OBS
        self.estimated_obs_buf = torch.zeros_like(self.base_lin_vel)
        
        # CRITIC OBS
        self.critic_obs_buf = torch.cat((
            self.obs_buf.clone().detach(),
            self.privileged_obs_buf.clone().detach(),
            self.estimated_obs_buf.clone().detach(),
        ), dim=-1)
        
        # Update the history buffer   
        self.obs_history_buf = torch.where((
            self.episode_length_buf <= 1)[:, None, None],
            torch.stack([cur_obs_buf] * (self.cfg.env.history_buffer_length), dim=1),
            torch.cat([self.obs_history_buf[:, 1:], cur_obs_buf.unsqueeze(1)], dim=1)
        )

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
            Positions are randomly selected within 0.5:1.5 x default positions.
            Velocities are set to zero.

        Args:
            env_ids (List[int]): Environment ids
        """
        # Set default dof positions for all environments
        self.dof_pos[env_ids] = self.default_dof_pos

        # Set dof velocities to zero
        self.dof_vel[env_ids] = 0.

        # Do whatever this function is supposed to do
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # Reset robot to base positions
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state           # value from cfg.init_state.pos
            self.root_states[env_ids, :3] += self.env_origins[env_ids] # center of that env
            # self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # Reset base velocities (randomize them...)
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

        # Do stuff
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        
        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # update feet contact variables
        self.phase = (self.episode_length_buf * self.dt) % self.cfg.env.period / self.cfg.env.period 
        self.foot_in_contact = self.contact_forces[:, self.feet_indices[0], 2] > 1.0

        # run post physics step callback
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)     # calls resample_commands ...
        self.compute_observations()

        # Update buffers that store 'previous' data
        self.last_actions[:] = self.actions[:]              # Update prev. actions
        self.last_dof_vel[:] = self.dof_vel[:]              # Update prev. dof velocity
        self.last_root_vel[:] = self.root_states[:, 7:13]   # Update prev. root velocity
        self.last_base_lin_vel[:] = self.base_lin_vel[:]    # Update prev. base linear velocity (NEW)
        self.last_torques[:] = self.torques[:]              # Update prev. torques (NEW)

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        self.root_states[:, [8, 10, 12]] *= 0 # Zero out the y, roll and yaw rates of the root states
        self.root_states[:, 1] = self.env_origins[:, 1] # Reset the y position of the root states to the env origins
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def set_design_params(self, params=None):
        """
        params: (num_envs, num_params)
        """
        # Nominal design parameters
        ks = 4115  # spring coefficient
        l0 = 0.138 # spring resting length
        l1 = 0.03  # Parallelogram short side
        l2 = 0.1   # PEA link
        l3 = 0.22  # Parallelogram long side
        l4 = 0.02  # PEA link offset from knee joint
        l5 = 0.01  # orthogonal offset from slider
        l6 = 0.003 # parallel offset from slider

        num_params = 0
        if params is not None:
            num_params = params.shape[1]
            assert(num_params <= 2, f"Expected ks or ks and l0, got {num_params} parameters.")

        # Overwrite design parameters if provided
        self.design_params = torch.zeros(self.num_envs, 8, device=self.device)
        self.design_params[:, 0] = ks if num_params < 1 else params[:, 0]
        self.design_params[:, 1] = l0 if num_params < 2 else params[:, 1]
        self.design_params[:, 2] = l1
        self.design_params[:, 3] = l2 if num_params < 3 else params[:, 2]
        self.design_params[:, 4] = l3
        self.design_params[:, 5] = l4 if num_params < 4 else params[:, 3]
        self.design_params[:, 6] = l5
        self.design_params[:, 7] = l6

        self.design_params_scale = torch.ones(self.num_envs, 8, device=self.device)
        self.design_params_scale[:, 0] = 1 / ks # ks
        self.design_params_scale[:, 1] = 1 / l0    # l0

        #! TODO only sweep params during training
        # # Random sample ks, l0, l2, l4 from their respective bounds
        # params_bounds = np.array([
        #     [ks * 0.2, ks * 2.2],
        #     [l0 * 0.7, l0 * 1.2],
        #     [l2 * 0.8, l2 * 1.4],
        #     [l4 * 1.0, l4 * 2.0],
        # ])

        # # Random sample all 4 parameters
        # ks_samples = np.random.uniform(params_bounds[0, 0], params_bounds[0, 1], size=self.num_envs)
        # l0_samples = np.random.uniform(params_bounds[1, 0], params_bounds[1, 1], size=self.num_envs)
        # l2_samples = np.random.uniform(params_bounds[2, 0], params_bounds[2, 1], size=self.num_envs)
        # l4_samples = np.random.uniform(params_bounds[3, 0], params_bounds[3, 1], size=self.num_envs)

        # self.design_params[:, 0] = torch.from_numpy(ks_samples).to(self.device)  # ks
        # self.design_params[:, 1] = torch.from_numpy(l0_samples).to(self.device)  # l0
        # self.design_params[:, 3] = torch.from_numpy(l2_samples).to(self.device)  # l2
        # self.design_params[:, 5] = torch.from_numpy(l4_samples).to(self.device)  # l4


    def get_spring_torque_on_knee(self):
        q_knee = -self.dof_pos[:, 1] - torch.pi / 2.0

        Ks = self.design_params[:, 0]
        l0 = self.design_params[:, 1]
        l1 = self.design_params[:, 2]
        l2 = self.design_params[:, 3]
        l3 = self.design_params[:, 4]
        l4 = self.design_params[:, 5]
        l5 = self.design_params[:, 6]
        l6 = self.design_params[:, 7]
        t2 = torch.cos(q_knee)
        t3 = torch.sin(q_knee)
        t4 = l1 + l4
        t5 = torch.square(l2)
        t6 = t2 * t4
        t7 = - t6
        t8 = l5 + t7
        t9 = torch.square(t8)
        t10 = - t9
        t11 = t5 + t10
        softplus = torch.nn.functional.softplus(l0 * 1.0e+3-l3 * 1.0e+3+l6 * 1.0e+3+t3 * t4 * 1.0e+3+torch.sqrt(t11) * 1.0e+3)
        tau = (Ks * softplus * (t6 - t3 * t4 * t8 * 1.0 / torch.sqrt(t11))) / 1.0e+3

        try:
            assert torch.isfinite(tau).all(), "NaN or inf in spring torque calculation"
        except:
            pdb.set_trace()

        return tau

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # Clip actions to specified range (from cfg)
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        
        # Step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            # Get motor torques from joint PD control
            self.torques = self._compute_torques(self.actions).view(self.torques.shape) # NOTE: Compute_torques multiplies actions by action scale
            
            # ! Clip torques to actual motor limits
            # self.torques = torch.clip(self.torques, -23, 23)

            # Get spring torques from parallel spring on the knee
            spring_torques = torch.zeros_like(self.torques)
            spring_torques[:, 1] = self.get_spring_torque_on_knee()
            # print(f"Knee motor torques: {self.torques[:10, 1]}\n Knee spring torques: {spring_torques[:10, 1]}")
            # ! Clip to URDF limits, otherwise nan dof pos may occur
            clipped_torques = torch.clip(self.torques + spring_torques, -self.torque_limits, self.torque_limits)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(clipped_torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        # Updates a lot of quantities, recomputes obs, invokes physics callback
        self.post_physics_step()

        # Clip observations
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.critic_obs_buf = torch.clip(self.critic_obs_buf, -clip_obs, clip_obs)
        self.estimated_obs_buf = torch.clip(self.estimated_obs_buf, -clip_obs, clip_obs)

        # DO NOT CLIP SCAN OBS

        # Return observations, privileged obs, rewards, reset flags, extras
        return self.obs_buf, self.privileged_obs_buf, self.critic_obs_buf, self.estimated_obs_buf, self.scan_obs_buf, self.rew_buf, self.reset_buf, self.extras


    def _draw_debug_vis(self):
        """ Draws a big ball at base_height_target above each robot
        """
        if not self.viewer:
            return
            
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # create sphere wireframe
        sphere_geom = gymutil.WireframeSphereGeometry(radius=0.02, num_lats=4, num_lons=4, pose=None, color=(0, 1, 0)) 
        
        for i in range(self.num_envs):
            # robot i's xyz
            base_pos = self.base_init_state[:3] + self.env_origins[i]
            base_pos = base_pos.cpu().numpy()
            
            # position sphere above robot spawn
            target_pos = gymapi.Vec3(
                base_pos[0],  # x
                base_pos[1],  # y
                self.cfg.rewards.base_height_target
            )
            # create transform and draw
            sphere_pose = gymapi.Transform(target_pos, r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)


    def _reward_delta_torques(self):
        """ Penalize changes in torques
        """
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    

    def _reward_dof_error(self):
        """ Penalize DOF positions away from default
        """
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error
    
    def _reward_lin_vel_z_up(self):
        """ Encourage z axis base linear velocity upwards
        """
        zero_mask = (self.root_states[:, 2] < self.cfg.rewards.base_height_target).float() # only if we are below base height target
        return torch.square(torch.clip(self.base_lin_vel[:, 2], min=0.0)) * zero_mask

    def _reward_lin_vel_z_up_with_penalty(self):
        """ Encourage upwards linear velocity along z-axis below the height target.
            Penalize upwards linear velocity along z-axis above the height target.
        """
        zero_mask = (self.root_states[:, 2] < self.cfg.rewards.base_height_target).float()
        vel_sq_below_height_target = torch.square(torch.clip(self.base_lin_vel[:, 2], min=0.0)) * zero_mask
        vel_sq_above_height_target = torch.square(torch.clip(self.base_lin_vel[:, 2], min=0.0)) * (1 - zero_mask)
        return vel_sq_below_height_target - vel_sq_above_height_target

    def _reward_hip_error(self):
        """ Penalize DOF positions away from default
        """
        dof_error = torch.square(self.dof_pos[:, 0] - self.default_dof_pos[:, 0])
        return dof_error
    
    def _reward_height_penalty(self):
        """ Penalize exceeding base height target
        """
        zero_mask = (self.root_states[:, 2] >= self.cfg.rewards.base_height_target).float() # mask for being above base height target
        error = torch.clip(self.root_states[:, 2] - self.cfg.rewards.base_height_target, min=0.0) # how much above the target we are
        return error * zero_mask
    
    def _reward_tracking_target_height(self):
        """ Tracking of target height using a Gaussian reward function
        """
        height_error = torch.square(self.root_states[:, 2] - self.cfg.rewards.base_height_target)
        return torch.exp(-height_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_phase_contact_match(self):
        """ Hello my old friend
        """

        # calculate duty cycle
        PERCENT_TIME_ON_GROUND = self.cfg.env.persent_time_on_ground
        stance_threshold = 2.0 * PERCENT_TIME_ON_GROUND - 1.0

        # mask and calculate reward
        stance_mask = torch.sin(2*np.pi*self.phase) <= stance_threshold
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        reward += torch.where(~(self.foot_in_contact ^ stance_mask), 1.0, -1.0)
        return reward