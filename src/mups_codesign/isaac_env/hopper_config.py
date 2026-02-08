from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class HopperCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 1024
        num_proprio = 1+3+3+3+3+2+2+2+2 # (z, lin_vel, ang_vel, proj_grav, vel_cmd, dof_pos, dof_vel, prev_actions, phase_features)
        num_scan_obs = 0
        num_estimated_obs = 1 # lin_vel
        num_actions = 2

        num_privileged_obs = 4+1+2+2+2+2 # (com_pos, friction, kp, kd, design_params)
        history_buffer_length = 10
        num_critic_obs = num_proprio+(num_proprio*history_buffer_length)+num_privileged_obs+num_estimated_obs+num_scan_obs
        num_observations = num_proprio+(num_proprio*history_buffer_length)

        # Gait features
        period = 0.4
        persent_time_on_ground = 1/3

    class terrain( LeggedRobotCfg.terrain ):
        static_friction = 1.0
        dynamic_friction = 1.0

        mesh_type = 'plane'
        measure_heights = True
        curriculum = False
        selected = False

        restitution = 0.8 #!

    class domain_rand:
        randomize_friction = True
        friction_range = [0.4, 1.2]

        randomize_base_mass = True
        added_mass_range = [-0.025, 0.025]
        
        randomize_center_of_mass = True
        added_com_range = [-0.05, 0.05]

        randomize_kp_kd = True
        kp_kd_range = [0.8, 1.2]

        push_robots = False
        push_interval_s = 8
        max_push_vel_xy = 0.5
    

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.3]      # [x, y, z] (metres)
        
        default_joint_angles = {
            # 'x_joint':  0, 'z_joint': 0.3, 'pitch_joint': 0, 
            'Hip': 0.9, 'Knee': -1.8,  #!
        }


    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'          # Position control 'P'
        stiffness = {
            # 'x_joint': 0., 'z_joint': 0, 'pitch_joint': 0, 
            'Hip': 35, 'Knee': 35}  # [N*m/rad]
        damping = {
            # 'x_joint': 0., 'z_joint': 0, 'pitch_joint': 0, 
            'Hip': 1.5, 'Knee': 1.5}     # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4 #!


    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/hopper-v2/hopper-v2.urdf"
        name = "hopper"
        foot_name = "Foot"
        penalize_contacts_on = ["Torso", "Thigh", "Shank", "Front_Weight", "Back_Weight"] # "Torso", "Thigh", "Shank", "Foot"
        terminate_after_contacts_on = ["Torso", "Thigh", "Shank", "Front_Weight", "Back_Weight"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False     # Some .obj meshes must be flipped from y-up to z-up
        fix_base_link = False               # fixe the base of the robot
        collapse_fixed_joints = False       # merge bodies connected by fixed joints.
        replace_cylinder_with_capsule = False


    # =================================================
    class commands ( LeggedRobotCfg.commands ):
        # General
        resampling_time = 10.     # [seconds]
        zero_command = True      
        zero_command_prob = 0.20
        
        # Command curriculum
        curriculum = False
        max_curriculum = 1.0
        vel_increment = 0.05      # [m/s]

        # Ranges
        heading_command = False

        class ranges:
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [0, 0]     # min max [m/s]
            ang_vel_yaw = [0, 0]   # min max [rad/s]
            heading = [0, 0]
    # =================================================


    class normalization( LeggedRobotCfg.normalization ):
        clip_observations = 50.
        clip_actions = 6.
        
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            xyz_pos = 1.0
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0


    class noise( LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales( LeggedRobotCfg.noise.noise_scales):
            lin_vel = 0.1
            dof_pos = 0.01
            dof_vel = 0.05
            ang_vel = 0.05
            gravity = 0.02
            height_measurements = 0.02
        

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.5
        only_positive_rewards = True

        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # ======================
            action_rate = -0.1
            ang_vel_xy = -0.01
            dof_acc = -5e-5
            torques = -0.005
            delta_torques = -1.0e-7
            # ====================== 
            collision = -10.0
            orientation = -1.0
            # ====================== 
            lin_vel_z_up = 0.10                # default: 0.10
            dof_error = -0.02                  # default: -0.01
            tracking_target_height = 2.0       #!
            phase_contact_match = 1.0          #!
            feet_contact_forces = -10          #!
            # ======================


class HopperCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):

        # Actor-Critic
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        init_noise_std = 1.0
        
        # Latent encoders
        priv_encoder_hidden_dims=[64, 20]
        latent_encoder_output_dim = 10

        # Scan encoder
        scan_encoder_hidden_dims=[128, 64]
        scan_encoder_output_dim = 32

        # Estimator
        estimator_hidden_dims = [256, 128]
        use_history = True
        
        # Activation (all)
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        estimator_learning_rate = 1e-4
        learning_rate = 2e-4
        schedule = 'fixed' # fixed or adaptive
        estimator_updates_per_ppo = 1

    class runner( LeggedRobotCfgPPO.runner ):
        # names
        run_name = 'hopper-v11'
        experiment_name = 'hopper'

        # training params
        max_iterations = 1000
        save_interval = 100

        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt