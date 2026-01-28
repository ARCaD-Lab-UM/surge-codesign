from legged_gym.envs.base.base_config import BaseConfig

class HopperStandaloneCfg(BaseConfig):
    class env:
        history_buffer_length = 10
        num_proprio = 1+3+3+3+3+2+2+2+2 # (z, lin_vel, ang_vel, proj_grav, vel_cmd, dof_pos, dof_vel, dof_pos_des, gait features)
        num_scan_obs = 0
        num_estimated_obs = 3 # lin_vel
        num_envs = 1024
        num_privileged_obs = 4+1+2+2+2 # (com_pos, friction, kp, kd, design_params) 
        num_critic_obs = num_proprio+(num_proprio*history_buffer_length)+num_privileged_obs+num_estimated_obs+num_scan_obs
        num_observations = num_proprio+(num_proprio*history_buffer_length)
        
        # Gait features
        period = 0.75
        
        num_actions = 2
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # (max) episode length in seconds


    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m] --> meters per grid cell
        vertical_scale = 0.005 # [m] --> meters per grid value
        border_size = 25 # [m]
        curriculum = False
        promote_threshold = 0.60 # [%] --> percentage of terrain traversed to move up a level
        demote_threshold = 0.40  # [%] --> percentage of terrain traversed to move down a level
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        # rough terrain only:
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10 # number of terrain rows (levels of difficulty)
        num_cols = 20 # number of terrain cols (max. number of terrain types)

        measure_heights = True
        measured_points_x = [-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 12
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75]   # 11
        
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain

        max_init_terrain_level = 5 # starting curriculum state
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2, 0.0, 0.0] # see terrain.py for details
        
        # disable custom functions
        add_roughness_to_selected_terrain = False
        parkour = False
        
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        resampling_time = 10.     # [seconds]
        zero_command = True      
        zero_command_prob = 0.20

        curriculum = False              # Curriculum will resample commands dynamically
        max_curriculum = 1.0
        vel_increment = 0.05
        num_commands = 4                # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        heading_command = False          # if true: compute ang vel command from heading error
        user_command = []               # if not empty: will override resampling logic         
        
        class ranges:
            lin_vel_x = [-0.25, 0.25] # min max [m/s]
            lin_vel_y = [0, 0]     # min max [m/s]
            ang_vel_yaw = [0, 0]   # min max [rad/s]
            heading = [0, 0]

    class init_state:
        pos = [0.0, 0.0, 0.4]      # [x, y, z] (metres)
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]   # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]   # x,y,z [rad/s]

        default_joint_angles = {
            # 'x_joint':  0, 'z_joint': 0.3, 'pitch_joint': 0, 
            'Hip': 0.9, 'Knee': -1.8, 
        }

    class control:
        # PD Drive parameters:
        control_type = 'P'          # Position control 'P'
        stiffness = {
            # 'x_joint': 0., 'z_joint': 0, 'pitch_joint': 0, 
            'Hip': 40, 'Knee': 40}  # [N*m/rad]
        damping = {
            # 'x_joint': 0., 'z_joint': 0, 'pitch_joint': 0, 
            'Hip': 1, 'Knee': 1}     # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4 # may set to 1 to reduce drift in constraint dims

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/hopper-v2/hopper-v2.urdf"
        name = "hopper"
        foot_name = "Foot"
        penalize_contacts_on = ["Torso", "Thigh", "Shank", "Front_Weight", "Back_Weight"] # "Torso", "Thigh", "Shank", "Foot"
        terminate_after_contacts_on = ["Torso", "Thigh", "Shank", "Front_Weight", "Back_Weight"]
        self_collisions = 0                     # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False          # Some .obj meshes must be flipped from y-up to z-up
        fix_base_link = False                   # fixe the base of the robot
        collapse_fixed_joints = False            # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        replace_cylinder_with_capsule = False    # replace collision cylinders with capsules, leads to faster/more stable simulation
        disable_gravity = False
        default_dof_drive_mode = 3              # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 1.3]

        randomize_base_mass = True
        added_mass_range = [-0.025, 0.025]
        
        randomize_center_of_mass = True
        added_com_range = [-0.05, 0.05]

        randomize_kp_kd = True
        kp_kd_range = [0.8, 1.2]

        push_robots = False
        push_interval_s = 8
        max_push_vel_xy = 0.5

    class rewards:
        # NOTE: variables below are used to compute the reward
        soft_dof_pos_limit = 0.9        # midpoint +/- 0.5 * (range) * (soft_dof_pos_limit)
        base_height_target = 0.5
        only_positive_rewards = True
        
        tracking_sigma = 0.25           # tracking reward = exp(-error^2/sigma)
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_contact_force = 100.        # forces above this value are penalized

        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # ======================
            action_rate = -0.1
            ang_vel_xy = -0.01
            dof_acc = -2.5e-7
            torques = -0.00001
            delta_torques = -1.0e-7
            # ====================== 
            collision = -10.0
            orientation = -1.0
            # ====================== 
            lin_vel_z_up = 0.10                # default: 0.10
            dof_error = -0.02                  # default: -0.01
            tracking_target_height = 2.0      # default: 1.0
            # ======================

            # Zeroed out (unused)
            termination = -0.0   
            lin_vel_z = 0.0                
            dof_vel = 0.0               
            base_height = 0.0           
            feet_air_time = 0.0                
            stumble_feet = 0.0               
            stand_still = 0.0           
            contact_forces = 0.0        
        
    class normalization:
        clip_observations = 100.
        clip_actions = 6
        class obs_scales:
            xyz_pos = 1.0
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values

        class noise_scales:
            xyz_pos = 0.1
            imu = 0.05      # imu & gravity mutually exclusive
            lin_vel = 0.1
            dof_pos = 0.01
            dof_vel = 0.05
            ang_vel = 0.05
            gravity = 0.02  # imu & gravity mutually exclusive
            height_measurements = 0.02

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class HopperStandaloneCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:

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

        # Activation function (all)
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        # training params
        estimator_learning_rate = 1e-4
        learning_rate = 2e-4       # 5.e-4
        schedule = 'fixed'       # could be adaptive, fixed
        estimator_updates_per_ppo = 1  # no. estimator updates per ppo batch

        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5     # default: 5
        num_mini_batches = 4        # mini batch size = num_envs*nsteps / nminibatches
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        dagger_update_freq = 20

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration

        # names
        run_name = 'hopper-v6 | dof_error=-0.02 | lin_vel_z_up = 0.10 | vel: [-0.25, 0.25] | best'
        experiment_name = 'hopper'

        # training params
        max_iterations = 1000
        save_interval = 1000

        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt