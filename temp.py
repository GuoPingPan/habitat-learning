seed : 100
env_task : GymHabitatEnv
env_task_gym_dependencies : []
env_task_gym_id : 
environment : {'max_episode_steps': 500, 'max_episode_seconds': 10000000, 'iterator_options': {'cycle': True, 'shuffle': True, 'group_by_scene': True, 'num_episode_sample': -1, 'max_scene_repeat_episodes': -1, 'max_scene_repeat_steps': 10000, 'step_repetition_range': 0.2}}
simulator : {'type': 'Sim-v0', 'action_space_config': 'v0', 'action_space_config_arguments': {}, 'forward_step_size': 0.25, 'create_renderer': False, 'requires_textures': True, 'auto_sleep': False, 'step_physics': True, 'concur_render': False, 'needs_markers': True, 'update_robot': True, 'scene': 'data/scene_datasets/habitat-test-scenes/van-gogh-room.glb', 'scene_dataset': 'default', 'additional_object_paths': [], 'seed': '${habitat.seed}', 'turn_angle': 10, 'tilt_angle': 15, 'default_agent_id': 0, 'debug_render': False, 'debug_render_robot': False, 'kinematic_mode': False, 'debug_render_goal': True, 'robot_joint_start_noise': 0.0, 'ctrl_freq': 120.0, 'ac_freq_ratio': 4, 'load_objs': False, 'hold_thresh': 0.15, 'grasp_impulse': 10000.0, 'agents': {'main_agent': {'height': 1.5, 'radius': 0.1, 'sim_sensors': {'rgb_sensor': {'type': 'HabitatSimRGBSensor', 'height': 256, 'width': 256, 'position': [0.0, 1.25, 0.0], 'orientation': [0.0, 0.0, 0.0], 'hfov': 90, 'sensor_subtype': 'PINHOLE', 'noise_model': 'None', 'noise_model_kwargs': {}}, 'depth_sensor': {'type': 'HabitatSimDepthSensor', 'height': 256, 'width': 256, 'position': [0.0, 1.25, 0.0], 'orientation': [0.0, 0.0, 0.0], 'hfov': 90, 'sensor_subtype': 'PINHOLE', 'noise_model': 'None', 'noise_model_kwargs': {}, 'min_depth': 0.0, 'max_depth': 10.0, 'normalize_depth': True}}, 'is_set_start_state': False, 'start_position': [0.0, 0.0, 0.0], 'start_rotation': [0.0, 0.0, 0.0, 1.0], 'joint_start_noise': 0.1, 'robot_urdf': 'data/robots/hab_fetch/robots/hab_fetch.urdf', 'robot_type': 'FetchRobot', 'ik_arm_urdf': 'data/robots/hab_fetch/robots/fetch_onlyarm.urdf'}}, 'agents_order': ['main_agent'], 'habitat_sim_v0': {'gpu_device_id': 0, 'gpu_gpu': False, 'allow_sliding': True, 'frustum_culling': True, 'enable_physics': False, 'physics_config_file': './data/default.physics_config.json', 'leave_context_with_background_renderer': False, 'enable_gfx_replay_save': False}, 'ep_info': None}
task : {'reward_measure': 'distance_to_goal_reward', 'success_measure': 'spl', 'success_reward': 2.5, 'slack_reward': -0.01, 'end_on_success': True, 'type': 'Nav-v0', 'lab_sensors': {'pointgoal_with_gps_compass_sensor': {'type': 'PointGoalWithGPSCompassSensor', 'goal_format': 'POLAR', 'dimensionality': 2}}, 'measurements': {'distance_to_goal': {'type': 'DistanceToGoal', 'distance_to': 'POINT'}, 'success': {'type': 'Success', 'success_distance': 0.2}, 'spl': {'type': 'SPL'}, 'distance_to_goal_reward': {'type': 'DistanceToGoalReward'}}, 'goal_sensor_uuid': 'pointgoal_with_gps_compass', 'count_obj_collisions': True, 'settle_steps': 5, 'constraint_violation_ends_episode': True, 'constraint_violation_drops_object': False, 'force_regenerate': False, 'should_save_to_cache': False, 'must_look_at_targ': True, 'object_in_hand_sample_prob': 0.167, 'min_start_distance': 3.0, 'render_target': True, 'physics_stability_steps': 1, 'num_spawn_attempts': 200, 'spawn_max_dists_to_obj': 2.0, 'base_angle_noise': 0.523599, 'ee_sample_factor': 0.2, 'ee_exclude_region': 0.0, 'base_noise': 0.05, 'spawn_region_scale': 0.2, 'joint_max_impulse': -1.0, 'desired_resting_position': [0.5, 0.0, 1.0], 'use_marker_t': True, 'cache_robot_init': False, 'success_state': 0.0, 'easy_init': False, 'should_enforce_target_within_reach': False, 'task_spec_base_path': 'habitat/task/rearrange/pddl/', 'task_spec': '', 'pddl_domain_def': 'replica_cad', 'obj_succ_thresh': 0.3, 'enable_safe_drop': False, 'art_succ_thresh': 0.15, 'robot_at_thresh': 2.0, 'filter_nav_to_tasks': [], 'actions': {'stop': {'type': 'StopAction'}, 'move_forward': {'type': 'MoveForwardAction'}, 'turn_left': {'type': 'TurnLeftAction'}, 'turn_right': {'type': 'TurnRightAction'}}}
dataset : {'type': 'PointNav-v1', 'split': 'train', 'scenes_dir': 'data/scene_datasets', 'content_scenes': ['*'], 'data_path': 'data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz'}
gym : {'auto_name': '', 'obs_keys': None, 'action_keys': None, 'achieved_goal_keys': [], 'desired_goal_keys': []}


# @markdown This example demonstrates updating the agent state to follow the motion of an object during simulation.

rigid_obj_mgr.remove_all_objects()
visual_sensor = sim._sensors["color_sensor_1st_person"]
initial_sensor_position = np.array(visual_sensor._spec.position)
initial_sensor_orientation = np.array(visual_sensor._spec.orientation)
# set the color sensor transform to be the agent transform
visual_sensor._spec.position = np.array([0, 0, 0])
visual_sensor._spec.orientation = np.array([0, 0, 0])
visual_sensor._sensor_object.set_transformation_from_spec()

# boost the agent off the floor
sim.get_agent(0).scene_node.translation += np.array([0, 1.5, 0])
observations = []

# @markdown ---
# @markdown ### Set example parameters:
seed = 23  # @param {type:"integer"}
random.seed(seed)
sim.seed(seed)
np.random.seed(seed)

# add an object and position the agent
sel_file_obj = rigid_obj_mgr.add_object_by_template_handle(sel_file_obj_handle)
rand_position = np.random.uniform(
    np.array([-0.4, -0.3, -1.0]), np.array([0.4, 0.3, -0.5])
)
set_object_state_from_agent(sim, sel_file_obj, rand_position, ut.random_quaternion())

# simulate with updated camera at each frame
start_time = sim.get_world_time()
while sim.get_world_time() - start_time < 2.0:
    sim.step_physics(1.0 / 60.0)
    # set agent state to look at object
    camera_position = sim.get_agent(0).scene_node.translation
    camera_look_at = sel_file_obj.translation
    sim.get_agent(0).scene_node.rotation = mn.Quaternion.from_matrix(
        mn.Matrix4.look_at(
            camera_position, camera_look_at, np.array([0, 1.0, 0])  # up
        ).rotation()
    )
    observations.append(sim.get_sensor_observations())

# video rendering with embedded 1st person view
video_prefix = "motion tracking"
if make_video:
    vut.make_video(
        observations,
        "color_sensor_1st_person",
        "color",
        output_path + video_prefix,
        open_vid=show_video,
    )

# reset the sensor state for other examples
visual_sensor._spec.position = initial_sensor_position
visual_sensor._spec.orientation = initial_sensor_orientation
visual_sensor._sensor_object.set_transformation_from_spec()
# put the agent back
sim.reset()
rigid_obj_mgr.remove_all_objects()