import math
import os
import random
import sys

import git
import imageio
import magnum as mn
import numpy as np

from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

# Change to do something like this maybe: https://stackoverflow.com/a/41432704
def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    ''' display rgb、semantic、depth image '''
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--no-make-video", dest="make_video", action="store_false")
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()
    show_video = args.display
    display = args.display
    do_make_video = args.make_video
else:
    show_video = False
    do_make_video = False
    display = False

# import the maps module alone for topdown mapping
if display:
    from habitat.utils.visualizations import maps


# This is the scene we are going to load.
# we support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
test_scene = '/home/pgp/habitat/habitat-learning/data/replica_cad/stages/Stage_v3_sc0_staging.glb'

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
}

def make_simple_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings['scene']
    sensor_specs = []
    sensor_layers = []

    sensor_layer_names = ['color_sensor', 'depth_sensor', 'semantic_sensor']

    for name in sensor_layer_names:
        if name == 'color_sensor':
            sensor_layers.append(habitat_sim.SensorType.COLOR)
        elif name == 'depth_sensor':
            sensor_layers.append(habitat_sim.SensorType.DEPTH)
        elif name == 'semantic_sensor':
            sensor_layers.append(habitat_sim.SensorType.SEMANTIC)

    for name, layer in zip(sensor_layer_names, sensor_layers):
        print(name,layer)
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = name
        sensor_spec.sensor_type = layer
        sensor_spec.resolution = [settings["height"], settings["width"]]
        sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(sensor_spec)


    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

cfg = make_simple_cfg(sim_settings)

try:  # Needed to handle out of order cell run in Colab
    sim.close()
except NameError:
    pass
sim = habitat_sim.Simulator(cfg)

agent = sim.initialize_agent(sim_settings['default_agent'])
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([0.6, 0, 0])
agent.set_state(agent_state)

agent_state = agent.get_state()
print('agent_state: position', agent_state.position)
print('agent_state: rotation', agent_state.rotation)

action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)

def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        print(observations)
        print("action: ", action)
        if display:
            display_sample(observations["color_sensor"], observations["semantic_sensor"], observations["depth_sensor"])


# action = "turn_right"
# navigateAndSee(action)
#
# action = "turn_right"
# navigateAndSee(action)
#
# action = "move_forward"
# navigateAndSee(action)
#
# action = "turn_left"
# navigateAndSee(action)
import habitat_sim.utils.datasets_download
##############################################################################

test_scene = "../data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
mp3d_scene_dataset = "../data/scene_datasets/mp3d_example/mp3d.scene_dataset_config.json"

rgb_sensor = True  # @param {type:"boolean"}
depth_sensor = True  # @param {type:"boolean"}
semantic_sensor = True  # @param {type:"boolean"}

sim_settings = {
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
    "scene": test_scene,  # Scene path
    "scene_dataset": mp3d_scene_dataset,  # the scene dataset configuration files
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": rgb_sensor,  # RGB sensor
    "depth_sensor": depth_sensor,  # Depth sensor
    "semantic_sensor": semantic_sensor,  # Semantic sensor
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
}

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

cfg = make_cfg(sim_settings)
# Needed to handle out of order cell run in Colab
try:  # Got to make initialization idiot proof
    sim.close()
except NameError:
    pass
sim = habitat_sim.Simulator(cfg)


def print_scene_recur(scene, limit_output=10):
    print(
        f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
    )
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return


# Print semantic annotation information (id, category, bounding box details)
# about levels, regions and objects in a hierarchical fashion
scene = sim.semantic_scene
# print_scene_recur(scene)

# the randomness is needed when choosing the actions
random.seed(sim_settings["seed"])
sim.seed(sim_settings["seed"])

# Set agent state
agent = sim.initialize_agent(sim_settings["default_agent"])
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0])  # world space
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

total_frames = 0
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

max_frames = 5

# while total_frames < max_frames:
#     action = random.choice(action_names)
#     print("action", action)
#     observations = sim.step(action)
#     rgb = observations["color_sensor"]
#     semantic = observations["semantic_sensor"]
#     depth = observations["depth_sensor"]
#
#     if display:
#         display_sample(rgb, semantic, depth)
#
#     total_frames += 1

##############################################################################

''' NavMesh '''

# The following example cell defines a matplotlib function
# to display a top down map with optional key points overlay.
# It then generates a topdown map of the current scene
# using the [minimum y] coordinate of the scene bounding box as the height,
# or an [optionally configured custom height].
# Note that this height is in scene global coordinates,
# so we cannot assume that 0 is the bottom floor.


def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown

def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=False)

meters_per_pixel = 0.1
custom_height = False
height = 1

down_bound = sim.pathfinder.get_bounds()[0]
up_bound = sim.pathfinder.get_bounds()[1]

print("The NavMesh bounds are:\n"
      "\tdown_bound: {}\n"
      "\tup_bound: {}".format(down_bound, up_bound) )

if not custom_height:
    height = sim.pathfinder.get_bounds()[0][1]

output_path = 'output'

if not sim.pathfinder.is_loaded:
    print("Pathfinder not initialized, aborting.")
else:
    # @markdown You can get the topdown map directly from the Habitat-sim API with *PathFinder.get_topdown_view*.
    # This map is a 2D boolean array
    sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)

    if display:
        # @markdown Alternatively, you can process the map using the Habitat-Lab [maps module](https://github.com/facebookresearch/habitat-lab/blob/main/habitat/utils/visualizations/maps.py)
        hablab_topdown_map = maps.get_topdown_map(
            sim.pathfinder, height, meters_per_pixel=meters_per_pixel
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        hablab_topdown_map = recolor_map[hablab_topdown_map]
        # print("Displaying the raw map from get_topdown_view:")
        # display_map(sim_topdown_map)
        # print("Displaying the map from the Habitat-Lab maps module:")
        # display_map(hablab_topdown_map)

        # easily save a map to file:
        map_filename = os.path.join(output_path, "top_down_map.png")
        imageio.imsave(map_filename, hablab_topdown_map)

if not sim.pathfinder.is_loaded:
    print("Pathfinder not initialized, aborting.")
else:
    # @markdown NavMesh area and bounding box can be queried via *navigable_area* and *get_bounds* respectively.
    print("NavMesh area = " + str(sim.pathfinder.navigable_area))
    print("Bounds = " + str(sim.pathfinder.get_bounds()))

    # @markdown A random point on the NavMesh can be queried with *get_random_navigable_point*.
    pathfinder_seed = 1  # @param {type:"integer"}
    sim.pathfinder.seed(pathfinder_seed)
    nav_point = sim.pathfinder.get_random_navigable_point()
    print("Random navigable point : " + str(nav_point))
    print("Is point navigable? " + str(sim.pathfinder.is_navigable(nav_point)))

    # @markdown The radius of the minimum containing circle (with vertex centroid origin) for the isolated navigable island of a point can be queried with *island_radius*.
    # @markdown This is analogous to the size of the point's connected component and can be used to check that a queried navigable point is on an interesting surface (e.g. the floor), rather than a small surface (e.g. a table-top).

    # @markdown 一个点的孤立通航岛的最小包含圆（以顶点质心为原点）的半径可以用*island_radius*查询。
    # @markdown 这类似于点的连接分量的大小，可用于检查查询的可导航点是否在有趣的表面（例如地板）上，而不是小表面（例如桌面）。
    print("Nav island radius : " + str(sim.pathfinder.island_radius(nav_point)))


    max_search_radius = 2.0  # @param {type:"number"}
    print(
        "Distance to obstacle: "
        + str(sim.pathfinder.distance_to_closest_obstacle(nav_point, max_search_radius))
    )


    hit_record = sim.pathfinder.closest_obstacle_surface_point(
        nav_point, max_search_radius
    )
    print("Closest obstacle HitRecord:")
    print(" point: " + str(hit_record.hit_pos))
    print(" normal: " + str(hit_record.hit_normal))
    print(" distance: " + str(hit_record.hit_dist))


    vis_points = [nav_point]

    # HitRecord will have infinite distance if no valid point was found:
    if math.isinf(hit_record.hit_dist):
        print("No obstacle found within search radius.")
    else:
        # @markdown Points near the boundary or above the NavMesh can be snapped onto it.
        perturbed_point = hit_record.hit_pos - hit_record.hit_normal * 0.2
        print("Perturbed point : " + str(perturbed_point))
        print(
            "Is point navigable? " + str(sim.pathfinder.is_navigable(perturbed_point))
        )
        snapped_point = sim.pathfinder.snap_point(perturbed_point)
        print("Snapped point : " + str(snapped_point))
        print("Is point navigable? " + str(sim.pathfinder.is_navigable(snapped_point)))
        vis_points.append(snapped_point)

    # @markdown ---
    # @markdown ### Visualization
    # @markdown Running this cell generates a topdown visualization of the NavMesh with sampled points overlayed.
    meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}

    if display:
        xy_vis_points = convert_points_to_topdown(
            sim.pathfinder, vis_points, meters_per_pixel
        )
        # use the y coordinate of the sampled nav_point for the map height slice
        top_down_map = maps.get_topdown_map(
            sim.pathfinder, height=nav_point[1], meters_per_pixel=meters_per_pixel
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        top_down_map = recolor_map[top_down_map]
        print("\nDisplay the map with key_point overlay:")
        display_map(top_down_map, key_points=xy_vis_points)

    # initialize a new simulator with the apartment_1 scene
    # this will automatically load the accompanying .navmesh file
    sim_settings["scene"] = "../data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    cfg = make_cfg(sim_settings)
    try:  # Got to make initialization idiot proof
        sim.close()
    except NameError:
        pass
    sim = habitat_sim.Simulator(cfg)

    # the navmesh can also be explicitly loaded
    sim.pathfinder.load_nav_mesh(
        "../data/scene_datasets/habitat-test-scenes/apartment_1.navmesh"
    )


print('**************************************************************************************************************************************')
# @title Discrete and Continuous Navigation:

# @markdown Take moment to run this cell a couple times and note the differences between discrete and continuous navigation with and without sliding.

# @markdown ---
# @markdown ### Set example parameters:
seed = 7  # @param {type:"integer"}
# @markdown Optionally navigate on the currently configured scene and NavMesh instead of re-loading with defaults:
use_current_scene = False  # @param {type:"boolean"}

output_directory = 'output'

sim_settings["seed"] = seed
if not use_current_scene:
    # reload a default nav scene
    sim_settings[
        "scene"
    ] = "./data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
    cfg = make_cfg(sim_settings)
    try:  # make initialization Colab cell order proof
        sim.close()
    except NameError:
        pass
    sim = habitat_sim.Simulator(cfg)
random.seed(sim_settings["seed"])
sim.seed(sim_settings["seed"])
# set new initial state
sim.initialize_agent(agent_id=0)
agent = sim.agents[0]

# @markdown Seconds to simulate:
sim_time = 10  # @param {type:"integer"}

# @markdown Optional continuous action space parameters:
continuous_nav = True  # @param {type:"boolean"}

# defaults for discrete control
# control frequency (actions/sec):
control_frequency = 3
# observation/integration frames per action
frame_skip = 1
if continuous_nav:
    control_frequency = 5  # @param {type:"slider", min:1, max:30, step:1}
    frame_skip = 12  # @param {type:"slider", min:1, max:30, step:1}


fps = control_frequency * frame_skip
print("fps = " + str(fps))
control_sequence = []
for _action in range(int(sim_time * control_frequency)):
    if continuous_nav:
        # allow forward velocity and y rotation to vary
        control_sequence.append(
            {
                "forward_velocity": random.random() * 2.0,  # [0,2)
                "rotation_velocity": (random.random() - 0.5) * 2.0,  # [-1,1)
            }
        )
    else:
        control_sequence.append(random.choice(action_names))

# create and configure a new VelocityControl structure
vel_control = habitat_sim.physics.VelocityControl()
vel_control.controlling_lin_vel = True
vel_control.lin_vel_is_local = True
vel_control.controlling_ang_vel = True
vel_control.ang_vel_is_local = True

# try 2 variations of the control experiment
for iteration in range(2):
    # reset observations and robot state
    observations = []

    video_prefix = "nav_sliding"
    sim.config.sim_cfg.allow_sliding = True
    # turn sliding off for the 2nd pass
    if iteration == 1:
        sim.config.sim_cfg.allow_sliding = False
        video_prefix = "nav_no_sliding"

    print(video_prefix)

    # manually control the object's kinematic state via velocity integration
    time_step = 1.0 / (frame_skip * control_frequency)
    print("time_step = " + str(time_step))
    for action in control_sequence:

        # apply actions
        if continuous_nav:
            # update the velocity control
            # local forward is -z
            vel_control.linear_velocity = np.array([0, 0, -action["forward_velocity"]])
            # local up is y
            vel_control.angular_velocity = np.array([0, action["rotation_velocity"], 0])

        else:  # discrete action navigation
            discrete_action = agent.agent_config.action_space[action]

            did_collide = False
            if agent.controls.is_body_action(discrete_action.name):
                did_collide = agent.controls.action(
                    agent.scene_node,
                    discrete_action.name,
                    discrete_action.actuation,
                    apply_filter=True,
                )
            else:
                for _, v in agent._sensors.items():
                    habitat_sim.errors.assert_obj_valid(v)
                    agent.controls.action(
                        v.object,
                        discrete_action.name,
                        discrete_action.actuation,
                        apply_filter=False,
                    )

        # simulate and collect frames
        for _frame in range(frame_skip):
            if continuous_nav:
                # Integrate the velocity and apply the transform.
                # Note: this can be done at a higher frequency for more accuracy
                agent_state = agent.state
                previous_rigid_state = habitat_sim.RigidState(
                    utils.quat_to_magnum(agent_state.rotation), agent_state.position
                )

                # manually integrate the rigid state
                target_rigid_state = vel_control.integrate_transform(
                    time_step, previous_rigid_state
                )

                # snap rigid state to navmesh and set state to object/agent
                # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
                end_pos = sim.step_filter(
                    previous_rigid_state.translation, target_rigid_state.translation
                )

                # set the computed state
                agent_state.position = end_pos
                agent_state.rotation = utils.quat_from_magnum(
                    target_rigid_state.rotation
                )
                agent.set_state(agent_state)

                # Check if a collision occured
                dist_moved_before_filter = (
                    target_rigid_state.translation - previous_rigid_state.translation
                ).dot()
                dist_moved_after_filter = (
                    end_pos - previous_rigid_state.translation
                ).dot()

                # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
                # collision _didn't_ happen. One such case is going up stairs.  Instead,
                # we check to see if the the amount moved after the application of the filter
                # is _less_ than the amount moved before the application of the filter

                # 注意：有些情况下 ||filter_end - end_pos|| > 0 当
                # 碰撞_没有_发生。 一个这样的例子是上楼梯。 取而代之，
                # 我们检查应用过滤器后移动的数量是否
                # _小于_ 应用过滤器之前移动的量
                EPS = 1e-5
                collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter

            # run any dynamics simulation
            sim.step_physics(time_step)

            # render observation
            observations.append(sim.get_sensor_observations())

    print("frames = " + str(len(observations)))
    # video rendering with embedded 1st person view
    if do_make_video:
        # use the vieo utility to render the observations
        vut.make_video(
            observations=observations,
            primary_obs="color_sensor",
            primary_obs_type="color",
            video_file=output_directory + "continuous_nav",
            fps=fps,
            open_vid=show_video,
        )

    sim.reset()

# [/embodied_agent_navmesh]