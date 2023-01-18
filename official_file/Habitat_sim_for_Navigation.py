import math
import os
import os.path as osp
import random
import sys

import imageio
import magnum as mn
import numpy as np

from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

work_dir = '/home/pgp/habitat/habitat-learning'
data_path = osp.join(work_dir, 'data')
output_dir = osp.join(work_dir, 'output/official_tutorials/Habitat_sim_for_Navigation')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

''' **************************** init **************************** '''


# Change to do something like this maybe: https://stackoverflow.com/a/41432704
def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    ''' display rgb、semantic、depth image '''
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        # PIL 的 P 模式
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())

        # 类别数%40
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        # todo: depth image的单位
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

''' **************************** create simulator **************************** '''

# download dataset
os.system('python -m habitat_sim.utils.datasets_download --uids mp3d_example_scene --data-path /home/pgp/habitat/data/')

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

# simple agent from simulator
# todo: what kinds of default agent?
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


action = "turn_right"
navigateAndSee(action)

action = "turn_right"
navigateAndSee(action)

action = "move_forward"
navigateAndSee(action)

action = "turn_left"
navigateAndSee(action)

''' **************************** complicate simulator **************************** '''

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


# print scene
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
# todo: scene is what kind of type?
scene = sim.semantic_scene
print(type(scene))
print_scene_recur(scene)

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
max_frames = 5
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

while total_frames < max_frames:
    action = random.choice(action_names)
    print("action", action)
    observations = sim.step(action)
    rgb = observations["color_sensor"]
    semantic = observations["semantic_sensor"]
    depth = observations["depth_sensor"]

    if display:
        display_sample(rgb, semantic, depth)

    total_frames += 1

##############################################################################

''' **************************** play with navmesh **************************** '''



def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    ''' Convert the x,z in habitat to x,y
        It is a little confusing, because minor the bounds
    '''
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
self_decide_height = False
height = 1

down_bound = sim.pathfinder.get_bounds()[0]
up_bound = sim.pathfinder.get_bounds()[1]
print("The NavMesh bounds are:\n"
      "\tdown_bound: {}\n"
      "\tup_bound: {}".format(down_bound, up_bound))

if not self_decide_height:
    height = sim.pathfinder.get_bounds()[0][1]

if not sim.pathfinder.is_loaded:
    print("Pathfinder not initialized, aborting.")
else:

    # Habitat-sim API
    # @markdown You can get the topdown map directly from the
    # Habitat-sim API with *PathFinder.get_topdown_view*.
    # This map is a 2D boolean array
    sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
    print("Displaying the raw map from get_topdown_view:")
    display_map(sim_topdown_map)

    if display:

        # Habitat-Lab
        # @markdown Alternatively, you can process the map using the
        # Habitat-Lab [maps module](https://github.com/facebookresearch/habitat-lab/blob/main/habitat/utils/visualizations/maps.py)
        hablab_topdown_map = maps.get_topdown_map(
            sim.pathfinder, height, meters_per_pixel=meters_per_pixel
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        hablab_topdown_map = recolor_map[hablab_topdown_map]
        print("Displaying the map from the Habitat-Lab maps module:")
        display_map(hablab_topdown_map)

        # easily save a map to file:
        map_filename = os.path.join(output_dir, "top_down_map.png")
        imageio.imsave(map_filename, hablab_topdown_map)

if not sim.pathfinder.is_loaded:
    print("Pathfinder not initialized, aborting.")
else:
    # @markdown NavMesh area and bounding box can be queried
    # via *navigable_area* and *get_bounds* respectively.
    print("NavMesh area = " + str(sim.pathfinder.navigable_area))
    print("Bounds = " + str(sim.pathfinder.get_bounds()))

    # @markdown A random point on the NavMesh can be queried with *get_random_navigable_point*.
    pathfinder_seed = 1  # @param {type:"integer"}
    sim.pathfinder.seed(pathfinder_seed)
    nav_point = sim.pathfinder.get_random_navigable_point()
    print("Random navigable point : " + str(nav_point))
    print("Is point navigable? " + str(sim.pathfinder.is_navigable(nav_point)))


    ''' island '''

    # @markdown The radius of the minimum containing circle (with vertex centroid origin) for the isolated navigable island of a point can be queried with *island_radius*.
    # @markdown This is analogous to the size of the point's connected component and can be used to check that a queried navigable point is on an interesting surface (e.g. the floor), rather than a small surface (e.g. a table-top).
    # @markdown 一个点的孤立通航岛的最小包含圆（以顶点质心为原点）的半径可以用*island_radius*查询。
    # @markdown 这类似于点的连接分量的大小，可用于检查查询的可导航点是否在有趣的表面（例如地板）上，而不是小表面（例如桌面）。
    print("Nav island radius : " + str(sim.pathfinder.island_radius(nav_point)))

    max_search_radius = 2.0  # @param {type:"number"}
    print("Distance to obstacle: "
        + str(sim.pathfinder.distance_to_closest_obstacle(nav_point, max_search_radius)))


    ''' hit '''

    # todo: 不懂
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
        print("Is point navigable? " + str(sim.pathfinder.is_navigable(perturbed_point)))

        snapped_point = sim.pathfinder.snap_point(perturbed_point)
        print("Snapped point : " + str(snapped_point))
        print("Is point navigable? " + str(sim.pathfinder.is_navigable(snapped_point)))
        vis_points.append(snapped_point)


    ''' sample point topdown '''

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

# todo: haven't read
def path_queries_on_navmesh():
    # @markdown ## Pathfinding Queries on NavMesh

    # @markdown The shortest path between valid points on the NavMesh can be queried as shown in this example.

    # @markdown With a valid PathFinder instance:
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        seed = 4  # @param {type:"integer"}
        sim.pathfinder.seed(seed)

        # fmt off
        # @markdown 1. Sample valid points on the NavMesh for agent spawn location and pathfinding goal.
        # fmt on
        sample1 = sim.pathfinder.get_random_navigable_point()
        sample2 = sim.pathfinder.get_random_navigable_point()

        # @markdown 2. Use ShortestPath module to compute path between samples.
        path = habitat_sim.ShortestPath()
        path.requested_start = sample1
        path.requested_end = sample2
        found_path = sim.pathfinder.find_path(path)
        geodesic_distance = path.geodesic_distance
        path_points = path.points
        # @markdown - Success, geodesic path length, and 3D points can be queried.
        print("found_path : " + str(found_path))
        print("geodesic_distance : " + str(geodesic_distance))
        print("path_points : " + str(path_points))

        # @markdown 3. Display trajectory (if found) on a topdown map of ground floor
        if found_path:
            meters_per_pixel = 0.025
            scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
            height = scene_bb.y().min
            if display:
                top_down_map = maps.get_topdown_map(
                    sim.pathfinder, height, meters_per_pixel=meters_per_pixel
                )
                recolor_map = np.array(
                    [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
                )
                top_down_map = recolor_map[top_down_map]
                grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
                # convert world trajectory points to maps module grid points
                trajectory = [
                    maps.to_grid(
                        path_point[2],
                        path_point[0],
                        grid_dimensions,
                        pathfinder=sim.pathfinder,
                    )
                    for path_point in path_points
                ]
                grid_tangent = mn.Vector2(
                    trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
                )
                path_initial_tangent = grid_tangent / grid_tangent.length()
                initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
                # draw the agent and trajectory on the map
                maps.draw_path(top_down_map, trajectory)
                maps.draw_agent(
                    top_down_map, trajectory[0], initial_angle, agent_radius_px=8
                )
                print("\nDisplay the map with agent and path overlay:")
                display_map(top_down_map)

            # @markdown 4. (optional) Place agent and render images at trajectory points (if found).
            display_path_agent_renders = True  # @param{type:"boolean"}
            if display_path_agent_renders:
                print("Rendering observations at path points:")
                tangent = path_points[1] - path_points[0]
                agent_state = habitat_sim.AgentState()
                for ix, point in enumerate(path_points):
                    if ix < len(path_points) - 1:
                        tangent = path_points[ix + 1] - point
                        agent_state.position = point
                        tangent_orientation_matrix = mn.Matrix4.look_at(
                            point, point + tangent, np.array([0, 1.0, 0])
                        )
                        tangent_orientation_q = mn.Quaternion.from_matrix(
                            tangent_orientation_matrix.rotation()
                        )
                        agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                        agent.set_state(agent_state)

                        observations = sim.get_sensor_observations()
                        rgb = observations["color_sensor"]
                        semantic = observations["semantic_sensor"]
                        depth = observations["depth_sensor"]

                        if display:
                            display_sample(rgb, semantic, depth)


''' **************************** load navmesh **************************** '''


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

''' **************************** recompute navmesh **************************** '''

navmesh_settings = habitat_sim.NavMeshSettings()

use_self_decide_settings = False  # @param {type:"boolean"}
sim.navmesh_visualization = True  # @param {type:"boolean"}

# Habitat-sim defaults (e.g. for point-nav tasks)
navmesh_settings.set_defaults()

# custom settings
if use_self_decide_settings:
    # fmt: off
    #@markdown ---
    #@markdown ## Configure custom settings (if use_custom_settings):
    #@markdown Configure the following NavMeshSettings for customized NavMesh recomputation.
    #@markdown **Voxelization parameters**:
    navmesh_settings.cell_size = 0.05 #@param {type:"slider", min:0.01, max:0.2, step:0.01}
    #default = 0.05
    navmesh_settings.cell_height = 0.2 #@param {type:"slider", min:0.01, max:0.4, step:0.01}
    #default = 0.2

    #@markdown **Agent parameters**:
    navmesh_settings.agent_height = 1.5 #@param {type:"slider", min:0.01, max:3.0, step:0.01}
    #default = 1.5
    navmesh_settings.agent_radius = 0.1 #@param {type:"slider", min:0.01, max:0.5, step:0.01}
    #default = 0.1
    navmesh_settings.agent_max_climb = 0.2 #@param {type:"slider", min:0.01, max:0.5, step:0.01}
    #default = 0.2
    navmesh_settings.agent_max_slope = 45 #@param {type:"slider", min:0, max:85, step:1.0}
    # default = 45.0
    # fmt: on
    # @markdown **Navigable area filtering options**:
    navmesh_settings.filter_low_hanging_obstacles = True  # @param {type:"boolean"}
    # default = True
    navmesh_settings.filter_ledge_spans = True  # @param {type:"boolean"}
    # default = True
    navmesh_settings.filter_walkable_low_height_spans = True  # @param {type:"boolean"}
    # default = True

    # fmt: off
    #@markdown **Detail mesh generation parameters**:
    #@markdown For more details on the effects
    navmesh_settings.region_min_size = 20 #@param {type:"slider", min:0, max:50, step:1}
    #default = 20
    navmesh_settings.region_merge_size = 20 #@param {type:"slider", min:0, max:50, step:1}
    #default = 20
    navmesh_settings.edge_max_len = 12.0 #@param {type:"slider", min:0, max:50, step:1}
    #default = 12.0
    navmesh_settings.edge_max_error = 1.3 #@param {type:"slider", min:0, max:5, step:0.1}
    #default = 1.3
    navmesh_settings.verts_per_poly = 6.0 #@param {type:"slider", min:3, max:6, step:1}
    #default = 6.0
    navmesh_settings.detail_sample_dist = 6.0 #@param {type:"slider", min:0, max:10.0, step:0.1}
    #default = 6.0
    navmesh_settings.detail_sample_max_error = 1.0 #@param {type:"slider", min:0, max:10.0, step:0.1}
    # default = 1.0
    # fmt: on

navmesh_success = sim.recompute_navmesh(
    sim.pathfinder, navmesh_settings, include_static_objects=False
)

if not navmesh_success:
    print("Failed to build the navmesh! Try different parameters?")
else:

    agent_state = sim.agents[0].get_state()
    set_random_valid_state = False  # @param {type:"boolean"}
    seed = 5  # @param {type:"integer"}
    sim.seed(seed)
    orientation = 0
    if set_random_valid_state:
        agent_state.position = sim.pathfinder.get_random_navigable_point()
        orientation = random.random() * math.pi * 2.0
    # @markdown Optionally configure the agent state (overrides random state):
    set_agent_state = True  # @param {type:"boolean"}
    try_to_make_valid = True  # @param {type:"boolean"}
    if set_agent_state:
        pos_x = 0  # @param {type:"number"}
        pos_y = 0  # @param {type:"number"}
        pos_z = 0.0  # @param {type:"number"}
        # @markdown Y axis rotation (radians):
        orientation = 1.56  # @param {type:"number"}
        agent_state.position = np.array([pos_x, pos_y, pos_z])
        if try_to_make_valid:
            snapped_point = np.array(sim.pathfinder.snap_point(agent_state.position))
            if not np.isnan(np.sum(snapped_point)):
                print("Successfully snapped point to: " + str(snapped_point))
                agent_state.position = snapped_point
    if set_agent_state or set_random_valid_state:
        agent_state.rotation = utils.quat_from_magnum(
            mn.Quaternion.rotation(-mn.Rad(orientation), mn.Vector3(0, 1.0, 0))
        )
        sim.agents[0].set_state(agent_state)

    agent_state = sim.agents[0].get_state()
    print("Agent state: " + str(agent_state))
    print(" position = " + str(agent_state.position))
    print(" rotation = " + str(agent_state.rotation))
    print(" orientation (about Y) = " + str(orientation))

    observations = sim.get_sensor_observations()
    rgb = observations["color_sensor"]
    semantic = observations["semantic_sensor"]
    depth = observations["depth_sensor"]

    if display:
        display_sample(rgb, semantic, depth)
        # @markdown **Map parameters**:
        # fmt: off
        meters_per_pixel = 0.025  # @param {type:"slider", min:0.01, max:0.1, step:0.005}
        # fmt: on
        agent_pos = agent_state.position
        # topdown map at agent position
        top_down_map = maps.get_topdown_map(
            sim.pathfinder, height=agent_pos[1], meters_per_pixel=meters_per_pixel
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        top_down_map = recolor_map[top_down_map]
        grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
        # convert world agent position to maps module grid point
        agent_grid_pos = maps.to_grid(
            agent_pos[2], agent_pos[0], grid_dimensions, pathfinder=sim.pathfinder
        )
        agent_forward = utils.quat_to_magnum(
            sim.agents[0].get_state().rotation
        ).transform_vector(mn.Vector3(0, 0, -1.0))
        agent_orientation = math.atan2(agent_forward[0], agent_forward[2])
        # draw the agent and trajectory on the map
        maps.draw_agent(
            top_down_map, agent_grid_pos, agent_orientation, agent_radius_px=8
        )
        print("\nDisplay topdown map with agent:")
        display_map(top_down_map)


# Saving the NavMesh

if sim.pathfinder.is_loaded:
    navmesh_save_path = "/content/habitat-sim/data/test_saving.navmesh" #@param {type:"string"}
    sim.pathfinder.save_nav_mesh(navmesh_save_path)
    print('Saved NavMesh to "' + navmesh_save_path + '"')
    sim.pathfinder.load_nav_mesh(navmesh_save_path)

''' **************************** Discrete and Continuous Navigation **************************** '''

# Discrete and Continuous Navigation:
'''
## What is sliding?
Most game engines allow agents to slide along obstacles when commanding actions 
which collide with the environment. While this is a reasonable behavior in games,
it does not accuractely reflect the result of collisions between robotic agents and the environment.

We note that **allowing sliding** makes training easier and results in higher simulation performance, 
but **hurts sim-2-real transfer** of trained policies.

'''


# @markdown Take moment to run this cell a couple times and note the differences between discrete and continuous navigation with and without sliding.

seed = 7
use_current_scene = False  # @param {type:"boolean"}

output_directory = 'output'

sim_settings["seed"] = seed
if not use_current_scene:
    # reload a default nav scene
    sim_settings["scene"] = "./data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
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

# Seconds to simulate:
sim_time = 10

# Optional continuous action space parameters:
continuous_nav = True

# defaults for discrete control
# control frequency (actions/sec):
control_frequency = 3
# observation/integration frames per action
frame_skip = 1

if continuous_nav:
    control_frequency = 5
    frame_skip = 12

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

# todo: physic?
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
    # 对于1s的控制频率为control_frequency，而每个控制将会执行frame_skip帧
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


            # agent body action
            # 这里是实际上是调用了注册后的action，但是对于body和non-body是通过控制对象的不同来实现，
            # 而对于注册action的时候如果设置了body选项，会自动执行
            # todo: right? what is the underlying of action
            # todo: use shift shift of pycharm to check [NoisyStrafeLeft] [habitat_sim.SceneNodeControl)]
            if agent.controls.is_body_action(discrete_action.name):
                did_collide = agent.controls.action(
                    agent.scene_node,
                    discrete_action.name,
                    discrete_action.actuation,
                    apply_filter=True,
                )

            # agent sensor action
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

                # 将方向转向机器人局部坐标系下,并且根据时间积分
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

