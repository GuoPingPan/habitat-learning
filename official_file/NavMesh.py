import matplotlib.pyplot as plt
import numpy as np
import os
import habitat_sim
import habitat
import math

# test_scene = '/media/pgp/U330/毕设/office_1/habitat/mesh_semantic.glb'
test_scene = '/media/pgp/U330/毕设/objglb/mesh_semantic.glb'
# test_scene = '/media/pgp/U330/毕设/objglb/output/output.glb'

raplica_scene_dataset = "./mydata/datasets/pointnav/mydata/v1/{split}/{split}.json.gz"

rgb_sensor = True  # @param {type:"boolean"}
depth_sensor = True  # @param {type:"boolean"}
semantic_sensor = True  # @param {type:"boolean"}

sim_settings = {
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
    "scene": test_scene,  # Scene path
    "scene_dataset": raplica_scene_dataset,  # the scene dataset configuration files
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

try:  # Needed to handle out of order cell run in Colab
    sim.close()
except NameError:
    pass
sim = habitat_sim.Simulator(cfg)
exit()
# Print semantic annotation information (id, category, bounding box details)
# about levels, regions and objects in a hierarchical fashion
scene = sim.semantic_scene

output_path = './output'

def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
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


# @markdown ###Configure Example Parameters:
# @markdown Configure the map resolution:
meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}
# @markdown ---
# @markdown Customize the map slice height (global y coordinate):
self_defined_height = False  # @param {type:"boolean"}
height = -5.4  # @param {type:"slider", min:-10, max:10, step:0.1}
# @markdown If not using custom height, default to scene lower limit.
# @markdown (Cell output provides scene height range from bounding box for reference.)

print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
if not self_defined_height:
    # get bounding box minumum elevation for automatic height
    height = sim.pathfinder.get_bounds()[0][1]

if not sim.pathfinder.is_loaded:
    print("Pathfinder not initialized, aborting.")
else:
    # @markdown You can get the topdown map directly from the Habitat-sim API with *PathFinder.get_topdown_view*.
    # This map is a 2D boolean array
    sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)


display = True

from habitat.utils.visualizations import maps

# if not sim.pathfinder.is_loaded:
#     print("Pathfinder not initialized, aborting.")
# else:
#     # @markdown You can get the topdown map directly from the Habitat-sim API with *PathFinder.get_topdown_view*.
#     # This map is a 2D boolean array
#     sim_topdown_map = sim.pathfinder.get_topdown_island_view(meters_per_pixel, height)
#
#     if display:
#         # @markdown Alternatively, you can process the map using the Habitat-Lab [maps module](https://github.com/facebookresearch/habitat-lab/blob/main/habitat/utils/visualizations/maps.py)
#         hablab_topdown_map = maps.get_topdown_map(
#             sim.pathfinder, height, meters_per_pixel=meters_per_pixel
#         )
#         recolor_map = np.array(
#             [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
#         )
#         hablab_topdown_map = recolor_map[hablab_topdown_map]
#         print("Displaying the raw map from get_topdown_view:")
#         display_map(sim_topdown_map)
#         print("Displaying the map from the Habitat-Lab maps module:")
#         display_map(hablab_topdown_map)
#
#         # easily save a map to file:
#         map_filename = os.path.join(output_path, "top_down_map.png")

# if not sim.pathfinder.is_loaded:
#     print("Pathfinder not initialized, aborting.")
# else:
#     print('navmesh area = ' + str(sim.pathfinder.navigable_area))
#     print("Bounds = " + str(sim.pathfinder.get_bounds()))
#     print('\n')
#
#     pathfinder_seed = 1  # @param {type:"integer"}
#     sim.pathfinder.seed(pathfinder_seed)
#     nav_point = sim.pathfinder.get_random_navigable_point()
#     print("Random navigable point : " + str(nav_point))
#     print("Is point navigable? " + str(sim.pathfinder.is_navigable(nav_point)))
#     print('\n')
#
#     # @markdown The radius of the minimum containing circle (with vertex centroid origin) for the isolated navigable island of a point can be queried with *island_radius*.
#     # @markdown This is analogous to the size of the point's connected component and can be used to check that a queried navigable point is on an interesting surface (e.g. the floor), rather than a small surface (e.g. a table-top).
#     print("Nav island radius : " + str(sim.pathfinder.island_radius(nav_point)))
#     print("num of islands: ", sim.pathfinder.num_islands)
#     print('\n')
#
#
#     # @markdown The closest boundary point can also be queried (within some radius).
#     max_search_radius = 2.0  # @param {type:"number"}
#     print(
#         "Distance to obstacle: "
#         + str(sim.pathfinder.distance_to_closest_obstacle(nav_point, max_search_radius))
#     )
#     print('\n')
#
#     hit_record = sim.pathfinder.closest_obstacle_surface_point(
#         nav_point, max_search_radius
#     )
#     print("Closest obstacle HitRecord:")
#     print(" point: " + str(hit_record.hit_pos))
#     print(" normal: " + str(hit_record.hit_normal))
#     print(" distance: " + str(hit_record.hit_dist))
#
#     vis_points = [nav_point]
#
#     # HitRecord will have infinite distance if no valid point was found:
#     if math.isinf(hit_record.hit_dist):
#         print("No obstacle found within search radius.")
#     else:
#         # @markdown Points near the boundary or above the NavMesh can be snapped onto it.
#         perturbed_point = hit_record.hit_pos - hit_record.hit_normal * 0.2
#         print("Perturbed point : " + str(perturbed_point))
#         print(
#             "Is point navigable? " + str(sim.pathfinder.is_navigable(perturbed_point))
#         )
#         snapped_point = sim.pathfinder.snap_point(perturbed_point)
#         print("Snapped point : " + str(snapped_point))
#         print("Is point navigable? " + str(sim.pathfinder.is_navigable(snapped_point)))
#         vis_points.append(snapped_point)
#
#     # @markdown ---
#     # @markdown ### Visualization
#     # @markdown Running this cell generates a topdown visualization of the NavMesh with sampled points overlayed.
#     meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}
#
#     if display:
#         xy_vis_points = convert_points_to_topdown(
#             sim.pathfinder, vis_points, meters_per_pixel
#         )
#         # use the y coordinate of the sampled nav_point for the map height slice
#         top_down_map = maps.get_topdown_map(
#             sim.pathfinder, height=nav_point[1], meters_per_pixel=meters_per_pixel
#         )
#         recolor_map = np.array(
#             [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
#         )
#         top_down_map = recolor_map[top_down_map]
#         print("\nDisplay the map with key_point overlay:")
#         display_map(top_down_map, key_points=xy_vis_points)


navmesh_settings = habitat_sim.NavMeshSettings()

# @markdown Choose Habitat-sim defaults (e.g. for point-nav tasks), or custom settings.
use_self_defined_settings = False  # @param {type:"boolean"}
sim.navmesh_visualization = True  # @param {type:"boolean"}
navmesh_settings.set_defaults()
if use_self_defined_settings:
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


import random
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
import magnum as mn
from PIL import Image

def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
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

navmesh_success = sim.recompute_navmesh(
    sim.pathfinder, navmesh_settings, include_static_objects=False
)


# if not navmesh_success:
#     print("Failed to build the navmesh! Try different parameters?")
# else:
#
#     agent_state = sim.agents[0].get_state()
#     set_random_valid_state = False  # @param {type:"boolean"}
#     seed = 5  # @param {type:"integer"}
#     sim.seed(seed)
#     orientation = 0
#     if set_random_valid_state:
#         agent_state.position = sim.pathfinder.get_random_navigable_point()
#         orientation = random.random() * math.pi * 2.0
#     # @markdown Optionally configure the agent state (overrides random state):
#     set_agent_state = True  # @param {type:"boolean"}
#     try_to_make_valid = True  # @param {type:"boolean"}
#     if set_agent_state:
#         pos_x = 0  # @param {type:"number"}
#         pos_y = 0  # @param {type:"number"}
#         pos_z = 0.0  # @param {type:"number"}
#         # @markdown Y axis rotation (radians):
#         orientation = 1.56  # @param {type:"number"}
#         agent_state.position = np.array([pos_x, pos_y, pos_z])
#         if try_to_make_valid:
#             snapped_point = np.array(sim.pathfinder.snap_point(agent_state.position))
#             if not np.isnan(np.sum(snapped_point)):
#                 print("Successfully snapped point to: " + str(snapped_point))
#                 agent_state.position = snapped_point
#     if set_agent_state or set_random_valid_state:
#         agent_state.rotation = utils.quat_from_magnum(
#             mn.Quaternion.rotation(-mn.Rad(orientation), mn.Vector3(0, 1.0, 0))
#         )
#         sim.agents[0].set_state(agent_state)
#
#     agent_state = sim.agents[0].get_state()
#     print("Agent state: " + str(agent_state))
#     print(" position = " + str(agent_state.position))
#     print(" rotation = " + str(agent_state.rotation))
#     print(" orientation (about Y) = " + str(orientation))
#
#     observations = sim.get_sensor_observations()
#     rgb = observations["color_sensor"]
#     semantic = observations["semantic_sensor"]
#     depth = observations["depth_sensor"]
#
#     if display:
#         display_sample(rgb, semantic, depth)
#         # @markdown **Map parameters**:
#         # fmt: off
#         meters_per_pixel = 0.025  # @param {type:"slider", min:0.01, max:0.1, step:0.005}
#         # fmt: on
#         agent_pos = agent_state.position
#         # topdown map at agent position
#         top_down_map = maps.get_topdown_map(
#             sim.pathfinder, height=agent_pos[1], meters_per_pixel=meters_per_pixel
#         )
#         recolor_map = np.array(
#             [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
#         )
#         top_down_map = recolor_map[top_down_map]
#         grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
#         # convert world agent position to maps module grid point
#         agent_grid_pos = maps.to_grid(
#             agent_pos[2], agent_pos[0], grid_dimensions, pathfinder=sim.pathfinder
#         )
#         agent_forward = utils.quat_to_magnum(
#             sim.agents[0].get_state().rotation
#         ).transform_vector(mn.Vector3(0, 0, -1.0))
#         agent_orientation = math.atan2(agent_forward[0], agent_forward[2])
#         # draw the agent and trajectory on the map
#         maps.draw_agent(
#             top_down_map, agent_grid_pos, agent_orientation, agent_radius_px=8
#         )
#         print("\nDisplay topdown map with agent:")
#         display_map(top_down_map)

if sim.pathfinder.is_loaded and navmesh_success:
    navmesh_save_path = "output/mesh_semantic_small.navmesh" #@param {type:"string"}
    sim.pathfinder.save_nav_mesh(navmesh_save_path)
    print('Saved NavMesh to "' + navmesh_save_path + '"')
    sim.pathfinder.load_nav_mesh(navmesh_save_path)
# fmt: on