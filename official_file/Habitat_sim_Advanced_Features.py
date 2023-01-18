import math
import os
import os.path as osp
import random
import sys

import magnum as mn
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as ut
from habitat_sim.utils import viz_utils as vut

work_dir = '/home/pgp/habitat/habitat-learning'
data_path = osp.join(work_dir, 'data')
output_dir = osp.join(work_dir, 'output/official_tutorials/Habitat_sim_Advanced_Features')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if "sim" not in globals():
    global sim
    sim = None
    # all kins of manager
    global obj_attr_mgr  # sim.get_object_template_manager()
    obj_attr_mgr = None
    global prim_attr_mgr # sim.get_asset_template_manager()
    prim_attr_mgr = None
    global stage_attr_mgr # sim.get_stage_template_manager()
    stage_attr_mgr = None
    global rigid_obj_mgr # sim.get_rigid_object_manager()
    rigid_obj_mgr = None

# @title Define Configuration Utility Functions { display-mode: "form" }
# @markdown (double click to show code)

# @markdown This cell defines a number of utility functions used throughout the tutorial to make simulator reconstruction easy:
# @markdown - make_cfg
# @markdown - make_default_settings
# @markdown - make_simulator_from_settings
def set_object_state_from_agent(
    sim,
    obj,
    offset=np.array([0, 2.0, -1.5]), # 高出2m，agent前面-1.5m的物体
    orientation=mn.Quaternion(((0, 0, 0), 1)),
):
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    ob_translation = agent_transform.transform_point(offset)
    obj.translation = ob_translation
    obj.rotation = orientation


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    # Optional; Specify the location of an existing scene dataset configuration
    # that describes the locations and configurations of all the assets to be used
    if "scene_dataset_config" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config"]

    # Note: all sensors must have the same resolution
    sensor_specs = []
    if settings["color_sensor_1st_person"]:
        color_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
        color_sensor_1st_person_spec.uuid = "color_sensor_1st_person"
        color_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_1st_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        color_sensor_1st_person_spec.position = [0.0, settings["sensor_height"], 0.0]
        color_sensor_1st_person_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        color_sensor_1st_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_1st_person_spec)
    if settings["depth_sensor_1st_person"]:
        depth_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_1st_person_spec.uuid = "depth_sensor_1st_person"
        depth_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_1st_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        depth_sensor_1st_person_spec.position = [0.0, settings["sensor_height"], 0.0]
        depth_sensor_1st_person_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        depth_sensor_1st_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(depth_sensor_1st_person_spec)
    if settings["semantic_sensor_1st_person"]:
        semantic_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_1st_person_spec.uuid = "semantic_sensor_1st_person"
        semantic_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_1st_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        semantic_sensor_1st_person_spec.position = [
            0.0,
            settings["sensor_height"],
            0.0,
        ]
        semantic_sensor_1st_person_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        semantic_sensor_1st_person_spec.sensor_subtype = (
            habitat_sim.SensorSubType.PINHOLE
        )
        sensor_specs.append(semantic_sensor_1st_person_spec)
    if settings["color_sensor_3rd_person"]:
        color_sensor_3rd_person_spec = habitat_sim.CameraSensorSpec()
        color_sensor_3rd_person_spec.uuid = "color_sensor_3rd_person"
        color_sensor_3rd_person_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_3rd_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        color_sensor_3rd_person_spec.position = [
            0.0,
            settings["sensor_height"] + 0.2,
            0.2,
        ]
        color_sensor_3rd_person_spec.orientation = [-math.pi / 4, 0, 0]
        color_sensor_3rd_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_3rd_person_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def make_default_settings():
    settings = {
        "width": 720,  # Spatial resolution of the observations
        "height": 544,
        "scene": "./data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb",  # Scene path
        "scene_dataset": "./data/scene_datasets/mp3d_example/mp3d.scene_dataset_config.json",  # mp3d scene dataset
        "default_agent": 0,
        "sensor_height": 1.5,  # Height of sensors in meters
        "sensor_pitch": -math.pi / 8.0,  # sensor pitch (x rotation in rads)
        "color_sensor_1st_person": True,  # RGB sensor
        "color_sensor_3rd_person": False,  # RGB sensor 3rd person
        "depth_sensor_1st_person": False,  # Depth sensor
        "semantic_sensor_1st_person": False,  # Semantic sensor
        "seed": 1,
        "enable_physics": True,  # enable dynamics simulation
    }
    return settings

def make_simulator_from_settings(sim_settings):
    cfg = make_cfg(sim_settings)
    # clean-up the current simulator instance if it exists
    global sim
    global obj_attr_mgr
    global prim_attr_mgr
    global stage_attr_mgr
    global rigid_obj_mgr

    if sim != None:
        sim.close()
    # initialize the simulator
    sim = habitat_sim.Simulator(cfg)
    # Managers of various Attributes templates
    obj_attr_mgr = sim.get_object_template_manager()
    obj_attr_mgr.load_configs(str(os.path.join(data_path, "objects/example_objects")))
    prim_attr_mgr = sim.get_asset_template_manager()
    stage_attr_mgr = sim.get_stage_template_manager()
    # Manager providing access to rigid objects
    rigid_obj_mgr = sim.get_rigid_object_manager()

    # UI-populated handles used in various cells.  Need to initialize to valid
    # value in case IPyWidgets are not available.
    # Holds the user's desired file-based object template handle
    global sel_file_obj_handle
    sel_file_obj_handle = obj_attr_mgr.get_file_template_handles()[0]
    # Holds the user's desired primitive-based object template handle
    global sel_prim_obj_handle
    sel_prim_obj_handle = obj_attr_mgr.get_synth_template_handles()[0]
    # Holds the user's desired primitive asset template handle
    global sel_asset_handle
    sel_asset_handle = prim_attr_mgr.get_template_handles()[0]

def display_sample(
    rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([]), key_points=None
):
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
        # plot points on images
        if key_points is not None:
            for point in key_points:
                plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
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
    make_video = args.make_video
else:
    show_video = False
    make_video = False
    display = False

def build_widget_ui(obj_attr_mgr, prim_attr_mgr):
    # Holds the user's desired file-based object template handle
    global sel_file_obj_handle
    sel_file_obj_handle = ""

    # Holds the user's desired primitive-based object template handle
    global sel_prim_obj_handle
    sel_prim_obj_handle = ""

    # Holds the user's desired primitive asset template handle
    global sel_asset_handle
    sel_asset_handle = ""

    # Construct DDLs and assign event handlers
    # All file-based object template handles
    file_obj_handles = obj_attr_mgr.get_file_template_handles()
    prim_obj_handles = obj_attr_mgr.get_synth_template_handles()
    prim_asset_handles = prim_attr_mgr.get_template_handles()

    sel_file_obj_handle = file_obj_handles[0]
    sel_prim_obj_handle = prim_obj_handles[0]
    sel_asset_handle = prim_asset_handles[0]



''' Advanced Features '''

'''
    agent的相机视角跟着object走
'''

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
video_prefix = "motion_tracking"
if make_video:
    vut.make_video(
        observations,
        "color_sensor_1st_person",
        "color",
        osp.join(output_dir, video_prefix),
        open_vid=show_video,
    )

# reset the sensor state for other examples
visual_sensor._spec.position = initial_sensor_position
visual_sensor._spec.orientation = initial_sensor_orientation
visual_sensor._sensor_object.set_transformation_from_spec()
# put the agent back
sim.reset()
rigid_obj_mgr.remove_all_objects()

'''
    将三维空间点映射到二维图像上
'''

# @markdown ###Display 2D Projection of Object COMs
# fmt: off
# @markdown First define the projection function using the current state of a chosen VisualSensor to set camera parameters and then projects the 3D point.
# fmt: on
# project a 3D point into 2D image space for a particular sensor
def get_2d_point(sim, sensor_name, point_3d):
    # get the scene render camera and sensor object
    render_camera = sim._sensors[sensor_name]._sensor_object.render_camera

    # use the camera and projection matrices to transform the point onto the near plane
    projected_point_3d = render_camera.projection_matrix.transform_point(
        render_camera.camera_matrix.transform_point(point_3d)
    )

    # todo： 这里不是很懂
    # convert the 3D near plane point to integer pixel space
    point_2d = mn.Vector2(projected_point_3d[0], -projected_point_3d[1])
    point_2d = point_2d / render_camera.projection_size()[0]
    point_2d += mn.Vector2(0.5)
    point_2d *= render_camera.viewport
    return mn.Vector2i(point_2d)


# fmt: off
# @markdown Use this function to compute the projected object center of mass (COM) 2D projection and display on the image.
# fmt: on
# @markdown ---
# @markdown ### Set example parameters:
seed = 27  # @param {type:"integer"}
random.seed(seed)
sim.seed(seed)
np.random.seed(seed)

rigid_obj_mgr.remove_all_objects()

# add an object and plot the COM on the image
sel_file_obj = rigid_obj_mgr.add_object_by_template_handle(sel_file_obj_handle)
rand_position = np.random.uniform(
    np.array([-0.4, 1.2, -1.0]), np.array([0.4, 1.8, -0.5])
)
set_object_state_from_agent(sim, sel_file_obj, rand_position, ut.random_quaternion())

obs = sim.get_sensor_observations()

com_2d = get_2d_point(
    sim, sensor_name="color_sensor_1st_person", point_3d=sel_file_obj.translation
)
if display:
    display_sample(obs["color_sensor_1st_person"], key_points=[com_2d])
rigid_obj_mgr.remove_all_objects()

'''
    自定义语义id
'''

# @markdown ###Display 2D Projection of Object COMs
# fmt: off
# @markdown First define the projection function using the current state of a chosen VisualSensor to set camera parameters and then projects the 3D point.
# fmt: on
# project a 3D point into 2D image space for a particular sensor
def get_2d_point(sim, sensor_name, point_3d):
    # get the scene render camera and sensor object
    render_camera = sim._sensors[sensor_name]._sensor_object.render_camera

    # use the camera and projection matrices to transform the point onto the near plane
    projected_point_3d = render_camera.projection_matrix.transform_point(
        render_camera.camera_matrix.transform_point(point_3d)
    )
    # convert the 3D near plane point to integer pixel space
    point_2d = mn.Vector2(projected_point_3d[0], -projected_point_3d[1])
    point_2d = point_2d / render_camera.projection_size()[0]
    point_2d += mn.Vector2(0.5)
    point_2d *= render_camera.viewport
    return mn.Vector2i(point_2d)


# fmt: off
# @markdown Use this function to compute the projected object center of mass (COM) 2D projection and display on the image.
# fmt: on
# @markdown ---
# @markdown ### Set example parameters:
seed = 27  # @param {type:"integer"}
random.seed(seed)
sim.seed(seed)
np.random.seed(seed)

rigid_obj_mgr.remove_all_objects()

# add an object and plot the COM on the image
sel_file_obj = rigid_obj_mgr.add_object_by_template_handle(sel_file_obj_handle)
rand_position = np.random.uniform(
    np.array([-0.4, 1.2, -1.0]), np.array([0.4, 1.8, -0.5])
)
set_object_state_from_agent(sim, sel_file_obj, rand_position, ut.random_quaternion())

obs = sim.get_sensor_observations()

com_2d = get_2d_point(
    sim, sensor_name="color_sensor_1st_person", point_3d=sel_file_obj.translation
)
if display:
    display_sample(obs["color_sensor_1st_person"], key_points=[com_2d])
rigid_obj_mgr.remove_all_objects()

''' Advanced Topic : Object and Primitive Asset Customization '''
# key: 这里是讲如何操作object attribute 的
# key: details in [https://colab.research.google.com/github/facebookresearch/habitat-sim/blob/main/examples/tutorials/colabs/ECCV_2020_Advanced_Features.ipynb#scrollTo=z5wJXSHuWW47 ]