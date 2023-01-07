# 这个案例来自：https://github.com/matthew-so/habitat-interactive-tasks/blob/master/Copy_of_Habitat_Interactive_Tasks.ipynb
# 是一个 Rearrangement 的任务，为了简单只需要重排一个物体

# 构建基于 habitat-sim 和 habitat-lab
# 1. 构建 simulator：使用 habitat-sim，能够添加和移除一些物体，能够 spawn 一些 agent 和 objects 在预定义的一些位置
# 2. 构建 new dataset：save and load episodes
# 3. 构建 grab/release action
# 4. 拓展 simulator 以满足新定义的 action
# 5. 创建 new task：创建新的任务定义，new sensors and metrics
# 6. 训练 RL agent：定义 reward，用 PPO 算法训练



import gzip
import json
import os
import sys
from typing import Any, Dict, List, Optional, Type

import attr
import cv2
import git
import magnum as mn
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image

import habitat
import habitat_sim
from habitat.config import DictConfig as Config
from habitat.core.registry import registry
from habitat_sim.utils import viz_utils as vut

dir_path = '/home/pgp/habitat'
data_path = os.path.join(dir_path, "data")
output_directory = "data/tutorials/output/"  # @param {type:"string"}
output_path = output_directory

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument(
        "--no-make-video", dest="make_video", action="store_false"
    )
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()
    show_video = args.display
    display = args.display
    make_video = args.make_video
else:
    show_video = False
    make_video = False
    display = False

if make_video and not os.path.exists(output_path):
    os.makedirs(output_path)

''' Utils functions to visualize observations'''
def make_video_cv2(
    observations, cross_hair=None, prefix="", open_vid=True, fps=60
):
    sensor_keys = list(observations[0])
    videodims = observations[0][sensor_keys[0]].shape
    videodims = (videodims[1], videodims[0])  # flip to w,h order
    print(videodims)
    video_file = output_path + prefix + ".mp4"
    print("Encoding the video: %s " % video_file)
    writer = vut.get_fast_video_writer(video_file, fps=fps)
    for ob in observations:
        # If in RGB/RGBA format, remove the alpha channel
        rgb_im_1st_person = cv2.cvtColor(ob["rgb"], cv2.COLOR_RGBA2RGB)
        if cross_hair is not None:
            rgb_im_1st_person[
                cross_hair[0] - 2 : cross_hair[0] + 2,
                cross_hair[1] - 2 : cross_hair[1] + 2,
            ] = [255, 0, 0]

        if rgb_im_1st_person.shape[:2] != videodims:
            rgb_im_1st_person = cv2.resize(
                rgb_im_1st_person, videodims, interpolation=cv2.INTER_AREA
            )
        # write the 1st person observation to video
        writer.append_data(rgb_im_1st_person)
    writer.close()

    if open_vid:
        print("Displaying video")
        vut.display_video(video_file)


def simulate(sim, dt=1.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())
    return observations


# convenience wrapper for simulate and make_video_cv2
def simulate_and_make_vid(sim, crosshair, prefix, dt=1.0, open_vid=True):
    observations = simulate(sim, dt)
    make_video_cv2(observations, crosshair, prefix=prefix, open_vid=open_vid)


def display_sample(
    rgb_obs,
    semantic_obs=np.array([]),
    depth_obs=np.array([]),
    key_points=None,  # noqa: B006
):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGB")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new(
            "P", (semantic_obs.shape[1], semantic_obs.shape[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray(
            (depth_obs / 10 * 255).astype(np.uint8), mode="L"
        )
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
                plt.plot(
                    point[0], point[1], marker="o", markersize=10, alpha=0.8
                )
        plt.imshow(data)

    plt.show(block=False)


'''----------------------------------------------------------'''


''' Constructing Simulator and Agent '''
def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.default_agent_id = settings["default_agent_id"]
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    sim_cfg.physics_config_file = settings["physics_config_file"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "rgb"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(rgb_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # 将 sensor 传入 agent
    agent_cfg.sensor_specifications = sensor_specs

    # 类似 action_space
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

settings = {
    "max_frames": 10,
    "width": 256,
    "height": 256,
    "scene": "data/scene_datasets/coda/coda.glb",
    "default_agent_id": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "rgb": True,  # RGB sensor
    "depth": True,  # Depth sensor
    "seed": 1,
    "enable_physics": True,
    "physics_config_file": "data/default.physics_config.json",
    "silent": False,
    "compute_shortest_path": False,
    "compute_action_shortest_path": False,
    "save_png": True,
}

cfg = make_cfg(settings)


# 其实在创建 simulator 的时候已经根据 config 完成了 agent 的创建
# 这里只是调用了一下 simulator 的接口来调整 agent 的位置
def init_agent(sim):
    agent_pos = np.array([-0.15776923, 0.18244143, 0.2988735])

    # Place the agent
    sim.agents[0].scene_node.translation = agent_pos
    agent_orientation_y = -40
    sim.agents[0].scene_node.rotation = mn.Quaternion.rotation(
        mn.Deg(agent_orientation_y), mn.Vector3(0, 1.0, 0)
    )


cfg.sim_cfg.default_agent_id = 0
with habitat_sim.Simulator(cfg) as sim:
    init_agent(sim)
    if make_video:
        # Visualize the agent's initial position
        simulate_and_make_vid(
            sim, None, "sim-init", dt=1.0, open_vid=show_video
        )

'''----------------------------------------------------------'''

''' Set Objects '''
# 使用 sim 完成

def remove_all_objects(sim):
    for obj_id in sim.get_existing_object_ids():
        sim.remove_object(obj_id)


def set_object_in_front_of_agent(sim, obj_id, z_offset=-1.5):
    r"""
    Adds an object in front of the agent at some distance.
    """

    # 利用 agent 的转换矩阵来 transform object 到世界坐标系
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    obj_translation = agent_transform.transform_point(
        np.array([0, 0, z_offset])
    )
    sim.set_translation(obj_translation, obj_id)

    obj_node = sim.get_object_scene_node(obj_id)
    xform_bb = habitat_sim.geo.get_transformed_bb(
        obj_node.cumulative_bb, obj_node.transformation
    )

    # also account for collision margin of the scene
    scene_collision_margin = 0.04
    y_translation = mn.Vector3(
        0, xform_bb.size_y() / 2.0 + scene_collision_margin, 0
    )
    sim.set_translation(y_translation + sim.get_translation(obj_id), obj_id)


def init_objects(sim):
    # 加载模板
    # Manager of Object Attributes Templates
    obj_attr_mgr = sim.get_object_template_manager()
    obj_attr_mgr.load_configs(
        str(os.path.join(data_path, "test_assets/objects"))
    )

    # Add a chair into the scene.
    obj_path = "test_assets/objects/chair"
    chair_template_id = obj_attr_mgr.load_object_configs(
        str(os.path.join(data_path, obj_path))
    )[0]
    chair_attr = obj_attr_mgr.get_template_by_id(chair_template_id)
    obj_attr_mgr.register_template(chair_attr)

    # Object's initial position 3m away from the agent.
    object_id = sim.add_object_by_handle(chair_attr.handle)
    set_object_in_front_of_agent(sim, object_id, -3.0)
    sim.set_object_motion_type(
        habitat_sim.physics.MotionType.STATIC, object_id
    )

    # Object's final position 7m away from the agent
    goal_id = sim.add_object_by_handle(chair_attr.handle)
    set_object_in_front_of_agent(sim, goal_id, -7.0)
    sim.set_object_motion_type(habitat_sim.physics.MotionType.STATIC, goal_id)

    return object_id, goal_id


with habitat_sim.Simulator(cfg) as sim:
    init_agent(sim)
    init_objects(sim)

    # Visualize the scene after the chair is added into the scene.
    if make_video:
        simulate_and_make_vid(
            sim, None, "object-init", dt=1.0, open_vid=show_video
        )



'''----------------------------------------------------------'''


''' Create New Dataset Use Above Episode '''

def get_rotation(sim, object_id):
    quat = sim.get_rotation(object_id)
    return np.array(quat.vector).tolist() + [quat.scalar]


def add_object_details(sim, episode_dict, obj_id, object_template, object_id):
    object_template = {
        "object_id": obj_id,
        "object_template": object_template,
        "position": np.array(sim.get_translation(object_id)).tolist(),
        "rotation": get_rotation(sim, object_id),
    }
    episode_dict["objects"] = object_template
    return episode_dict


def add_goal_details(sim, episode_dict, object_id):
    goal_template = {
        "position": np.array(sim.get_translation(object_id)).tolist(),
        "rotation": get_rotation(sim, object_id),
    }
    episode_dict["goals"] = goal_template
    return episode_dict

def init_episode_dict(episode_id, scene_id, agent_pos, agent_rot):
    episode_dict = {
        "episode_id": episode_id,
        "scene_id": "data/scene_datasets/coda/coda.glb",
        "start_position": agent_pos,
        "start_rotation": agent_rot,
        "info": {},
    }
    return episode_dict

def build_episode(sim, episode_num, object_id, goal_id):
    episodes = {'episodes': []}
    for i in range(episode_num):
        agent_state = sim.get_agent(0).get_state()
        agent_pos = np.array(agent_state.position).tolist()
        agent_quat = agent_state.rotation
        # todo 这里是将 wxyz 换成 xyzw 吗？
        agent_rot = np.array(agent_quat.vec).tolist() + [agent_quat.real]
        episode_dict = init_episode_dict(
            i, settings["scene"], agent_pos, agent_rot
        )
        object_attr = sim.get_object_initialization_template(object_id)
        object_path = os.path.relpath(
            os.path.splitext(object_attr.render_asset_handle)[0]
        )

        episode_dict = add_object_details(
            sim, episode_dict, 0, object_path, object_id
        )
        episode_dict = add_goal_details(sim, episode_dict, goal_id)
        episodes["episodes"].append(episode_dict)

    return episodes

with habitat_sim.Simulator(cfg) as sim:
    init_agent(sim)
    object_id, goal_id = init_objects(sim)

    episodes = build_episode(sim, 1, object_id, goal_id)

    dataset_content_path = 'data/dataset/rearrangement/coda/v1/train'
    os.makedirs(dataset_content_path, exist_ok=True)

    with gzip.open(
        os.path.join(dataset_content_path, "train.json.gz"), 'wt'
    ) as f:
        json.dump(episodes, f)

    print(
        "Dataset written to {}".format(
            os.path.join(dataset_content_path, "train.json.gz")
        )
    )

'''
{
  'episode_id': 0,
  'scene_id': 'data/scene_datasets/coda/coda.glb',
  'goals': {
    'position': [4.34, 0.67, -5.06],
    'rotation': [0.0, 0.0, 0.0, 1.0]
   },
  'objects': {
    'object_id': 0,
    'object_template': 'data/test_assets/objects/chair',
    'position': [1.77, 0.67, -1.99],
    'rotation': [0.0, 0.0, 0.0, 1.0]
  },
  'start_position': [-0.15, 0.18, 0.29],
  'start_rotation': [-0.0, -0.34, -0.0, 0.93]}
}
'''

''' Load Dataset '''

# @title Dataset class to read the saved dataset in Habitat-Lab.
# @markdown To read the saved episodes in Habitat-Lab,
# we will extend the `Dataset` class and the `Episode` base class.
# It will help provide all the relevant details about the episode
# through a consistent API to all downstream tasks.


from habitat.core.utils import DatasetFloatJSONEncoder, not_none_validator
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)

from habitat.tasks.nav.nav import NavigationEpisode
@attr.s(auto_attribs=True, kw_only=True)
class RearrangementSpec:
    r"""Specifications that capture a particular position of final position
    or initial position of the object.
    """

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    info: Optional[Dict[str, str]] = attr.ib(default=None)

@attr.s(auto_attribs=True, kw_only=True)
class RearrangementObjectSpec(RearrangementSpec):
    r"""Object specifications that capture position of each object in the scene,
    the associated object template.
    """
    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_template: Optional[str] = attr.ib(
        default="data/test_assets/objects/chair"
    )

@attr.s(auto_attribs=True, kw_only=True)
class RearrangementEpisode(NavigationEpisode):
    r"""Specification of episode that includes initial position and rotation
    of agent, all goal specifications, all object specifications

    Args:
        episode_id: id of episode in the dataset
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        goal: object's goal position and rotation
        object: object's start specification defined with object type,
            position, and rotation.
    """
    objects: RearrangementObjectSpec = attr.ib(
        default=None, validator=not_none_validator
    )
    goals: RearrangementSpec = attr.ib(
        default=None, validator=not_none_validator
    )

@registry.register_dataset(name="RearrangementDataset-v0")
class RearrangementDatasetV0(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Rearrangement dataset."""
    episodes: List[RearrangementEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for i, episode in enumerate(deserialized["episodes"]):
            rearrangement_episode = RearrangementEpisode(**episode)
            rearrangement_episode.episode_id = str(i)

            if scenes_dir is not None:
                if rearrangement_episode.scene_id.startswith(
                    DEFAULT_SCENE_PATH_PREFIX
                ):
                    # data/scene_datasets/ 开头的去掉
                    rearrangement_episode.scene_id = (
                        rearrangement_episode.scene_id[
                            len(DEFAULT_SCENE_PATH_PREFIX) :
                        ]
                    )

                rearrangement_episode.scene_id = os.path.join(
                    scenes_dir, rearrangement_episode.scene_id
                )

            rearrangement_episode.objects = RearrangementObjectSpec(
                **rearrangement_episode.objects
            )
            rearrangement_episode.goals = RearrangementSpec(
                **rearrangement_episode.goals
            )

            self.episodes.append(rearrangement_episode)

config = habitat.get_config("configs/datasets/pointnav/habitat_test.yaml")
with habitat.config.read_write(config):

    config.habitat.dataset.data_path = (
        "data/datasets/rearrangement/coda/v1/{split}/{split}.json.gz"
    )
    config.habitat.dataset.type = "RearrangementDataset-v0"

dataset = RearrangementDatasetV0(config.habitat.dataset)

# check if the dataset got correctly deserialized
assert len(dataset.episodes) == 1

assert dataset.episodes[0].objects.position == [
    1.770593523979187,
    0.6726829409599304,
    -1.9992598295211792,
]
assert dataset.episodes[0].objects.rotation == [0.0, 0.0, 0.0, 1.0]
assert (
    dataset.episodes[0].objects.object_template
    == "data/test_assets/objects/chair"
)

assert dataset.episodes[0].goals.position == [
    4.3417439460754395,
    0.6726829409599304,
    -5.0634379386901855,
]
assert dataset.episodes[0].goals.rotation == [0.0, 0.0, 0.0, 1.0]

''' Add Grab/Release Action '''

def raycast(sim, sensor_name, crosshair_pos=(128, 128), max_distance=2.0):
    r"""Cast a ray in the direction of crosshair and check if it collides
    with another object within a certain distance threshold
    :param sim: Simulator object
    :param sensor_name: name of the visual sensor to be used for raycasting
    :param crosshair_pos: 2D coordiante in the viewport towards which the
        ray will be cast
    :param max_distance: distance threshold beyond which objects won't
        be considered
    """
    render_camera = sim._sensors[sensor_name]._sensor_object.render_camera
    center_ray = render_camera.unproject(mn.Vector2i(crosshair_pos))

    raycast_results = sim.cast_ray(center_ray, max_distance=max_distance)

    closest_object = -1
    closest_dist = 1000.0
    if raycast_results.has_hits():
        for hit in raycast_results.hits:
            if hit.ray_distance < closest_dist:
                closest_dist = hit.ray_distance
                closest_object = hit.object_id

    return closest_object

# with habitat_sim.Simulator(cfg) as sim:
#     init_agent(sim)
#     obj_attr_mgr = sim.get_object_template_manager()
#     obj_attr_mgr.load_configs(
#         str(os.path.join(data_path, "test_assets/objects"))
#     )
#     obj_path = "test_assets/objects/chair"
#     chair_template_id = obj_attr_mgr.load_object_configs(
#         str(os.path.join(data_path, obj_path))
#     )[0]
#     chair_attr = obj_attr_mgr.get_template_by_id(chair_template_id)
#     obj_attr_mgr.register_template(chair_attr)
#     object_id = sim.add_object_by_handle(chair_attr.handle)
#     print(f"Chair's object id is {object_id}")
#
#     set_object_in_front_of_agent(sim, object_id, -1.5)
#     sim.set_object_motion_type(
#         habitat_sim.physics.MotionType.STATIC, object_id
#     )
#     if make_video:
#         # Visualize the agent's initial position
#         simulate_and_make_vid(
#             sim, [190, 128], "sim-before-grab", dt=1.0, open_vid=show_video
#         )
#
#     # Distance threshold=2 is greater than agent-to-chair distance.
#     # Should return chair's object id
#     closest_object = raycast(
#         sim, "rgb", crosshair_pos=[128, 190], max_distance=2.0
#     )
#     print(f"Closest Object ID: {closest_object} using 2.0 threshold")
#     assert (
#         closest_object == object_id
#     ), f"Could not pick chair with ID: {object_id}"
#
#     # Distance threshold=1 is smaller than agent-to-chair distance .
#     # Should return -1
#     closest_object = raycast(
#         sim, "rgb", crosshair_pos=[128, 190], max_distance=1.0
#     )
#     print(f"Closest Object ID: {closest_object} using 1.0 threshold")
#     assert closest_object == -1, "Agent shoud not be able to pick any object"


from habitat.core.embodied_task import SimulatorTaskAction
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat_sim.agent.controls.controls import ActuationSpec
from habitat_sim.physics import MotionType

@attr.s(auto_attribs=True, slots=True)
class GrabReleaseActuationSpec(ActuationSpec):
    visual_sensor_name: str = "rgb"
    crosshair_pos: List[int] = [128, 128]
    amount: float = 2.0

@registry.register_action_space_configuration(name="RearrangementActions-v0")
class RearrangementSimV0ActionSpaceConfiguration(
    HabitatSimV1ActionSpaceConfiguration
):
    def __init__(self, config):
        super().__init__(config)
        if not HabitatSimActions.has_action("GRAB_RELEASE"):
            HabitatSimActions.extend_action_space("GRAB_RELEASE")

    def get(self):
        config = super().get()
        new_config = {
            HabitatSimActions.GRAB_RELEASE: habitat_sim.ActionSpec(
                "grab_or_release_object_under_crosshair",
                GrabReleaseActuationSpec(
                    visual_sensor_name=self.config.VISUAL_SENSOR,
                    crosshair_pos=self.config.CROSSHAIR_POS,
                    amount=self.config.GRAB_DISTANCE,
                ),
            )
        }

        config.update(new_config)

        return config


@registry.register_task_action
class GrabOrReleaseAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""This method is called from ``Env`` on each ``step``."""
        return self._sim.step(HabitatSimActions.GRAB_RELEASE)

# _C.TASK.ACTIONS.GRAB_RELEASE = CN()
# _C.TASK.ACTIONS.GRAB_RELEASE.TYPE = "GrabOrReleaseAction"
# _C.SIMULATOR.CROSSHAIR_POS = [128, 160]
# _C.SIMULATOR.GRAB_DISTANCE = 2.0
# _C.SIMULATOR.VISUAL_SENSOR = "rgb"


''' Create Simulator '''

from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_sim.nav import NavMeshSettings
from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum

@registry.register_simulator(name="RearrangementSim-v0")
class RearrangementSim(HabitatSim):
    r"""Simulator wrapper over habitat-sim with
    object rearrangement functionalities.
    """

    def __init__(self, config: Config) -> None:
        self.did_reset = False
        super().__init__(config=config)
        self.grip_offset = np.eye(4)

        agent_id = self.habitat_config.default_agent_id
        agent_config = self._get_agent_config(agent_id)

        self.navmesh_settings = NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_radius = agent_config.radius
        self.navmesh_settings.agent_height = agent_config.height

    def reconfigure(self, config: Config) -> None:
        super().reconfigure(config)
        self._initialize_objects()

    def reset(self):
        sim_obs = super().reset()
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations()

        self._prev_sim_obs = sim_obs
        self.did_reset = True
        self.grip_offset = np.eye(4)
        return self._sensor_suite.get_observations(sim_obs)

    def _initialize_objects(self):
        objects = self.habitat_config.objects[0]
        obj_attr_mgr = self.get_object_template_manager()
        obj_attr_mgr.load_configs(
            str(os.path.join(data_path, "test_assets/objects"))
        )
        # first remove all existing objects
        existing_object_ids = self.get_existing_object_ids()

        if len(existing_object_ids) > 0:
            for obj_id in existing_object_ids:
                self.remove_object(obj_id)

        self.sim_object_to_objid_mapping = {}
        self.objid_to_sim_object_mapping = {}

        if objects is not None:
            object_template = objects["object_template"]
            object_pos = objects["position"]
            object_rot = objects["rotation"]

            object_template_id = obj_attr_mgr.load_object_configs(
                object_template
            )[0]
            object_attr = obj_attr_mgr.get_template_by_id(object_template_id)
            obj_attr_mgr.register_template(object_attr)

            object_id = self.add_object_by_handle(object_attr.handle)
            self.sim_object_to_objid_mapping[object_id] = objects["object_id"]
            self.objid_to_sim_object_mapping[objects["object_id"]] = object_id

            self.set_translation(object_pos, object_id)
            if isinstance(object_rot, list):
                object_rot = quat_from_coeffs(object_rot)

            object_rot = quat_to_magnum(object_rot)
            self.set_rotation(object_rot, object_id)

            self.set_object_motion_type(MotionType.STATIC, object_id)

        # Recompute the navmesh after placing all the objects.
        self.recompute_navmesh(self.pathfinder, self.navmesh_settings, True)

    def _sync_gripped_object(self, gripped_object_id):
        r"""
        Sync the gripped object with the object associated with the agent.
        """
        if gripped_object_id != -1:
            agent_body_transformation = (
                self._default_agent.scene_node.transformation
            )
            self.set_transformation(
                agent_body_transformation, gripped_object_id
            )
            translation = agent_body_transformation.transform_point(
                np.array([0, 2.0, 0])
            )
            self.set_translation(translation, gripped_object_id)

    @property
    def gripped_object_id(self):
        return self._prev_sim_obs.get("gripped_object_id", -1)

    def step(self, action: int):
        dt = 1 / 60.0
        self._num_total_frames += 1
        collided = False
        gripped_object_id = self.gripped_object_id

        agent_config = self._default_agent.agent_config
        action_spec = agent_config.action_space[action]

        if action_spec.name == "grab_or_release_object_under_crosshair":
            # If already holding an agent
            if gripped_object_id != -1:
                agent_body_transformation = (
                    self._default_agent.scene_node.transformation
                )
                T = np.dot(agent_body_transformation, self.grip_offset)

                self.set_transformation(T, gripped_object_id)

                position = self.get_translation(gripped_object_id)

                if self.pathfinder.is_navigable(position):
                    self.set_object_motion_type(
                        MotionType.STATIC, gripped_object_id
                    )
                    gripped_object_id = -1
                    self.recompute_navmesh(
                        self.pathfinder, self.navmesh_settings, True
                    )
            # if not holding an object, then try to grab
            else:
                gripped_object_id = raycast(
                    self,
                    action_spec.actuation.visual_sensor_name,
                    crosshair_pos=action_spec.actuation.crosshair_pos,
                    max_distance=action_spec.actuation.amount,
                )

                # found a grabbable object.
                if gripped_object_id != -1:
                    agent_body_transformation = (
                        self._default_agent.scene_node.transformation
                    )

                    self.grip_offset = np.dot(
                        np.array(agent_body_transformation.inverted()),
                        np.array(self.get_transformation(gripped_object_id)),
                    )
                    self.set_object_motion_type(
                        MotionType.KINEMATIC, gripped_object_id
                    )
                    self.recompute_navmesh(
                        self.pathfinder, self.navmesh_settings, True
                    )

        else:
            collided = self._default_agent.act(action)
            self._last_state = self._default_agent.get_state()

        # step physics by dt
        super().step_world(dt)

        # Sync the gripped object after the agent moves.
        self._sync_gripped_object(gripped_object_id)

        # obtain observations
        self._prev_sim_obs = self.get_sensor_observations()
        self._prev_sim_obs["collided"] = collided
        self._prev_sim_obs["gripped_object_id"] = gripped_object_id

        observations = self._sensor_suite.get_observations(self._prev_sim_obs)
        return observations

'''-----------------------------------------------------------------------------------------'''

''' Creating New Task '''

from gym import spaces

import habitat_sim
# from habitat.config.default import CN, Config
from habitat.core.dataset import Episode
from habitat.core.embodied_task import Measure
from habitat.core.simulator import Observations, Sensor, SensorTypes, Simulator
from habitat.tasks.nav.nav import PointGoalSensor

# @markdown **Sensors**:
# @markdown - Object's current position will be made given by the `ObjectPosition` sensor
# @markdown - Object's goal position will be available through the `ObjectGoal` sensor.
# @markdown - Finally, we will also use `GrippedObject` sensor to tell the agent if it's holding any object or not.

# @markdown **Measures**: These define various metrics about the task which can be used to measure task progress and define rewards. Note that measurements are *privileged* information not accessible to the agent as part of the observation space. We will need the following measurements:
# @markdown - `AgentToObjectDistance` which measure the euclidean distance between the agent and the object.
# @markdown - `ObjectToGoalDistance`

@registry.register_sensor
class GrippedObjectSensor(Sensor):
    cls_uuid = "gripped_object_id"

    def __init__(
        self, *args: Any, sim: RearrangementSim, config: Config, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any):

        return spaces.Discrete(len(self._sim.get_existing_object_ids()))

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def get_observation(
        self,
        observations: Dict[str, Observations],
        episode: Episode,
        *args: Any,
        **kwargs: Any,
    ):
        obj_id = self._sim.sim_object_to_objid_mapping.get(
            self._sim.gripped_object_id, -1
        )
        return obj_id

@registry.register_sensor
class ObjectPosition(PointGoalSensor):
    cls_uuid: str = "object_position"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation

        object_id = self._sim.get_existing_object_ids()[0]
        object_position = self._sim.get_translation(object_id)
        pointgoal = self._compute_pointgoal(
            agent_position, rotation_world_agent, object_position
        )
        return pointgoal


@registry.register_sensor
class ObjectGoal(PointGoalSensor):
    cls_uuid: str = "object_goal"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation

        goal_position = np.array(episode.goals.position, dtype=np.float32)

        point_goal = self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )
        return point_goal


@registry.register_measure
class ObjectToGoalDistance(Measure):
    """The measure calculates distance of object towards the goal."""

    cls_uuid: str = "object_to_goal_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return ObjectToGoalDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _geo_dist(self, src_pos, goal_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [goal_pos])

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        sim_obj_id = self._sim.get_existing_object_ids()[0]

        previous_position = np.array(
            self._sim.get_translation(sim_obj_id)
        ).tolist()
        goal_position = episode.goals.position
        self._metric = self._euclidean_distance(
            previous_position, goal_position
        )


@registry.register_measure
class AgentToObjectDistance(Measure):
    """The measure calculates the distance of objects from the agent"""

    cls_uuid: str = "agent_to_object_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return AgentToObjectDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        sim_obj_id = self._sim.get_existing_object_ids()[0]
        previous_position = np.array(
            self._sim.get_translation(sim_obj_id)
        ).tolist()

        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position

        self._metric = self._euclidean_distance(
            previous_position, agent_position
        )


from habitat.tasks.nav.nav import NavigationTask, merge_sim_episode_config


# # -----------------------------------------------------------------------------
# # # REARRANGEMENT TASK GRIPPED OBJECT SENSOR
# # -----------------------------------------------------------------------------
# _C.TASK.GRIPPED_OBJECT_SENSOR = CN()
# _C.TASK.GRIPPED_OBJECT_SENSOR.TYPE = "GrippedObjectSensor"
# # -----------------------------------------------------------------------------
# # # REARRANGEMENT TASK ALL OBJECT POSITIONS SENSOR
# # -----------------------------------------------------------------------------
# _C.TASK.OBJECT_POSITION = CN()
# _C.TASK.OBJECT_POSITION.TYPE = "ObjectPosition"
# _C.TASK.OBJECT_POSITION.GOAL_FORMAT = "POLAR"
# _C.TASK.OBJECT_POSITION.DIMENSIONALITY = 2
# # -----------------------------------------------------------------------------
# # # REARRANGEMENT TASK ALL OBJECT GOALS SENSOR
# # -----------------------------------------------------------------------------
# _C.TASK.OBJECT_GOAL = CN()
# _C.TASK.OBJECT_GOAL.TYPE = "ObjectGoal"
# _C.TASK.OBJECT_GOAL.GOAL_FORMAT = "POLAR"
# _C.TASK.OBJECT_GOAL.DIMENSIONALITY = 2
# # -----------------------------------------------------------------------------
# # # OBJECT_DISTANCE_TO_GOAL MEASUREMENT
# # -----------------------------------------------------------------------------
# _C.TASK.OBJECT_TO_GOAL_DISTANCE = CN()
# _C.TASK.OBJECT_TO_GOAL_DISTANCE.TYPE = "ObjectToGoalDistance"
# # -----------------------------------------------------------------------------
# # # OBJECT_DISTANCE_FROM_AGENT MEASUREMENT
# # -----------------------------------------------------------------------------
# _C.TASK.AGENT_TO_OBJECT_DISTANCE = CN()
# _C.TASK.AGENT_TO_OBJECT_DISTANCE.TYPE = "AgentToObjectDistance"
#
# from habitat.config.default import CN, Config

def merge_sim_episode_with_object_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config = merge_sim_episode_config(sim_config, episode)
    sim_config.defrost()
    sim_config.objects = [episode.objects.__dict__]
    sim_config.freeze()

    return sim_config


@registry.register_task(name="RearrangementTask-v0")
class RearrangementTask(NavigationTask):
    r"""Embodied Rearrangement Task
    Goal: An agent must place objects at their corresponding goal position.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_object_config(sim_config, episode)

'''-----------------------------------------------------------------------------------------'''
