import glob
import gzip
import json
import multiprocessing as mp
import os

import tqdm

import habitat
import habitat_sim
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode
from habitat.config.default import read_write

num_episodes_per_scene = int(100)


# test_scene = 'data/scene_datasets/habitat-test-scenes/apartment_1.glb'
# test_scene = '/home/pgp/autonomous_exploration_development_environment/src/vehicle_simulator/mesh/forest/preview/pointcloud.ply'
# test_scene = '/media/pgp/U330/毕设/ImageToStl.com_meshed-poisson.glb'
# test_scene = '/media/pgp/U330/毕设/objglb/scene.glb'
# test_scene = '/media/pgp/U330/毕设/objglb/mesh_semantic.obj'
# test_scene = '/media/pgp/U330/毕设/ImageToStl.com_meshed-poisson.glb'
# test_scene = '/media/pgp/U330/毕设/objglb/output/output.glb'

test_scene = '/media/pgp/U330/毕设/office_1/habitat/mesh_semantic.glb'


sim_settings = {
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
    "scene": test_scene,  # Scene path
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": True,  # RGB sensor
    "semantic_sensor": True,  # Semantic sensor
    "depth_sensor": True,  # Depth sensor
    "seed": 1,
}

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
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

    # todo amout指的是控制量，但是量纲是什么？
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
    return habitat_sim.Configuration(sim_cfg,[agent_cfg])

# cfg = make_cfg(sim_settings)
# sim = habitat_sim.Simulator(cfg)
# sim = habitat.sims.make_sim('Sim-v0', config=cfg)

def _generate_fn(scene):
    cfg = habitat.get_config(config_path='benchmark/nav/pointnav/pointnav_base.yaml')
    with read_write(cfg):
        cfg.habitat.simulator.scene = scene
        cfg.habitat.simulator.agents.main_agent.sim_sensors = {}

    # for i in cfg.habitat:
    #     print(i,cfg.habitat[i])

    sim = habitat.sims.make_sim('Sim-v0', config=cfg.habitat.simulator)
    navmesh_settings = habitat_sim.NavMeshSettings()
    sim.recompute_navmesh(
        sim.pathfinder, navmesh_settings, include_static_objects=False
    )

    dataset = habitat.datasets.make_dataset('PointNav-v1')
    dataset.episodes = list(
        generate_pointnav_episode(sim, num_episodes_per_scene, is_gen_shortest_path=False)
    )
    # print(dataset.episodes[0].scene_id) # /media/pgp/U330/毕设/office_1/habitat/mesh_semantic.ply
    for e in dataset.episodes:
        print(e.scene_id)
        e.scene_id = e.scene_id[len('/media/pgp/U330/毕设/office_1/'):]

    scene_key = os.path.splitext(os.path.basename(scene))[0]
    # print(scene_key) #　mesh_semantic
    out_file = f"./mydata/datasets/pointnav/mydata/v1/all/content/{scene_key}.json.gz"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    out_file_origin = f"./mydata/datasets/pointnav/mydata/v1/all/content/{scene_key}.json"

    with gzip.open(out_file, "wt") as f:
        f.write(dataset.to_json())

    with open(out_file_origin, "wt") as f:
        f.write(dataset.to_json())

# scenes = glob.glob("./data/scene_datasets/gibson/*.glb")
# with mp.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
#     for _ in pool.imap_unordered(_generate_fn, scenes):
#         pbar.update()

_generate_fn(test_scene)

with gzip.open(f"./mydata/datasets/pointnav/mydata/v1/all/all.json.gz", "wt") as f:
    json.dump(dict(episodes=[]), f)
with open(f"./mydata/datasets/pointnav/mydata/v1/all/all.json", "wt") as f:
    json.dump(dict(episodes=[]), f)

# a = None
# with gzip.open(f"/home/pgp/habitat/ANM/data/datasets/pointnav/gibson/v1/val/val.json.gz", "rb") as f:
#     # for i in f:
#     #     print('\n')
#     #     json.loads(i)
#     a = json.load(f)
#
# print(a)


# cfg = habitat.get_config(config_path='benchmark/nav/pointnav/pointnav_mydata.yaml')
#
# absolute_path_prefix = __file__[:__file__.rfind('/')]+'/'
# print(absolute_path_prefix)
#
# with read_write(cfg):
#     cfg.habitat.dataset.split = 'all'
#     cfg.habitat.dataset.data_path = absolute_path_prefix + cfg.habitat.dataset.data_path
#     cfg.habitat.dataset.scenes_dir = absolute_path_prefix + cfg.habitat.dataset.scenes_dir
#
# from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
#
#
# scenes = PointNavDatasetV1(cfg.habitat.dataset)
# print(scenes)