import habitat_sim
import random
import matplotlib.pyplot as plt

import numpy as np

# test_scene = 'data/scene_datasets/habitat-test-scenes/apartment_1.glb'
# test_scene = '/home/pgp/autonomous_exploration_development_environment/src/vehicle_simulator/mesh/forest/preview/pointcloud.ply'
# test_scene = '/media/pgp/U330/毕设/ImageToStl.com_meshed-poisson.glb'
# test_scene = '/media/pgp/U330/毕设/objglb/scene.glb'
# test_scene = '/media/pgp/U330/毕设/objglb/mesh_semantic.obj'
test_scene = '/media/pgp/U330/毕设/office_1/habitat/mesh_semantic.ply'


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

cfg = make_cfg(sim_settings)
print('cfg: ', cfg)
sim = habitat_sim.Simulator(cfg)
exit()
def print_scene_recur(scene, limit_output=10):
    print(f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
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
                    return None

# Print semantic annotation information (id, category, bounding box details)
# about levels, regions and objects in a hierarchical fashion

# scene = sim.semantic_scene
# print_scene_recur(scene)



random.seed(sim_settings["seed"])
sim.seed(sim_settings["seed"])

# Set agent state
agent = sim.initialize_agent(sim_settings["default_agent"])
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([0.0, 0.072447, 0.0])
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)




from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb

def display_sample(rgb_obs, semantic_obs, depth_obs):
    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")

    depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")

    arr = [rgb_img, semantic_img, depth_img]
    titles = ['rgb', 'semantic', 'depth']
    plt.figure(figsize=(12 ,8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i+1)
        ax.axis('off')
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show()


total_frames = 0
action_names = list(
    cfg.agents[
        sim_settings["default_agent"]
    ].action_space.keys()
)

max_frames = 5

while total_frames < max_frames:
    action = random.choice(action_names)
    print("action", action)
    observations = sim.step(action)
    rgb = observations["color_sensor"]
    semantic = observations["semantic_sensor"]
    depth = observations["depth_sensor"]

    display_sample(rgb, semantic, depth)

    total_frames += 1

