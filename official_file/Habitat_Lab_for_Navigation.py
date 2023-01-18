import logging
import os
import random
import sys
from typing import Any

import numpy as np
from gym import spaces
from PIL import Image
from matplotlib import pyplot as plt


import habitat
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationTask
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config as get_baselines_config



def display_sample(
    rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])
):  # noqa: B006
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
        plt.imshow(data)
    plt.show(block=False)

''' ************************ Step PointNav Task ************************ '''

if __name__ == "__main__":
    config = habitat.get_config(
        config_path="./test/habitat_all_sensors_test.yaml"
    )

    try:
        env.close()  # type: ignore[has-type]
    except NameError:
        pass
    env = habitat.Env(config=config)


    action = None
    obs = env.reset()
    valid_actions = ["turn_left", "turn_right", "move_forward", "stop"]
    interactive_control = False  # @param {type:"boolean"}
    while action != "stop":
        display_sample(obs["rgb"])
        print("distance to goal: {:.2f}".format(obs["pointgoal_with_gps_compass"][0]))
        print("angle to goal (radians): {:.2f}".format(obs["pointgoal_with_gps_compass"][1]))

        if interactive_control:
            action = input("enter action out of {}:\n".format(", ".join(valid_actions)))
            assert (action in valid_actions), "invalid action {} entered, choose one amongst " + ",".join(valid_actions)
        else:
            action = valid_actions.pop()
        obs = env.step(
            {"action": action}
        )

    env.close()
    print(env.get_metrics())

''' ************************ RL Training ************************ '''

if __name__ == "__main__":

    config = get_baselines_config("pointnav/ppo_pointnav_example.yaml")

    seed = "42"
    steps_in_thousands = "10"

    with habitat.config.read_write(config):
        config.habitat.seed = int(seed)
        config.habitat_baselines.total_num_steps = int(steps_in_thousands)
        config.habitat_baselines.log_interval = 1

    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)

    trainer_init = baseline_registry.get_trainer(
        config.habitat_baselines.trainer_name
    )
    trainer = trainer_init(config)
    trainer.train()


''' ************************ Create New Task ************************ '''

if __name__ == "__main__":
    config = habitat.get_config(
        config_path="./test/habitat_all_sensors_test.yaml"
    )

@registry.register_task(name='TestNav-v0')
class NewNavigationTask(NavigationTask):
    def __init__(self, config, sim, dataset):
        # key: official info
        logger.info('Create a new type of task')
        super().__init__(config=config, sim=sim, dataset=dataset)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        logging.info("Current agent position: {}".format(self._sim.get_agent_state()))
        collision = self._sim.previous_step_collided
        stop_called = not getattr(self, 'is_stop_called', False)
        return collision or stop_called

if __name__ == "__main__":

    # key: 如何用新的task
    with habitat.config.read_write(config):
        config.habitat.task.type = "TestNav-v0"

    try:
        env.close()
    except NameError:
        pass
    env = habitat.Env(config=config)

    action = None
    env.reset()
    valid_actions = ["turn_left", "turn_right", "move_forward", "stop"]
    interactive_control = False  # @param {type:"boolean"}
    while env.episode_over is not True:
        display_sample(obs["rgb"])
        if interactive_control:
            action = input("enter action out of {}:\n".format(", ".join(valid_actions)))
            assert (action in valid_actions), "invalid action {} entered, choose one amongst " + ",".join(
                valid_actions)
        else:
            action = valid_actions.pop()
        obs = env.step(
            {
                "action": action,
                "action_args": None,
            }
        )
        print("Episode over:", env.episode_over)

    env.close()

@registry.register_sensor(name='agent_position_sensor')
class AgentPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return 'agent_position'

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def get_observation(self, observations, *args, episode, **kwargs):
        '''defalut: return agent_0 position'''
        return self._sim.get_agent_state().position

if __name__ == "__main__":
    config = habitat.get_config(
        config_path="./test/habitat_all_sensors_test.yaml"
    )

    from habitat.config.default_structured_configs import LabSensorConfig

    # We use the base sensor config, but you could also define your own
    # AgentPositionSensorConfig that inherits from LabSensorConfig

    # key: 如何用新的sensor
    with habitat.config.read_write(config):
        # Now define the config for the sensor
        config.habitat.task.lab_sensors[
            "agent_position_sensor"
        ] = LabSensorConfig(type="agent_position_sensor")

    try:
        env.close()
    except NameError:
        pass
    env = habitat.Env(config=config)
    obs = env.reset()
    obs.keys()
    print(obs["agent_position"])
    env.close()


class ForwardOnlyAgent(habitat.Agent):
    def __init__(self, success_distance, goal_sensor_uuid):
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid

    def reset(self):
        pass

    def is_goal_reached(self, observations):
        dist = observations[self.goal_sensor_uuid][0]
        return dist <= self.dist_threshold_to_stop

    def act(self, observations):
        if self.is_goal_reached(observations):
            action = HabitatSimActions.stop
        else:
            action = HabitatSimActions.move_forward
        return {"action": action}


''' ************************ Sim2Real ************************ '''

#Deploy habitat-sim trained models on real robots with the habitat-pyrobot bridge
#Paper: https://arxiv.org/abs/1912.06321
# Are we in sim or reality?

# if args.use_robot: # Use LoCoBot via PyRobot
#     config.habitat.simulator.type = "PyRobot-Locobot-v0"
# else: # Use simulation
#     config.habitat.simulator.type = "Habitat-Sim-v0"git

