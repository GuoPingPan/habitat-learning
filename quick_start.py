import habitat
import habitat_sim.sensor
import quaternion
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2

forward_key = 'w'
left_key = 'a'
right_key = 'd'
backward_key = 's'
finish = 'f'

def transform_rgb_bgr(image):
    return image[:,:,[2,1,0]]


def example():

    # config = habitat.get_config('test/habitat_all_sensors_test.yaml')    

    # print(type(config))
    # # print(config.task_config_base)
    # print(len(config.items()))
    # for key,val in config['habitat'].items():
    #     print(f"{key} : {val}")

    env = habitat.Env(
        config=habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
        )

    print("Environment creation successful")


    observations = env.reset()

    # print(f"obs : {type(observation)},{observation} ")

    for key in observations.keys():
        print(key)
    # rgb
    # depth
    # pointgoal_with_gps_compass


    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
    cv2.imshow("Depth", observations["depth"])

    # print(habitat_sim.sensor.VisualSensor.far)

    count_steps = 0
    while not env.episode_over:
        key = cv2.waitKey(0)
        # ------------ Default action ------------ #
        # stop = 0
        # move_forward = 1
        # turn_left = 2
        # turn_right = 3
        # look_up = 4
        # look_down = 5
        # ------------ Default action ------------ #

        # 添加自定义action
        # HabitatSimActions.has_action()
        # HabitatSimActions.extend_action_space()

        if key == ord(forward_key):
            action = HabitatSimActions.move_forward
            print('move_forward')
        elif key == ord(left_key):
            action = HabitatSimActions.turn_left
            print('turn_left')
        elif key == ord(right_key):
            action = HabitatSimActions.turn_right
            print('turn_right')
        # elif key == ord(backward_key):
        #     action = HabitatSimActions.move_backward
        #     print('move_backward')
        elif key == ord(finish):
            action = HabitatSimActions.stop
            print('stop')
        else:
            print('invalid key')
            continue

        observations = env.step(action)
        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0],
            observations["pointgoal_with_gps_compass"][1]))
        state = env.sim.get_agent_state(0)
        print(f"Agent stage: position = {state.position} rotation = {state.rotation}")
        print(f"Agent stage: xyz angle = {quaternion.as_euler_angles(state.rotation)}")

        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
        cv2.imshow("Depth", observations["depth"])

        print("Episode finished after {} steps.".format(count_steps))

        # 这里是直接返回距离point goal的距离
        if (action == HabitatSimActions.stop and observations["pointgoal_with_gps_compass"][0] < 0.2):
            print("you successfully navigated to destination point")
        else:
            print("your navigation was unsuccessful")

if __name__=='__main__':
    example()