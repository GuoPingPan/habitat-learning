# import re

# # 在遇到字符从大写变到小写的时候，利用大写字母作为开头重新开始下一段
# def _camel_to_snake(name):
#     s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
#     return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


# s = 'FasdffferfFSDSfdsf'

# print(_camel_to_snake(name=s))

# import habitat
# from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
# from habitat.config import read_write
#
# config = habitat.get_config(config_path='benchmark/nav/pointnav/pointnav_gibson.yaml')
# # print(cfg['habitat']['dataset']['pointnav'])
# env = habitat.Env(config)
# scene = PointNavDatasetV1.get_scenes_to_load(config.habitat.dataset)
#
# from omegaconf import DictConfig
#
# a = DictConfig({'af': 0})
#
# b = ['rgb', 'depth']
# a.simulator = b
# a.simulator.rgb = 10
# print(a)

# import torch
# import torchvision.models as models
# import torch.nn as nn
#
# resnet = models.resnet18(pretrained=True)
# resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])
# a = torch.rand(1,3,128,128)
# b = resnet_l5(a)
#
# print(a.shape)
# print(b.shape)

# import  numpy as np
#
# a = np.random.rand(10,10)
# b = [a,a,a,a]
# stg = np.stack(b, axis=0)
# print(stg.shape)
#
# b = ['q','q','q','q']
# a,d,f,g = b
# print(a)
#
# import torch
#
# a = torch.tensor(10).float()
# print(a)
# b = [a for i in range(10)]
# c = torch.stack(b, dim=0)
# print(c.shape)


# b = []
# a = torch.rand(10,10)
# for i in range(10):
#     b.append(a)
#
# c = torch.stack(b,dim=0)
# print(c.shape)


# from omegaconf.dictconfig import DictConfig
#
# a = DictConfig({})
# print(a._matedata)

# from a import b
# print(b)

# print(a)
# a.setdefault('task',{'fdas': 'fdsaf'})
#
# print(a)

# import numpy as np
#
# a = np.zeros([10,19])
# print(a.sum(1).shape)

# import skfmm
# import numpy as np
#
# def get_mask(sx, sy, scale, step_size):
#     size = int(step_size // scale) * 2 + 1
#     mask = np.zeros((size, size))
#     for i in range(size):
#         for j in range(size):
#             if ((i + 0.5) - (size // 2 + sx)) ** 2 + ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
#                     step_size ** 2:
#                 mask[i, j] = 1
#     return mask
#
# mask = get_mask(0,0,1,5)
# traversible_ma = np.ones((11,11))
# traversible_ma[5,5] = 0
# dist = skfmm.distance(traversible_ma, dx=1)
# dist *= mask
# dist -= dist[5,5]
# print(dist)

# noise_model = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
#
# print(noise_model) # 初始化为一个 object
# noise_model = 10
# print(noise_model)
#
# noise_model = 20.0
# print(noise_model)
# del noise_model
#
# a = lambda self,v: None
# b = 10
# b = a(b,20)
# print(b)

import gym
import habitat.utils.gym_definitions

env = gym.make("HabitatPick-v0")
print(
    "Pick observation space",
    {k: v.shape for k, v in env.observation_space.spaces.items()},
)
env.close()

# Array observation space
env = gym.make("HabitatReachState-v0")
print("Reach observation space", env.observation_space)
env.close()