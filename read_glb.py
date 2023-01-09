
''' quaternion '''
import quaternion
import numpy as np

# a = quaternion.quaternion()
# a.w = 0.923870146274567
# a.x = 0
# a.y = 0.382706195116043
# a.z = 0
#
# print(quaternion.as_euler_angles(a))
# print(np.rad2deg(quaternion.as_euler_angles(a)[1]))

''' np.bincount '''
# print(np.bincount([1,2,3,4]))


''' skfmm '''
# from numpy import ma
# import skfmm
#
# a = np.array(
#     [[1,0,0,0,1],
#      [1,-1,0,0,1],
#      [0,0,1,0,1],
#      [0,0,0,0,1],
#      [0,1,0,0,1]]
# )
#
# tra = ma.masked_values(a, 0)
# goal_x, goal_y = 3,3
# tra[goal_x,goal_y] = 0
#
# print(tra)
# print(type(tra))
#
# dd = skfmm.distance(tra, dx=1)
# print(dd)
# dd_mask = np.invert(np.isnan(ma.filled(dd,np.nan)))
# print(dd_mask)
#
# dd = ma.filled(dd,np.max(dd)+1)
# print(dd)
#
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
# def get_dist(sx, sy, scale, step_size):
#     size = int(step_size // scale) * 2 + 1
#     mask = np.zeros((size, size)) + 1e-10
#     for i in range(size):
#         for j in range(size):
#             if ((i + 0.5) - (size // 2 + sx)) ** 2 + ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
#                     step_size ** 2:
#                 mask[i, j] = max(5, (((i + 0.5) - (size // 2 + sx)) ** 2 +
#                                      ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
#     return mask
#
# dx = 0.5
# dy = 0.5
# mark = get_mask(dx,dy,1,5)
# print(mark)
#
# a = [1, 2, 3, 4, 5]
# print(np.pad(a, 3, 'constant', constant_values=(4, 6)))
#
# planner_pose_inputs = np.array([5, 5, 5, 0, 0, 120, 120]).reshape(1, -1).repeat(2, axis=0)
# print(planner_pose_inputs.shape)
#
#
# def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
#     '''
#         获得局部地图的左上角点和右下角点
#         主要是通过以当前位置和local_size去切割出一个local_map
#     '''
#     loc_r, loc_c = agent_loc
#     local_w, local_h = local_sizes
#     full_w, full_h = full_sizes
#
#     if 2 > 1:
#         gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
#         gx2, gy2 = gx1 + local_w, gy1 + local_h
#         if gx1 < 0:
#             gx1, gx2 = 0, local_w
#         if gx2 > full_w:
#             gx1, gx2 = full_w - local_w, full_w
#
#         if gy1 < 0:
#             gy1, gy2 = 0, local_h
#         if gy2 > full_h:
#             gy1, gy2 = full_h - local_h, full_h
#     else:
#         gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h
#
#     return [gx1, gx2, gy1, gy2]
#
#
# print(get_local_map_boundaries((240,240),(240,240),(480,480)))
#
# a = [0,0,0,1,1,1,0,0,0,0,0]
# print(np.flip(a))
# print(np.argmax(np.flip(a)))
#
# b = np.ones((11,11))
# b[5,5] = 0
# dd = skfmm.distance(b,dx=1)
# print(dd)

''' property '''
class Person:
    def __init__(self, name, sex):
        self._name = name
        self._sex = sex

    @property
    def name(self):
        return self._name
    @property
    def sex(self):
        return self._sex
    @name.setter
    def name(self, name):
        self._name = name


    name_sex = property(name,name)

person = Person('a','female')

print(person.name)
print(person.sex)
# print(type(person.name_sex))
person.name = 'fda'
# person.sex = 'fda'
# person.name_sex = 'fadsf'
print(person.name_sex.name)