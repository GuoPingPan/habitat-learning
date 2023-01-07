import habitat
from habitat.config import read_write


config = habitat.get_config(config_path='benchmark/nav/pointnav/pointnav_gibson.yaml')
# print(cfg['habitat']['dataset']['pointnav'])

with read_write(config):
    config.habitat.dataset.content_scenes = ['Bowlus']
    config.habitat.dataset.scenes_dir = '/home/pgp/habitat/ANM/' + config.habitat.dataset.scenes_dir
    config.habitat.dataset.data_path = '/home/pgp/habitat/ANM/' + config.habitat.dataset.data_path

env = habitat.Env(config)
