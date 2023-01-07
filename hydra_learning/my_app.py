from omegaconf import DictConfig, OmegaConf
import hydra

# hydra.main and hydra.initialize version_base
# https://hydra.cc/docs/upgrades/version_base/
# @hydra.main(version_base=None, config_name="config", config_path='')
# def my_app(cfg: DictConfig) -> None:
#     print(cfg.node.zippity)
#     print(OmegaConf.to_yaml(cfg))


@hydra.main(version_base=None, config_path='conf', config_name='config')
def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))




if __name__ == '__main__':
    my_app()
    import habitat
    habitat.get_config(config_path='benchmark/nav/pointnav/test.yaml')