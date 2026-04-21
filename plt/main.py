import os
import hydra
import wandb
import torch
import numpy as np 
import random

from plt.utils import random_seed
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from plt.splitter.cross_validator import CrossValidator

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("tuple", lambda x: tuple(x))

@hydra.main(version_base=None, config_path="../config", config_name="experiment")
def main(cfg: DictConfig):
    random_seed(cfg.seed)

    torch.use_deterministic_algorithms(True)

    if cfg.wandb:
        sweep_run = wandb.init(project="plate_experiment", config=OmegaConf.to_container(cfg, resolve=True))

    data = instantiate(cfg["data"])
    trainer = instantiate(cfg["trainer"], _recursive_=False)

    cross_validator = CrossValidator(data, trainer, cfg.models)
    results = cross_validator.cross_validate()

    return results

if __name__ == "__main__":
    main()
