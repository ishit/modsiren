"""Configuration manager."""
import os
import dataclasses
import typing
import logging

import git
from omegaconf import OmegaConf

logging.getLogger("git.cmd").setLevel(logging.ERROR)


@dataclasses.dataclass
class TrainingConfig:
    lr: float = 1e-4
    bs: int = 64
    val_bs: int = 64
    workers: int = 8
    # scheduler_step: int = 60
    # scheduler_gamma: float = 0.5

    loss: str = "l2"


@dataclasses.dataclass
class DefaultConfig:
    name: str = 'unnamed'
    git_commit: str = ''

    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    task: str = 'single_instance'
    domain: str = 'image'

    # Path to the dataset.
    dataset: str = ""
    # Data-specific parameters (see the corresponding class in datasets.py).
    data: dict = dataclasses.field(default_factory=dict)

    # Specify these configs if the train/val configuration need to override
    # certain paraemeters in `data`.
    train_data: dict = dataclasses.field(default_factory=dict)
    val_data: dict = dataclasses.field(default_factory=dict)

    # Model class in `models.py`.
    model: str = ""
    # Model-specific parameters (see the corresponding class in models.py).
    model_params: dict = dataclasses.field(default_factory=dict)

    # Path to a checkpoint from which to initialize the weights (for transfer
    # learning, etc).
    init_from: typing.Optional[str] = None


def default_config():
    return OmegaConf.structured(DefaultConfig)


def merge(conf1, conf2):
    return OmegaConf.merge(conf1, conf2)


def get_config(cli_args=None, filepath=None):
    conf = default_config()

    # Add git hash to conf
    repo = git.Repo(os.path.join(os.path.dirname(__file__), os.pardir))
    conf["git_commit"] = repo.head.object.hexsha

    if filepath is not None:
        file_conf = OmegaConf.load(filepath)
        conf = merge(conf, file_conf)

    if filepath is not None:
        cli_conf = OmegaConf.from_dotlist(cli_args)
        conf = merge(conf, cli_conf)

    conf.train_data = OmegaConf.merge(conf.data, conf.train_data)
    conf.val_data = OmegaConf.merge(conf.data, conf.val_data)

    return conf
