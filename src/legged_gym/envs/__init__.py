from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .hopper.hopper import HopperRobot
from .hopper.hopper_config import HopperCfg, HopperCfgPPO


import os

from legged_gym.utils.task_registry import task_registry

# Task name string is used as a CLI argument to select the task, experiment name (in cfg) defines folder name
task_registry.register( "hopper", HopperRobot, HopperCfg(), HopperCfgPPO() )
