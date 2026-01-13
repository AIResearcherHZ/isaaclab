# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Taks_T1 AMP locomotion environment with velocity command control.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-TaksT1-AMP-Walk-Direct-v0",
    entry_point=f"{__name__}.taks_t1_amp_env:TaksT1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.taks_t1_amp_env_cfg:TaksT1AmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-TaksT1-AMP-Run-Direct-v0",
    entry_point=f"{__name__}.taks_t1_amp_env:TaksT1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.taks_t1_amp_env_cfg:TaksT1AmpRunEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_run_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-TaksT1-AMP-Dance-Direct-v0",
    entry_point=f"{__name__}.taks_t1_amp_env:TaksT1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.taks_t1_amp_env_cfg:TaksT1AmpDanceEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_dance_amp_cfg.yaml",
    },
)