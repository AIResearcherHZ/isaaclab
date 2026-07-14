# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo: drive the Semi_Taks_LV1 closed-chain robot with sinusoidal joint targets.

Run with GUI to visually verify the closed-chain loops:

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/semi_taks_lv1.py

Watch for:
- waist motors <-> waist roll/pitch following through the parallel linkage
- arm linkage motors <-> wrist pitch/yaw following through the four-bar + bevel gears
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Demo for the Semi_Taks_LV1 closed-chain robot.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from isaaclab_assets.robots.taks import SEMI_TAKS_LV1_CFG

MOTION_GROUPS = {
    "waist_yaw_joint": (0.6, 0.15, 0.0),
    "waist_right_motor_joint": (0.35, 0.25, 0.0),
    "waist_left_motor_joint": (0.35, 0.25, math.pi / 2),
    ".*_shoulder_pitch_joint": (0.4, 0.2, 0.0),
    ".*_shoulder_roll_joint": (0.3, 0.2, 0.0),
    ".*_elbow_joint": (0.4, 0.25, 0.0),
    ".*_wrist_roll_joint": (0.8, 0.4, 0.0),
    ".*_arm_long_link_motor_joint": (0.5, 0.35, 0.0),
    ".*_arm_short_link_motor_joint": (0.5, 0.35, math.pi),
}

REPORT_PAIRS = [
    ("waist_right_motor_joint", "waist_pitch_joint"),
    ("waist_left_motor_joint", "waist_roll_joint"),
    ("right_arm_long_link_motor_joint", "right_wrist_pitch_joint"),
    ("right_arm_long_link_bevel_gear_joint", "right_wrist_pitch_joint"),
    ("left_arm_long_link_bevel_gear_joint", "left_wrist_pitch_joint"),
]


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    sim.set_camera_view(eye=(1.6, 1.6, 1.2), target=(0.0, 0.0, 0.6))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0)
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(SEMI_TAKS_LV1_CFG.replace(prim_path="/World/Robot"))
    sim.reset()

    joint_ids, amps, freqs, phases = [], [], [], []
    for expr, (amp, freq, phase) in MOTION_GROUPS.items():
        ids, names = robot.find_joints(expr)
        joint_ids += ids
        amps += [amp] * len(ids)
        freqs += [freq] * len(ids)
        phases += [phase] * len(ids)
        print(f"[motion] {expr}: {names}")
    joint_ids = torch.tensor(joint_ids, device=sim.device)
    amps = torch.tensor(amps, device=sim.device)
    freqs = torch.tensor(freqs, device=sim.device)
    phases = torch.tensor(phases, device=sim.device)

    name_to_idx = {n: i for i, n in enumerate(robot.joint_names)}
    base_pos = robot.data.default_joint_pos.clone()

    sim_dt = sim.get_physics_dt()
    t = 0.0
    step_count = 0
    while simulation_app.is_running():
        targets = base_pos.clone()
        targets[:, joint_ids] += amps * torch.sin(2.0 * math.pi * freqs * t + phases)
        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        t += sim_dt
        step_count += 1
        if step_count % 200 == 0:
            q = robot.data.joint_pos[0]
            report = " | ".join(
                f"{a.split('_joint')[0]}={float(q[name_to_idx[a]]):+.3f} -> {b.split('_joint')[0]}={float(q[name_to_idx[b]]):+.3f}"
                for a, b in REPORT_PAIRS
            )
            print(f"[t={t:5.1f}s] {report}")


if __name__ == "__main__":
    main()
    simulation_app.close()
