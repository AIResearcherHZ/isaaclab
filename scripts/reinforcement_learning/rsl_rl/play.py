# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# 扭矩录制参数
parser.add_argument("--enable_torque_recording", action="store_true", default=False, help="Enable torque recording.")
parser.add_argument("--torque_recording_dir", type=str, default="logs/torque_data", help="Directory to save torque recordings.")
parser.add_argument("--torque_recording_env_id", type=int, default=0, help="Environment ID to record torque from.")
parser.add_argument("--torque_recording_duration", type=float, default=10.0, help="Duration of torque recording in seconds.")
# 键盘控制参数
parser.add_argument("--keyboard", action="store_true", default=False, help="Enable keyboard control for velocity commands.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # 键盘控制初始化
    keyboard_controller = None
    if args_cli.keyboard:
        from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg
        keyboard_cfg = Se2KeyboardCfg(
            v_x_sensitivity=1.0,
            v_y_sensitivity=0.5,
            omega_z_sensitivity=1.0,
            sim_device=env.unwrapped.device,
        )
        keyboard_controller = Se2Keyboard(keyboard_cfg)
        print("[INFO] Keyboard control enabled. Use arrow keys to control velocity.")
        print(keyboard_controller)

    # 扭矩录制初始化
    torque_data = []
    recording_start_time = None
    if args_cli.enable_torque_recording:
        os.makedirs(args_cli.torque_recording_dir, exist_ok=True)
        print(f"[INFO] Torque recording enabled. Duration: {args_cli.torque_recording_duration}s")
        print(f"[INFO] Recording env_id: {args_cli.torque_recording_env_id}")
        print(f"[INFO] Save directory: {args_cli.torque_recording_dir}")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    sim_time = 0.0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # 键盘控制：覆盖速度命令
            if keyboard_controller is not None:
                vel_cmd = keyboard_controller.advance()
                # 直接设置command term的command属性
                cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
                cmd_term.command[:] = vel_cmd.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

            # 扭矩录制
            if args_cli.enable_torque_recording:
                if recording_start_time is None:
                    recording_start_time = sim_time
                elapsed = sim_time - recording_start_time
                if elapsed <= args_cli.torque_recording_duration:
                    env_id = args_cli.torque_recording_env_id
                    robot = env.unwrapped.scene["robot"]
                    torques = robot.data.applied_torque[env_id].cpu().numpy()
                    joint_pos = robot.data.joint_pos[env_id].cpu().numpy()
                    joint_vel = robot.data.joint_vel[env_id].cpu().numpy()
                    torque_data.append({
                        "time": elapsed,
                        "torques": torques.copy(),
                        "joint_pos": joint_pos.copy(),
                        "joint_vel": joint_vel.copy(),
                    })
                elif len(torque_data) > 0:
                    # 保存数据
                    save_path = os.path.join(args_cli.torque_recording_dir, f"torque_recording_{task_name}.npz")
                    np.savez(
                        save_path,
                        times=np.array([d["time"] for d in torque_data]),
                        torques=np.array([d["torques"] for d in torque_data]),
                        joint_pos=np.array([d["joint_pos"] for d in torque_data]),
                        joint_vel=np.array([d["joint_vel"] for d in torque_data]),
                    )
                    print(f"[INFO] Torque data saved to: {save_path}")
                    torque_data = []  # 清空防止重复保存

        sim_time += dt

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # 如果录制未完成但程序退出，保存已有数据
    if args_cli.enable_torque_recording and len(torque_data) > 0:
        save_path = os.path.join(args_cli.torque_recording_dir, f"torque_recording_{task_name}.npz")
        np.savez(
            save_path,
            times=np.array([d["time"] for d in torque_data]),
            torques=np.array([d["torques"] for d in torque_data]),
            joint_pos=np.array([d["joint_pos"] for d in torque_data]),
            joint_vel=np.array([d["joint_vel"] for d in torque_data]),
        )
        print(f"[INFO] Torque data saved to: {save_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
