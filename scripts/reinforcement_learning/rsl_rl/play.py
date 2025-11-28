# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument("--newton_visualizer", action="store_true", default=False, help="Enable Newton rendering.")
# torque recording arguments
parser.add_argument(
    "--enable_torque_recording", action="store_true", default=False, help="Enable torque recording during inference."
)
parser.add_argument(
    "--torque_recording_dir", type=str, default="torque_logs", help="Directory to save torque recordings."
)
parser.add_argument(
    "--torque_recording_env_id", type=int, default=0, help="Environment ID to record torque data from."
)
parser.add_argument(
    "--torque_recording_duration", type=float, default=5.0, help="Recording duration in seconds (default: 5.0)."
)
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

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

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

    # Enable Newton rendering if requested
    if args_cli.newton_visualizer:
        env_cfg.sim.enable_newton_rendering = True
        # Set play mode for Newton viewer if using Newton visualizer
        if hasattr(env_cfg.sim, "newton_cfg") and env_cfg.sim.newton_cfg is not None:
            env_cfg.sim.newton_cfg.visualizer_train_mode = False
        else:
            # Create newton_cfg if it doesn't exist
            try:
                from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
                newton_cfg = NewtonCfg()
                newton_cfg.visualizer_train_mode = False
                env_cfg.sim.newton_cfg = newton_cfg
            except ImportError:
                pass  # Newton not available

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

    # configure keyboard control if enabled
    controller = None
    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        if hasattr(env_cfg, 'terminations'):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg, 'commands') and hasattr(env_cfg.commands, 'base_velocity'):
            env_cfg.commands.base_velocity.debug_vis = True
            config = Se2KeyboardCfg(
                v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
                v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
                omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
            )
            controller = Se2Keyboard(config)
            # override velocity commands observation with keyboard input
            env_cfg.observations.policy.velocity_commands = ObsTerm(
                func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
            )

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
    # use command line arguments to override config if provided
    enable_torque_recording = args_cli.enable_torque_recording or getattr(agent_cfg, "enable_torque_recording", False)
    torque_recording_dir = args_cli.torque_recording_dir if args_cli.enable_torque_recording else getattr(
        agent_cfg, "torque_recording_dir", "torque_logs"
    )
    torque_recording_env_id = (
        args_cli.torque_recording_env_id if args_cli.enable_torque_recording else getattr(agent_cfg, "torque_recording_env_id", 0)
    )
    torque_recording_duration = (
        args_cli.torque_recording_duration if args_cli.enable_torque_recording else getattr(agent_cfg, "torque_recording_duration", 5.0)
    )

    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
        enable_torque_recording=enable_torque_recording,
        torque_recording_dir=torque_recording_dir,
        torque_recording_env_id=torque_recording_env_id,
        torque_recording_duration=torque_recording_duration,
    )

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

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            # camera follow for keyboard control
            if hasattr(env.unwrapped, 'scene') and hasattr(env.unwrapped.scene, 'robot'):
                robot = env.unwrapped.scene.robot
                if hasattr(robot, 'data') and hasattr(robot.data, 'root_pos_w'):
                    root_pos = robot.data.root_pos_w[0].cpu().numpy()
                    if hasattr(env.unwrapped.sim, 'set_camera_view'):
                        env.unwrapped.sim.set_camera_view(root_pos + [3.0, 3.0, 2.0], root_pos)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
