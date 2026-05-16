# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video-start-step", type=int, default=0, help="Simulation step to start the recorded video.")
parser.add_argument("--video-follow-robot", action="store_true", default=False, help="Track the first robot in recorded videos.")
parser.add_argument("--video-follow-env-index", type=int, default=0, help="Environment index to track in recorded videos.")
parser.add_argument(
    "--video-follow-best-robot",
    action="store_true",
    default=False,
    help="Track the robot with the most XY displacement after --video-follow-best-after-steps.",
)
parser.add_argument(
    "--video-follow-best-after-steps",
    type=int,
    default=50,
    help="Number of steps before selecting the best moving robot for video follow.",
)
parser.add_argument(
    "--video-camera-eye-offset",
    type=float,
    nargs=3,
    default=(-6.0, -5.0, 2.8),
    metavar=("X", "Y", "Z"),
    help="Camera eye offset from the first robot when --video-follow-robot is set.",
)
parser.add_argument(
    "--video-camera-target-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.9),
    metavar=("X", "Y", "Z"),
    help="Camera target offset from the first robot when --video-follow-robot is set.",
)
parser.add_argument(
    "--video-camera-orbit-deg",
    type=float,
    default=0.0,
    help="Yaw degrees to rotate the follow camera eye offset over the recorded clip.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--zero-actions",
    action="store_true",
    default=False,
    help="Step the environment with zero actions instead of the loaded policy.",
)
parser.add_argument(
    "--ragdoll-robot",
    action="store_true",
    default=False,
    help="Set robot joint stiffness and damping to zero before playback.",
)
parser.add_argument(
    "--disable-fall-terminations",
    action="store_true",
    default=False,
    help="Disable base-height and bad-orientation termination terms for diagnostic playback.",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    get_published_pretrained_checkpoint = None
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _set_follow_camera(env, timestep: int = 0):
    """Point the render camera at a selected robot for useful recorded videos."""
    if not args_cli.video_follow_robot:
        return
    try:
        robot = env.unwrapped.scene["robot"]
        root_pos_w = robot.data.root_pos_w.detach()
        if not hasattr(_set_follow_camera, "_initial_root_xy"):
            _set_follow_camera._initial_root_xy = root_pos_w[:, :2].clone()
            _set_follow_camera._selected_env_index = max(
                0, min(args_cli.video_follow_env_index, root_pos_w.shape[0] - 1)
            )
        if (
            args_cli.video_follow_best_robot
            and not getattr(_set_follow_camera, "_selected_best_robot", False)
            and timestep >= args_cli.video_follow_best_after_steps
        ):
            displacement = torch.linalg.norm(root_pos_w[:, :2] - _set_follow_camera._initial_root_xy, dim=1)
            _set_follow_camera._selected_env_index = int(torch.argmax(displacement).item())
            _set_follow_camera._selected_best_robot = True
            print(
                "[INFO] Video follow selected env "
                f"{_set_follow_camera._selected_env_index} "
                f"(XY displacement {displacement[_set_follow_camera._selected_env_index].item():.3f} m)"
            )
        env_index = max(0, min(_set_follow_camera._selected_env_index, root_pos_w.shape[0] - 1))
        root_pos = root_pos_w[env_index].cpu()
        eye_offset = torch.tensor(args_cli.video_camera_eye_offset, dtype=root_pos.dtype)
        target_offset = torch.tensor(args_cli.video_camera_target_offset, dtype=root_pos.dtype)
        if args_cli.video_camera_orbit_deg != 0.0:
            progress = 0.0
            if args_cli.video_length > 0:
                progress = (timestep - args_cli.video_start_step) / float(args_cli.video_length)
                progress = max(0.0, min(1.0, progress))
            angle = math.radians(args_cli.video_camera_orbit_deg) * progress
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            offset_x = float(eye_offset[0])
            offset_y = float(eye_offset[1])
            eye_offset[0] = cos_angle * offset_x - sin_angle * offset_y
            eye_offset[1] = sin_angle * offset_x + cos_angle * offset_y
        env.unwrapped.sim.set_camera_view((root_pos + eye_offset).tolist(), (root_pos + target_offset).tolist())
    except Exception as err:
        if not getattr(_set_follow_camera, "_warned", False):
            print(f"[WARN] Failed to update video follow camera: {err}")
            _set_follow_camera._warned = True


def _disable_robot_actuators(env):
    """Turn the robot into a passive articulation for startup diagnostics."""
    try:
        robot = env.unwrapped.scene["robot"]
        robot.write_joint_stiffness_to_sim(torch.zeros_like(robot.data.joint_stiffness))
        robot.write_joint_damping_to_sim(torch.zeros_like(robot.data.joint_damping))
        for actuator in robot.actuators.values():
            if hasattr(actuator, "stiffness"):
                actuator.stiffness[:] = 0.0
            if hasattr(actuator, "damping"):
                actuator.damping[:] = 0.0
        print("[INFO] Ragdoll playback: robot joint stiffness and damping set to zero.")
    except Exception as err:
        print(f"[WARN] Failed to disable robot actuators for ragdoll playback: {err}")


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    if args_cli.disable_fall_terminations and hasattr(env_cfg, "terminations"):
        for term_name in ("base_height", "bad_orientation"):
            if hasattr(env_cfg.terminations, term_name):
                setattr(env_cfg.terminations, term_name, None)
        if hasattr(env_cfg, "rewards") and hasattr(env_cfg.rewards, "fall_termination"):
            env_cfg.rewards.fall_termination = None
        print("[INFO] Diagnostic playback: disabled base_height and bad_orientation terminations.")
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env = env

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        base_env = env

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == args_cli.video_start_step,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args_cli.ragdoll_robot:
        _disable_robot_actuators(base_env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

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
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    _set_follow_camera(base_env, timestep=0)
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if args_cli.zero_actions:
                actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
            else:
                actions = policy(obs)
            _set_follow_camera(base_env, timestep=timestep)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep >= args_cli.video_start_step + args_cli.video_length:
                break

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
