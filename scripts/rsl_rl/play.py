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
    "--print-hand-handle-offsets",
    action="store_true",
    default=False,
    help="Print rubber-hand to wheelchair-handle offsets after reset for attached-wheelchair diagnostics.",
)
parser.add_argument(
    "--print-wheelchair-speed-stats",
    action="store_true",
    default=False,
    help="Print raw wheelchair velocity statistics during playback diagnostics.",
)
parser.add_argument(
    "--speed-stats-steps",
    type=int,
    default=300,
    help="Number of playback steps to sample when --print-wheelchair-speed-stats is set without video.",
)
parser.add_argument(
    "--exit-after-offset-print",
    action="store_true",
    default=False,
    help="Exit immediately after --print-hand-handle-offsets.",
)
parser.add_argument(
    "--show-wheelchair-urdf-proxy",
    action="store_true",
    default=False,
    help="Use the playback-only wheelchair URDF that renders the simplified collision/handle proxy.",
)
parser.add_argument(
    "--hide-wheelchair-visuals",
    action="store_true",
    default=False,
    help="Hide the wheelchair visual mesh during playback diagnostics.",
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
from pathlib import Path

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_apply, quat_apply_inverse

try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    get_published_pretrained_checkpoint = None
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

from pxr import UsdGeom
import omni.usd


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


def _print_hand_handle_offsets(env):
    """Print measured hand-to-handle offsets after reset for startup alignment debugging."""
    try:
        robot = env.unwrapped.scene["robot"]
        wheelchair = env.unwrapped.scene["wheelchair"]
        hand_ids = [robot.data.body_names.index("left_rubber_hand"), robot.data.body_names.index("right_rubber_hand")]
        handle_ids = [
            wheelchair.data.body_names.index("left_handle_frame"),
            wheelchair.data.body_names.index("right_handle_frame"),
        ]
        hand_pos_w = robot.data.body_pos_w[:, hand_ids, :]
        handle_pos_w = wheelchair.data.body_pos_w[:, handle_ids, :]
        grip_offsets_b = torch.tensor(
            [[0.05414, -0.00372, 0.00502], [0.05414, 0.00372, 0.00502]],
            device=hand_pos_w.device,
        )
        grip_offsets_b = grip_offsets_b.expand(hand_pos_w.shape[0], -1, -1)
        grip_pos_w = hand_pos_w + quat_apply(
            robot.data.body_quat_w[:, hand_ids, :].reshape(-1, 4),
            grip_offsets_b.reshape(-1, 3),
        ).reshape_as(hand_pos_w)
        offset_w = handle_pos_w - hand_pos_w
        grip_offset_w = handle_pos_w - grip_pos_w
        error = torch.linalg.norm(offset_w, dim=-1)
        grip_error = torch.linalg.norm(grip_offset_w, dim=-1)
        suggested_grip_offsets_b = quat_apply_inverse(
            robot.data.body_quat_w[:, hand_ids, :].reshape(-1, 4),
            offset_w.reshape(-1, 3),
        ).reshape_as(offset_w)
        env_index = 0
        print("[INFO] Hand-handle startup offsets for env 0:", flush=True)
        stage = omni.usd.get_context().get_stage()
        for side_index, side_name in enumerate(("left", "right")):
            joint_name = f"{side_name}_hand_to_handle_anchor_joint"
            joint_prim = stage.GetPrimAtPath(f"/World/envs/env_0/HandHandleFixedJoints/{joint_name}")
            joint_local_pos0 = None
            if joint_prim.IsValid():
                local_pos0_attr = joint_prim.GetAttribute("physics:localPos0")
                if local_pos0_attr.IsValid():
                    joint_local_pos0 = list(local_pos0_attr.Get())
            print(
                "[INFO] "
                f"{side_name}: handle_minus_hand_w={offset_w[env_index, side_index].detach().cpu().tolist()} "
                f"error_m={float(error[env_index, side_index].detach().cpu()):.6f} "
                f"handle_minus_grip_w={grip_offset_w[env_index, side_index].detach().cpu().tolist()} "
                f"grip_error_m={float(grip_error[env_index, side_index].detach().cpu()):.6f} "
                f"suggested_grip_local={suggested_grip_offsets_b[env_index, side_index].detach().cpu().tolist()} "
                f"joint_local_pos0={joint_local_pos0}",
                flush=True,
            )
        print(f"[INFO] robot_root_w={robot.data.root_pos_w[env_index].detach().cpu().tolist()}", flush=True)
        print(f"[INFO] wheelchair_root_w={wheelchair.data.root_pos_w[env_index].detach().cpu().tolist()}", flush=True)
    except Exception as err:
        print(f"[WARN] Failed to print hand-handle offsets: {err}", flush=True)


def _sample_wheelchair_speed_stats(env, samples):
    """Collect raw wheelchair speed samples for policy diagnostics."""
    try:
        wheelchair = env.unwrapped.scene["wheelchair"]
        command = env.unwrapped.command_manager.get_command("base_velocity")
        samples["forward"].append(wheelchair.data.root_lin_vel_w[:, 0].detach().cpu())
        samples["lateral"].append(wheelchair.data.root_lin_vel_w[:, 1].detach().cpu())
        samples["yaw"].append(wheelchair.data.root_ang_vel_w[:, 2].detach().cpu())
        centerline_y = wheelchair.data.root_pos_w[:, 1] - env.unwrapped.scene.env_origins[:, 1]
        samples["centerline_y"].append(centerline_y.detach().cpu())
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=env.unwrapped.device, dtype=wheelchair.data.root_quat_w.dtype)
        x_axis = x_axis.expand(env.unwrapped.num_envs, 3)
        heading_w = quat_apply(wheelchair.data.root_quat_w, x_axis)
        samples["heading_y"].append(heading_w[:, 1].detach().cpu())
        samples["command_x"].append(command[:, 0].detach().cpu())
    except Exception as err:
        if not samples.get("warned"):
            print(f"[WARN] Failed to sample wheelchair speed stats: {err}", flush=True)
            samples["warned"] = True


def _print_wheelchair_speed_stats(samples):
    """Print raw wheelchair speed stats after playback."""
    if not samples["forward"]:
        print("[WARN] No wheelchair speed samples collected.", flush=True)
        return

    forward = torch.cat(samples["forward"])
    lateral = torch.cat(samples["lateral"])
    yaw = torch.cat(samples["yaw"])
    centerline_y = torch.cat(samples["centerline_y"])
    heading_y = torch.cat(samples["heading_y"])
    command_x = torch.cat(samples["command_x"])
    env_count = samples["command_x"][-1].numel()
    error = forward - command_x
    abs_error = torch.abs(error)
    print("[INFO] Wheelchair raw speed stats:", flush=True)
    print(
        "[INFO] "
        f"samples={forward.numel()} "
        f"command_x_mean={command_x.mean().item():.4f}m/s "
        f"forward_mean={forward.mean().item():.4f}m/s "
        f"forward_min={forward.min().item():.4f}m/s "
        f"forward_max={forward.max().item():.4f}m/s",
        flush=True,
    )
    print(
        "[INFO] "
        f"forward_error_mean={error.mean().item():.4f}m/s "
        f"forward_abs_error_mean={abs_error.mean().item():.4f}m/s "
        f"within_0.05mps={(abs_error < 0.05).float().mean().item():.3f} "
        f"within_0.10mps={(abs_error < 0.10).float().mean().item():.3f}",
        flush=True,
    )
    print(
        "[INFO] "
        f"lateral_mean={lateral.mean().item():.4f}m/s "
        f"lateral_abs_mean={torch.abs(lateral).mean().item():.4f}m/s "
        f"yaw_mean={yaw.mean().item():.4f}rad/s "
        f"yaw_abs_mean={torch.abs(yaw).mean().item():.4f}rad/s",
        flush=True,
    )
    print(
        "[INFO] "
        f"centerline_y_mean={centerline_y.mean().item():.4f}m "
        f"centerline_y_final_mean={centerline_y[-env_count:].mean().item():.4f}m "
        f"heading_y_mean={heading_y.mean().item():.4f} "
        f"heading_y_abs_mean={torch.abs(heading_y).mean().item():.4f}",
        flush=True,
    )


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "assets" / "objects" / "wheelchair").exists():
            return parent
    raise RuntimeError("Could not locate unitree_rl_lab repository root.")


def _configure_wheelchair_proxy_visual_asset(env_cfg):
    """Swap the playback wheelchair to a URDF that renders the simplified proxy model."""
    proxy_urdf = (
        _repo_root()
        / "assets"
        / "objects"
        / "wheelchair"
        / "free3d_active_wheelchair"
        / "urdf"
        / "active_manual_wheelchair_proxy_visual.urdf"
    )
    wheelchair_cfg = getattr(getattr(env_cfg, "scene", None), "wheelchair", None)
    spawn_cfg = getattr(wheelchair_cfg, "spawn", None)
    if spawn_cfg is None or not hasattr(spawn_cfg, "asset_path"):
        print("[WARN] Could not swap wheelchair asset for URDF proxy visuals.", flush=True)
        return
    spawn_cfg.asset_path = str(proxy_urdf)
    print(f"[INFO] Using wheelchair URDF proxy visual asset: {proxy_urdf}", flush=True)


def _hide_wheelchair_visuals(stage):
    hidden = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "/Wheelchair/" in path and "/visuals" in path:
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
                hidden += 1
    print(f"[INFO] Hidden {hidden} wheelchair visual prims for URDF proxy playback.")


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
    if args_cli.show_wheelchair_urdf_proxy:
        _configure_wheelchair_proxy_visual_asset(env_cfg)
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
    print(
        "[INFO] Wheelchair visual diagnostics: "
        f"hide_visuals={args_cli.hide_wheelchair_visuals}, "
        f"show_urdf_proxy={args_cli.show_wheelchair_urdf_proxy}",
        flush=True,
    )
    if args_cli.hide_wheelchair_visuals and not args_cli.show_wheelchair_urdf_proxy:
        stage = omni.usd.get_context().get_stage()
        _hide_wheelchair_visuals(stage)
    if args_cli.print_hand_handle_offsets:
        _print_hand_handle_offsets(base_env)
        if args_cli.exit_after_offset_print:
            env.close()
            return
    _set_follow_camera(base_env, timestep=0)
    speed_samples = {
        "forward": [],
        "lateral": [],
        "yaw": [],
        "centerline_y": [],
        "heading_y": [],
        "command_x": [],
    }
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
            if args_cli.print_wheelchair_speed_stats:
                _sample_wheelchair_speed_stats(base_env, speed_samples)
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep >= args_cli.video_start_step + args_cli.video_length:
                break
        elif args_cli.print_wheelchair_speed_stats and timestep >= args_cli.speed_stats_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.print_wheelchair_speed_stats:
        _print_wheelchair_speed_stats(speed_samples)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
