# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apply a scripted force to the passive wheelchair and report raw motion stats."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check passive wheelchair rolling behavior under a scripted force.")
parser.add_argument(
    "--task",
    default="Unitree-G1-29dof-Wheelchair-Relaxed-Push-Attached",
    help="Task that contains a scene object named 'wheelchair'.",
)
parser.add_argument("--num-envs", type=int, default=10, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=400, help="Number of control steps to sample.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the scripted force check.")
parser.add_argument(
    "--video-dir",
    type=Path,
    default=None,
    help="Directory for recorded video. Defaults to logs/demos/wheelchair-force-check_<timestamp>.",
)
parser.add_argument(
    "--camera-eye-offset",
    type=float,
    nargs=3,
    default=(-3.2, -2.6, 1.7),
    metavar=("X", "Y", "Z"),
    help="Camera eye offset from the wheelchair root when recording video.",
)
parser.add_argument(
    "--camera-target-offset",
    type=float,
    nargs=3,
    default=(0.25, 0.0, 0.55),
    metavar=("X", "Y", "Z"),
    help="Camera target offset from the wheelchair root when recording video.",
)
parser.add_argument(
    "--camera-orbit-deg",
    type=float,
    default=0.0,
    help="Yaw degrees to rotate the camera around the wheelchair over the recorded clip.",
)
parser.add_argument(
    "--show-wheelchair-urdf-proxy",
    action="store_true",
    default=False,
    help="Render the simplified wheelchair URDF proxy instead of the downloaded visual mesh.",
)
parser.add_argument(
    "--force",
    type=float,
    nargs=3,
    default=(10.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="World-frame force in Newtons applied to each matched wheelchair body.",
)
parser.add_argument("--force-body", default="base_link", help="Wheelchair body regex to apply force to.")
parser.add_argument(
    "--keep-attachments",
    action="store_true",
    default=False,
    help="Keep hand-handle attachment events enabled if the selected task has them.",
)
parser.add_argument(
    "--robot-x",
    type=float,
    default=-5.0,
    help="Reset robot this many meters in X so it does not contact the chair during the check.",
)
parser.add_argument(
    "--robot-y",
    type=float,
    default=0.0,
    help="Reset robot this many meters in Y so it does not contact the chair during the check.",
)
parser.add_argument(
    "--disable-fall-terminations",
    action="store_true",
    default=True,
    help="Disable robot fall terminations so a zero-action robot does not reset the chair.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import math
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _disable_attachment_event(env_cfg) -> None:
    if args_cli.keep_attachments or not hasattr(env_cfg, "events"):
        return
    if hasattr(env_cfg.events, "attach_wheelchair_hands"):
        env_cfg.events.attach_wheelchair_hands = None
        print("[INFO] Disabled hand-handle attachment event for chair-only force check.", flush=True)


def _move_robot_away(env_cfg) -> None:
    if not hasattr(env_cfg, "events") or not hasattr(env_cfg.events, "reset_base"):
        return
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (args_cli.robot_x, args_cli.robot_x),
        "y": (args_cli.robot_y, args_cli.robot_y),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    print(f"[INFO] Robot reset offset set to x={args_cli.robot_x:.3f}, y={args_cli.robot_y:.3f}.", flush=True)


def _disable_fall_terminations(env_cfg) -> None:
    if not args_cli.disable_fall_terminations or not hasattr(env_cfg, "terminations"):
        return
    for term_name in ("base_height", "bad_orientation"):
        if hasattr(env_cfg.terminations, term_name):
            setattr(env_cfg.terminations, term_name, None)
    print("[INFO] Disabled robot fall terminations for the force check.", flush=True)


def _print_stats(name: str, tensor: torch.Tensor, unit: str = "") -> None:
    print(
        "[INFO] "
        f"{name}: mean={tensor.mean().item():.4f}{unit} "
        f"min={tensor.min().item():.4f}{unit} "
        f"max={tensor.max().item():.4f}{unit}",
        flush=True,
    )


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "logs" / "demos").exists():
            return parent
    return Path.cwd()


def _video_dir() -> Path:
    if args_cli.video_dir is not None:
        return args_cli.video_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _repo_root() / "logs" / "demos" / f"wheelchair-force-check_{timestamp}"


def _configure_wheelchair_proxy_visual_asset(env_cfg) -> None:
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


def _set_follow_camera(base_env, wheelchair, step: int) -> None:
    if not args_cli.video:
        return
    root_pos = wheelchair.data.root_pos_w[0].detach().cpu()
    eye_offset = torch.tensor(args_cli.camera_eye_offset, dtype=root_pos.dtype)
    target_offset = torch.tensor(args_cli.camera_target_offset, dtype=root_pos.dtype)
    if args_cli.camera_orbit_deg:
        progress = min(1.0, max(0.0, step / max(1, args_cli.steps)))
        angle = math.radians(args_cli.camera_orbit_deg) * progress
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        offset_x = float(eye_offset[0])
        offset_y = float(eye_offset[1])
        eye_offset[0] = cos_angle * offset_x - sin_angle * offset_y
        eye_offset[1] = sin_angle * offset_x + cos_angle * offset_y
    base_env.sim.set_camera_view((root_pos + eye_offset).tolist(), (root_pos + target_offset).tolist())


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
        entry_point_key="play_env_cfg_entry_point",
    )
    if args_cli.show_wheelchair_urdf_proxy:
        _configure_wheelchair_proxy_visual_asset(env_cfg)
    _disable_attachment_event(env_cfg)
    _move_robot_away(env_cfg)
    _disable_fall_terminations(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    video_dir = None
    if args_cli.video:
        video_dir = _video_dir()
        video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.steps,
            name_prefix="wheelchair-force-check",
            disable_logger=True,
        )
        print(f"[INFO] Recording scripted force video to: {video_dir}", flush=True)
    base_env = env.unwrapped
    env.reset()

    wheelchair = base_env.scene["wheelchair"]
    body_ids, body_names = wheelchair.find_bodies(args_cli.force_body)
    if not body_ids:
        env.close()
        raise RuntimeError(f"No wheelchair bodies matched regex: {args_cli.force_body}")

    force = torch.tensor(args_cli.force, device=base_env.device, dtype=torch.float32)
    forces = force.reshape(1, 1, 3).repeat(base_env.num_envs, len(body_ids), 1)
    torques = torch.zeros_like(forces)
    actions = torch.zeros((base_env.num_envs, base_env.action_manager.total_action_dim), device=base_env.device)

    initial_pos = wheelchair.data.root_pos_w.detach().clone()
    forward_samples = []
    lateral_samples = []
    yaw_samples = []
    heading_y_samples = []

    print(
        "[INFO] "
        f"Applying force={list(args_cli.force)} N to bodies={body_names} "
        f"for {args_cli.steps} steps across {base_env.num_envs} envs.",
        flush=True,
    )

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=base_env.device, dtype=torch.float32).repeat(base_env.num_envs, 1)
    with torch.inference_mode():
        wheelchair.set_external_force_and_torque(forces=forces, torques=torques, body_ids=body_ids, is_global=True)
        _set_follow_camera(base_env, wheelchair, 0)
        for step in range(args_cli.steps):
            env.step(actions)
            _set_follow_camera(base_env, wheelchair, step + 1)
            forward_samples.append(wheelchair.data.root_lin_vel_w[:, 0].detach().cpu())
            lateral_samples.append(wheelchair.data.root_lin_vel_w[:, 1].detach().cpu())
            yaw_samples.append(wheelchair.data.root_ang_vel_w[:, 2].detach().cpu())
            heading_w = quat_apply(wheelchair.data.root_quat_w, x_axis)
            heading_y_samples.append(heading_w[:, 1].detach().cpu())

    final_pos = wheelchair.data.root_pos_w.detach().clone()
    displacement = (final_pos - initial_pos).detach().cpu()
    forward = torch.cat(forward_samples)
    lateral = torch.cat(lateral_samples)
    yaw = torch.cat(yaw_samples)
    heading_y = torch.cat(heading_y_samples)

    print("[INFO] Wheelchair scripted force stats:", flush=True)
    print(f"[INFO] samples={forward.numel()} total_force_bodies={len(body_ids)}", flush=True)
    _print_stats("forward_velocity", forward, "m/s")
    _print_stats("lateral_velocity", lateral, "m/s")
    _print_stats("yaw_velocity", yaw, "rad/s")
    print(
        "[INFO] "
        f"lateral_abs_mean={torch.abs(lateral).mean().item():.4f}m/s "
        f"yaw_abs_mean={torch.abs(yaw).mean().item():.4f}rad/s "
        f"heading_y_abs_mean={torch.abs(heading_y).mean().item():.4f}",
        flush=True,
    )
    print(
        "[INFO] "
        f"final_dx_mean={displacement[:, 0].mean().item():.4f}m "
        f"final_dy_abs_mean={torch.abs(displacement[:, 1]).mean().item():.4f}m "
        f"final_dz_mean={displacement[:, 2].mean().item():.4f}m",
        flush=True,
    )

    env.close()
    if video_dir is not None:
        videos = sorted(video_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
        if videos:
            print(f"[INFO] Video: {videos[-1]}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
