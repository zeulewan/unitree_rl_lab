# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apply a scripted force to the passive wheelchair and report raw motion stats."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check passive wheelchair rolling behavior under a scripted force.")
parser.add_argument(
    "--task",
    default="Unitree-G1-29dof-Wheelchair-Relaxed-Push-Attached",
    help="Task that contains a scene object named 'wheelchair'.",
)
parser.add_argument("--num-envs", type=int, default=10, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=400, help="Number of control steps to sample.")
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

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
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


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
        entry_point_key="play_env_cfg_entry_point",
    )
    _disable_attachment_event(env_cfg)
    _move_robot_away(env_cfg)
    _disable_fall_terminations(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
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
        for _ in range(args_cli.steps):
            env.step(actions)
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


if __name__ == "__main__":
    main()
    simulation_app.close()
