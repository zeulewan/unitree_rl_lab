from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate as quat_apply
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def wheelchair_root_state_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheelchair_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
) -> torch.Tensor:
    """Wheelchair root pose/velocity cues in the robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    wheelchair: Articulation = env.scene[wheelchair_cfg.name]

    rel_pos_w = wheelchair.data.root_pos_w - robot.data.root_pos_w
    rel_pos_b = quat_apply_inverse(robot.data.root_quat_w, rel_pos_w)

    rel_lin_vel_w = wheelchair.data.root_lin_vel_w - robot.data.root_lin_vel_w
    rel_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, rel_lin_vel_w)

    x_axis_b = torch.zeros_like(rel_pos_w)
    x_axis_b[:, 0] = 1.0
    wheelchair_forward_w = quat_apply(wheelchair.data.root_quat_w, x_axis_b)
    wheelchair_forward_b = quat_apply_inverse(robot.data.root_quat_w, wheelchair_forward_w)

    rel_ang_vel_w = wheelchair.data.root_ang_vel_w - robot.data.root_ang_vel_w
    rel_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, rel_ang_vel_w)

    centerline_error = (wheelchair.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]).unsqueeze(-1)
    return torch.cat(
        (
            rel_pos_b,
            rel_lin_vel_b[:, :2],
            wheelchair_forward_b[:, :2],
            rel_ang_vel_b[:, 2:3],
            centerline_error,
        ),
        dim=-1,
    )


def wheelchair_handle_state_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheelchair_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
) -> torch.Tensor:
    """Wheelchair handle positions and wrist-to-handle errors in the robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    wheelchair: Articulation = env.scene[wheelchair_cfg.name]

    wrist_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]
    handle_pos_w = wheelchair.data.body_pos_w[:, wheelchair_cfg.body_ids, :]

    handle_pos_b = _vectors_in_robot_root_frame(robot, handle_pos_w - robot.data.root_pos_w[:, None, :])
    handle_error_b = _vectors_in_robot_root_frame(robot, handle_pos_w - wrist_pos_w)
    return torch.cat((handle_pos_b.flatten(start_dim=1), handle_error_b.flatten(start_dim=1)), dim=-1)


def _vectors_in_robot_root_frame(robot: Articulation, vectors_w: torch.Tensor) -> torch.Tensor:
    vectors_flat = vectors_w.reshape(-1, 3)
    root_quat_w = robot.data.root_quat_w[:, None, :].expand(-1, vectors_w.shape[1], -1).reshape(-1, 4)
    return quat_apply_inverse(root_quat_w, vectors_flat).reshape(vectors_w.shape)
