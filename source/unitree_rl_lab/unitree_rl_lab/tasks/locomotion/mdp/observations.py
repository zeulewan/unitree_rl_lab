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


def _finite_tensor(value: torch.Tensor, replacement: float = 0.0) -> torch.Tensor:
    return torch.nan_to_num(value, nan=replacement, posinf=replacement, neginf=replacement)


def _finite_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = _finite_tensor(quat)
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    return torch.where(norm > 1.0e-6, quat / norm.clamp_min(1.0e-6), identity)


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
    if wheelchair_cfg.body_ids != slice(None):
        body_id = wheelchair_cfg.body_ids[0]
        wheelchair_pos_w = _finite_tensor(wheelchair.data.body_pos_w[:, body_id, :])
        wheelchair_quat_w = _finite_quat(wheelchair.data.body_quat_w[:, body_id, :])
        wheelchair_lin_vel_w = _finite_tensor(wheelchair.data.body_lin_vel_w[:, body_id, :])
        wheelchair_ang_vel_w = _finite_tensor(wheelchair.data.body_ang_vel_w[:, body_id, :])
    else:
        wheelchair_pos_w = _finite_tensor(wheelchair.data.root_pos_w)
        wheelchair_quat_w = _finite_quat(wheelchair.data.root_quat_w)
        wheelchair_lin_vel_w = _finite_tensor(wheelchair.data.root_lin_vel_w)
        wheelchair_ang_vel_w = _finite_tensor(wheelchair.data.root_ang_vel_w)

    robot_root_pos_w = _finite_tensor(robot.data.root_pos_w)
    robot_root_quat_w = _finite_quat(robot.data.root_quat_w)

    rel_pos_w = wheelchair_pos_w - robot_root_pos_w
    rel_pos_b = quat_apply_inverse(robot_root_quat_w, rel_pos_w)

    rel_lin_vel_w = wheelchair_lin_vel_w - _finite_tensor(robot.data.root_lin_vel_w)
    rel_lin_vel_b = quat_apply_inverse(robot_root_quat_w, rel_lin_vel_w)

    x_axis_b = torch.zeros_like(rel_pos_w)
    x_axis_b[:, 0] = 1.0
    wheelchair_forward_w = quat_apply(wheelchair_quat_w, x_axis_b)
    wheelchair_forward_b = quat_apply_inverse(robot_root_quat_w, wheelchair_forward_w)

    rel_ang_vel_w = wheelchair_ang_vel_w - _finite_tensor(robot.data.root_ang_vel_w)
    rel_ang_vel_b = quat_apply_inverse(robot_root_quat_w, rel_ang_vel_w)

    centerline_error = (wheelchair_pos_w[:, 1] - env.scene.env_origins[:, 1]).unsqueeze(-1)
    state = torch.cat(
        (
            rel_pos_b,
            rel_lin_vel_b[:, :2],
            wheelchair_forward_b[:, :2],
            rel_ang_vel_b[:, 2:3],
            centerline_error,
        ),
        dim=-1,
    )
    return _finite_tensor(state)


def wheelchair_handle_state_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheelchair_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
    robot_body_local_positions: list[list[float]] | None = None,
    wheelchair_body_local_positions: list[list[float]] | None = None,
) -> torch.Tensor:
    """Wheelchair handle positions and hand-to-handle errors in the robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    wheelchair: Articulation = env.scene[wheelchair_cfg.name]

    hand_pos_w = _body_points_w(robot, robot_cfg.body_ids, robot_body_local_positions)
    handle_pos_w = _body_points_w(wheelchair, wheelchair_cfg.body_ids, wheelchair_body_local_positions)

    handle_pos_b = _vectors_in_robot_root_frame(robot, handle_pos_w - robot.data.root_pos_w[:, None, :])
    handle_error_b = _vectors_in_robot_root_frame(robot, handle_pos_w - hand_pos_w)
    return torch.cat((handle_pos_b.flatten(start_dim=1), handle_error_b.flatten(start_dim=1)), dim=-1)


def _body_points_w(
    asset: Articulation,
    body_ids: list[int],
    local_positions: list[list[float]] | None = None,
) -> torch.Tensor:
    body_pos_w = _finite_tensor(asset.data.body_pos_w[:, body_ids, :])
    if local_positions is None:
        return body_pos_w

    local_pos = torch.tensor(local_positions, device=body_pos_w.device, dtype=body_pos_w.dtype).unsqueeze(0)
    local_pos = local_pos.expand(body_pos_w.shape[0], -1, -1)
    body_quat_w = _finite_quat(asset.data.body_quat_w[:, body_ids, :])
    offsets_w = quat_apply(body_quat_w.reshape(-1, 4), local_pos.reshape(-1, 3)).reshape_as(body_pos_w)
    return body_pos_w + offsets_w


def _vectors_in_robot_root_frame(robot: Articulation, vectors_w: torch.Tensor) -> torch.Tensor:
    vectors_flat = _finite_tensor(vectors_w).reshape(-1, 3)
    root_quat_w = _finite_quat(robot.data.root_quat_w)[:, None, :].expand(-1, vectors_w.shape[1], -1).reshape(-1, 4)
    return _finite_tensor(quat_apply_inverse(root_quat_w, vectors_flat).reshape(vectors_w.shape))
