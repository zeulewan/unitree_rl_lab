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


def wheelchair_soft_attachment_state_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheelchair_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
    robot_body_local_positions: list[list[float]] | None = None,
    wheelchair_body_local_positions: list[list[float]] | None = None,
    stiffness: float = 2500.0,
    damping: float = 75.0,
    max_force: float = 350.0,
    force_scale: float = 350.0,
) -> torch.Tensor:
    """Soft hand-handle load state for compliant wheelchair attachment tasks."""

    robot: Articulation = env.scene[robot_cfg.name]
    wheelchair: Articulation = env.scene[wheelchair_cfg.name]

    if wheelchair_body_local_positions is None:
        wheelchair_body_local_positions = [[0.0, 0.0, 0.0] for _ in wheelchair_cfg.body_ids]

    hand_pos_w, hand_quat_w, hand_vel_w = _body_point_state_w(
        robot, robot_cfg.body_ids, robot_body_local_positions
    )
    handle_pos_w, handle_quat_w, handle_vel_w = _body_point_state_w(
        wheelchair, wheelchair_cfg.body_ids, wheelchair_body_local_positions
    )

    rel_vel_b = _vectors_in_robot_root_frame(robot, handle_vel_w - hand_vel_w)

    force_w = stiffness * (handle_pos_w - hand_pos_w) + damping * (handle_vel_w - hand_vel_w)
    force_norm = torch.linalg.norm(force_w, dim=-1, keepdim=True).clamp_min(1.0e-6)
    force_w = force_w * torch.clamp(max_force / force_norm, max=1.0)
    force_b = torch.clamp(_vectors_in_robot_root_frame(robot, force_w) / force_scale, min=-1.0, max=1.0)
    force_norm_scaled = torch.clamp(torch.linalg.norm(force_w, dim=-1) / force_scale, max=1.0)

    basis = torch.eye(3, device=env.device, dtype=hand_quat_w.dtype)
    basis = basis.reshape(1, 1, 3, 3).expand(hand_quat_w.shape[0], hand_quat_w.shape[1], 3, 3)
    handle_axes_w = quat_apply(
        handle_quat_w[:, :, None, :].expand(-1, -1, 3, -1).reshape(-1, 4),
        basis.reshape(-1, 3),
    ).reshape_as(basis)
    handle_axes_in_hand = quat_apply_inverse(
        hand_quat_w[:, :, None, :].expand(-1, -1, 3, -1).reshape(-1, 4),
        handle_axes_w.reshape(-1, 3),
    ).reshape_as(handle_axes_w)

    state = torch.cat(
        (
            rel_vel_b.flatten(start_dim=1),
            handle_axes_in_hand.flatten(start_dim=1),
            force_b.flatten(start_dim=1),
            force_norm_scaled,
        ),
        dim=-1,
    )
    return _finite_tensor(state)


def _body_points_w(
    asset: Articulation,
    body_ids: list[int],
    local_positions: list[list[float]] | None = None,
) -> torch.Tensor:
    return _body_point_state_w(asset, body_ids, local_positions)[0]


def _body_point_state_w(
    asset: Articulation,
    body_ids: list[int],
    local_positions: list[list[float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    body_pos_w = _finite_tensor(asset.data.body_pos_w[:, body_ids, :])
    body_quat_w = _finite_quat(asset.data.body_quat_w[:, body_ids, :])
    body_lin_vel_w = _finite_tensor(asset.data.body_lin_vel_w[:, body_ids, :])
    if local_positions is None:
        return body_pos_w, body_quat_w, body_lin_vel_w

    local_pos = torch.tensor(local_positions, device=body_pos_w.device, dtype=body_pos_w.dtype).unsqueeze(0)
    local_pos = local_pos.expand(body_pos_w.shape[0], -1, -1)
    offsets_w = quat_apply(body_quat_w.reshape(-1, 4), local_pos.reshape(-1, 3)).reshape_as(body_pos_w)
    body_ang_vel_w = _finite_tensor(asset.data.body_ang_vel_w[:, body_ids, :])
    point_vel_w = body_lin_vel_w + torch.cross(body_ang_vel_w, offsets_w, dim=-1)
    return body_pos_w + offsets_w, body_quat_w, point_vel_w


def _vectors_in_robot_root_frame(robot: Articulation, vectors_w: torch.Tensor) -> torch.Tensor:
    vectors_flat = _finite_tensor(vectors_w).reshape(-1, 3)
    root_quat_w = _finite_quat(robot.data.root_quat_w)[:, None, :].expand(-1, vectors_w.shape[1], -1).reshape(-1, 4)
    return _finite_tensor(quat_apply_inverse(root_quat_w, vectors_flat).reshape(vectors_w.shape))
