from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def non_finite_asset_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate envs whose selected articulation state has NaN or inf values."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos_w, quat_w, lin_vel_w, ang_vel_w = _root_or_selected_body_state(asset, asset_cfg)
    tensors = (pos_w, quat_w, lin_vel_w, ang_vel_w)

    finite = torch.ones(env.num_envs, dtype=torch.bool, device=asset.device)
    for tensor in tensors:
        finite &= torch.isfinite(tensor).all(dim=-1)
    return ~finite


def asset_state_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_linear_velocity: float | None = None,
    max_angular_velocity: float | None = None,
    max_position_norm: float | None = None,
) -> torch.Tensor:
    """Terminate envs whose selected asset body has non-finite or runaway finite state."""

    asset: Articulation = env.scene[asset_cfg.name]
    pos_w, quat_w, lin_vel_w, ang_vel_w = _root_or_selected_body_state(asset, asset_cfg)

    done = torch.zeros(env.num_envs, dtype=torch.bool, device=asset.device)
    for tensor in (pos_w, quat_w, lin_vel_w, ang_vel_w):
        done |= ~torch.isfinite(tensor).all(dim=-1)
    if max_linear_velocity is not None:
        done |= torch.linalg.norm(lin_vel_w, dim=-1) > max_linear_velocity
    if max_angular_velocity is not None:
        done |= torch.linalg.norm(ang_vel_w, dim=-1) > max_angular_velocity
    if max_position_norm is not None:
        done |= torch.linalg.norm(pos_w, dim=-1) > max_position_norm
    return done


def _root_or_selected_body_state(
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(asset_cfg.body_ids, slice):
        return (
            asset.data.root_pos_w,
            asset.data.root_quat_w,
            asset.data.root_lin_vel_w,
            asset.data.root_ang_vel_w,
        )

    body_id = asset_cfg.body_ids[0] if isinstance(asset_cfg.body_ids, list) else int(asset_cfg.body_ids)
    return (
        asset.data.body_pos_w[:, body_id, :],
        asset.data.body_quat_w[:, body_id, :],
        asset.data.body_lin_vel_w[:, body_id, :],
        asset.data.body_ang_vel_w[:, body_id, :],
    )
