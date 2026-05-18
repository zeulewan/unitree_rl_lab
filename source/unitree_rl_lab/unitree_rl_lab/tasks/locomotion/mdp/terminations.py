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
    if isinstance(asset_cfg.body_ids, slice):
        tensors = (
            asset.data.root_pos_w,
            asset.data.root_quat_w,
            asset.data.root_lin_vel_w,
            asset.data.root_ang_vel_w,
        )
    else:
        body_id = asset_cfg.body_ids[0] if isinstance(asset_cfg.body_ids, list) else int(asset_cfg.body_ids)
        tensors = (
            asset.data.body_pos_w[:, body_id, :],
            asset.data.body_quat_w[:, body_id, :],
            asset.data.body_lin_vel_w[:, body_id, :],
            asset.data.body_ang_vel_w[:, body_id, :],
        )

    finite = torch.ones(env.num_envs, dtype=torch.bool, device=asset.device)
    for tensor in tensors:
        finite &= torch.isfinite(tensor).all(dim=-1)
    return ~finite
