# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Passive wheelchair asset configurations."""

from __future__ import annotations

import copy
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "assets" / "objects" / "wheelchair").exists():
            return parent
    raise RuntimeError("Could not locate unitree_rl_lab repository root.")


ACTIVE_MANUAL_WHEELCHAIR_ROOT = _repo_root() / "assets" / "objects" / "wheelchair" / "free3d_active_wheelchair"
ACTIVE_MANUAL_WHEELCHAIR_URDF = ACTIVE_MANUAL_WHEELCHAIR_ROOT / "urdf" / "active_manual_wheelchair.urdf"


ACTIVE_MANUAL_WHEELCHAIR_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(ACTIVE_MANUAL_WHEELCHAIR_URDF),
        fix_base=False,
        merge_fixed_joints=False,
        self_collision=False,
        replace_cylinders_with_capsules=False,
        collision_from_visuals=False,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={},
)

ACTIVE_MANUAL_WHEELCHAIR_FIXED_BASE_CFG = copy.deepcopy(ACTIVE_MANUAL_WHEELCHAIR_CFG)
ACTIVE_MANUAL_WHEELCHAIR_FIXED_BASE_CFG.spawn.fix_base = True
