"""Custom event helpers for locomotion tasks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from pxr import Gf, Sdf, UsdGeom, UsdPhysics


def _iter_env_indices(env_ids, num_envs: int) -> Iterable[int]:
    if env_ids is None:
        return range(num_envs)
    if hasattr(env_ids, "detach"):
        env_ids = env_ids.detach().cpu().tolist()
    return [int(env_id) for env_id in env_ids]


def _vec3f(value) -> Gf.Vec3f:
    return Gf.Vec3f(float(value[0]), float(value[1]), float(value[2]))


def _quatf(value) -> Gf.Quatf:
    imag = value.GetImaginary()
    return Gf.Quatf(float(value.GetReal()), _vec3f(imag))


def _set_joint_frame_at_body1(stage, joint: UsdPhysics.Joint, body0_path: str, body1_path: str) -> None:
    """Place the joint frame at body1 while preserving the current body0 offset."""

    cache = UsdGeom.XformCache()
    body0_world = Gf.Transform(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body0_path)))
    body1_world = Gf.Transform(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body1_path)))

    local0 = Gf.Transform(body0_world.GetMatrix().GetInverse()) * body1_world
    local1 = Gf.Transform(body1_world.GetMatrix().GetInverse()) * body1_world

    joint.CreateLocalPos0Attr().Set(_vec3f(local0.GetTranslation()))
    joint.CreateLocalRot0Attr().Set(_quatf(local0.GetRotation().GetQuat()))
    joint.CreateLocalPos1Attr().Set(_vec3f(local1.GetTranslation()))
    joint.CreateLocalRot1Attr().Set(_quatf(local1.GetRotation().GetQuat()))


def _set_joint_frame_at_body_origins(joint: UsdPhysics.Joint) -> None:
    """Attach body origins directly, avoiding a startup-pose-dependent offset."""

    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def _set_joint_frame_at_local_offsets(
    stage,
    joint: UsdPhysics.Joint,
    body0_path: str,
    body1_path: str,
    body0_local_pos: Sequence[float],
    body1_local_pos: Sequence[float],
) -> None:
    """Attach two explicit local points instead of the rigid-body origins."""

    cache = UsdGeom.XformCache()
    body0_world = Gf.Transform(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body0_path)))
    body1_world = Gf.Transform(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body1_path)))
    body1_in_body0 = Gf.Transform(body0_world.GetMatrix().GetInverse()) * body1_world

    joint.CreateLocalPos0Attr().Set(_vec3f(body0_local_pos))
    joint.CreateLocalRot0Attr().Set(_quatf(body1_in_body0.GetRotation().GetQuat()))
    joint.CreateLocalPos1Attr().Set(_vec3f(body1_local_pos))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def _mask_collision_pair(stage, body0_path: str, body1_path: str) -> None:
    filtering_pairs = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(body0_path))
    rel = filtering_pairs.CreateFilteredPairsRel()
    target = Sdf.Path(body1_path)
    if target not in rel.GetTargets():
        rel.AddTarget(target)


def _env_ids_tensor(env, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if hasattr(env_ids, "to"):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _body_ids_list(asset_cfg: SceneEntityCfg) -> list[int]:
    if isinstance(asset_cfg.body_ids, slice):
        raise ValueError("Soft hand attachment requires explicit body_ids, not a slice.")
    if hasattr(asset_cfg.body_ids, "detach"):
        return asset_cfg.body_ids.detach().cpu().tolist()
    return list(asset_cfg.body_ids)


def _body_points_w(asset, env_ids: torch.Tensor, body_ids: list[int], local_positions: Sequence[Sequence[float]]):
    body_pos_w = torch.nan_to_num(asset.data.body_pos_w[env_ids][:, body_ids, :])
    body_quat_w = torch.nan_to_num(asset.data.body_quat_w[env_ids][:, body_ids, :])
    local_pos = torch.as_tensor(local_positions, device=asset.device, dtype=body_pos_w.dtype).unsqueeze(0)
    local_pos = local_pos.expand(body_pos_w.shape[0], -1, -1)
    offset_w = math_utils.quat_apply(body_quat_w.reshape(-1, 4), local_pos.reshape(-1, 3)).reshape_as(body_pos_w)
    return body_pos_w + offset_w, body_quat_w


def apply_soft_hand_handle_attachment(
    env,
    env_ids,
    robot_cfg: SceneEntityCfg,
    wheelchair_cfg: SceneEntityCfg,
    robot_body_local_positions: Sequence[Sequence[float]],
    wheelchair_body_local_positions: Sequence[Sequence[float]] | None = None,
    stiffness: float = 2500.0,
    damping: float = 75.0,
    max_force: float = 350.0,
) -> None:
    """Apply a bounded spring-damper between robot hand grip points and wheelchair handles."""

    robot = env.scene[robot_cfg.name]
    wheelchair = env.scene[wheelchair_cfg.name]
    env_ids = _env_ids_tensor(env, env_ids)
    robot_body_ids = _body_ids_list(robot_cfg)
    wheelchair_body_ids = _body_ids_list(wheelchair_cfg)

    if wheelchair_body_local_positions is None:
        wheelchair_body_local_positions = [(0.0, 0.0, 0.0) for _ in wheelchair_body_ids]

    hand_pos_w, hand_quat_w = _body_points_w(robot, env_ids, robot_body_ids, robot_body_local_positions)
    handle_pos_w, handle_quat_w = _body_points_w(
        wheelchair, env_ids, wheelchair_body_ids, wheelchair_body_local_positions
    )
    hand_vel_w = torch.nan_to_num(robot.data.body_lin_vel_w[env_ids][:, robot_body_ids, :])
    handle_vel_w = torch.nan_to_num(wheelchair.data.body_lin_vel_w[env_ids][:, wheelchair_body_ids, :])

    force_w = stiffness * (handle_pos_w - hand_pos_w) + damping * (handle_vel_w - hand_vel_w)
    force_norm = torch.linalg.norm(force_w, dim=-1, keepdim=True).clamp_min(1.0e-6)
    force_w = force_w * torch.clamp(max_force / force_norm, max=1.0)
    force_w = torch.nan_to_num(force_w)

    robot_force_b = math_utils.quat_apply_inverse(hand_quat_w.reshape(-1, 4), force_w.reshape(-1, 3)).reshape_as(
        force_w
    )
    wheelchair_force_b = math_utils.quat_apply_inverse(
        handle_quat_w.reshape(-1, 4), (-force_w).reshape(-1, 3)
    ).reshape_as(force_w)

    robot_positions_b = torch.as_tensor(
        robot_body_local_positions, device=robot.device, dtype=force_w.dtype
    ).unsqueeze(0)
    robot_positions_b = robot_positions_b.expand(force_w.shape[0], -1, -1)
    wheelchair_positions_b = torch.as_tensor(
        wheelchair_body_local_positions, device=wheelchair.device, dtype=force_w.dtype
    ).unsqueeze(0)
    wheelchair_positions_b = wheelchair_positions_b.expand(force_w.shape[0], -1, -1)
    zero_torque = torch.zeros_like(force_w)

    robot.permanent_wrench_composer.set_forces_and_torques(
        forces=robot_force_b,
        torques=zero_torque,
        positions=robot_positions_b,
        body_ids=robot_body_ids,
        env_ids=env_ids,
        is_global=False,
    )
    wheelchair.permanent_wrench_composer.set_forces_and_torques(
        forces=wheelchair_force_b,
        torques=zero_torque,
        positions=wheelchair_positions_b,
        body_ids=wheelchair_body_ids,
        env_ids=env_ids,
        is_global=False,
    )


def attach_wheelchair_hands_to_handles(
    env,
    env_ids,
    attachments: Sequence[dict[str, object]],
    robot_prim_name: str = "Robot",
    wheelchair_prim_name: str = "Wheelchair",
    joint_root_name: str = "HandHandleFixedJoints",
    joint_type: str = "spherical",
    mask_collisions: bool = True,
    anchor_at_body_origins: bool = False,
    skip_existing: bool = False,
) -> None:
    """Create USD joints from G1 hand bodies to wheelchair handle bodies."""

    stage = env.sim.get_initial_stage()
    missing_paths: list[str] = []

    for env_index in _iter_env_indices(env_ids, env.num_envs):
        env_path = env.scene.env_prim_paths[env_index]
        joint_root_path = f"{env_path}/{joint_root_name}"
        stage.DefinePrim(joint_root_path, "Xform")

        for attachment in attachments:
            body0_path = f"{env_path}/{robot_prim_name}/{attachment['robot_body']}"
            body1_path = f"{env_path}/{wheelchair_prim_name}/{attachment['wheelchair_body']}"
            joint_path = f"{joint_root_path}/{attachment['joint_name']}"
            if skip_existing and stage.GetPrimAtPath(joint_path).IsValid():
                continue

            body0_prim = stage.GetPrimAtPath(body0_path)
            body1_prim = stage.GetPrimAtPath(body1_path)
            if not body0_prim.IsValid() or not body1_prim.IsValid():
                if not body0_prim.IsValid():
                    missing_paths.append(body0_path)
                if not body1_prim.IsValid():
                    missing_paths.append(body1_path)
                continue

            if joint_type == "fixed":
                joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
            elif joint_type == "spherical":
                joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)
            else:
                raise ValueError(f"Unsupported hand attachment joint_type: {joint_type}")

            robot_local_pos = attachment.get("robot_local_pos")
            wheelchair_local_pos = attachment.get("wheelchair_local_pos")
            # Author the joint frame before binding bodies. Otherwise PhysX can see a transient
            # origin-to-origin joint and report disjoint bodies before the grip offsets exist.
            if robot_local_pos is not None or wheelchair_local_pos is not None:
                _set_joint_frame_at_local_offsets(
                    stage,
                    joint,
                    body0_path,
                    body1_path,
                    robot_local_pos or (0.0, 0.0, 0.0),
                    wheelchair_local_pos or (0.0, 0.0, 0.0),
                )
            elif anchor_at_body_origins:
                _set_joint_frame_at_body_origins(joint)
            else:
                _set_joint_frame_at_body1(stage, joint, body0_path, body1_path)

            joint.GetExcludeFromArticulationAttr().Set(True)
            joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])

            if mask_collisions:
                _mask_collision_pair(stage, body0_path, body1_path)

    if missing_paths:
        preview = ", ".join(missing_paths[:8])
        suffix = "" if len(missing_paths) <= 8 else f", ... ({len(missing_paths)} missing paths total)"
        raise RuntimeError(f"Could not create wheelchair hand attachments; missing prim paths: {preview}{suffix}")


def constrain_root_to_forward_rail(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
    lateral_position: float = 0.0,
    root_height: float | None = None,
    roll: float | None = None,
    pitch: float | None = None,
    yaw: float = 0.0,
    zero_lateral_velocity: bool = True,
    zero_vertical_velocity: bool = False,
    zero_roll_pitch_velocity: bool = False,
    zero_yaw_velocity: bool = True,
) -> None:
    """Keep an asset on a straight rail while preserving forward motion."""

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)
    elif not hasattr(env_ids, "device"):
        env_ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)

    root_pose = torch.cat([asset.data.root_pos_w[env_ids].clone(), asset.data.root_quat_w[env_ids].clone()], dim=-1)
    root_pose[:, 1] = env.scene.env_origins[env_ids, 1] + lateral_position
    if root_height is not None:
        root_pose[:, 2] = env.scene.env_origins[env_ids, 2] + root_height

    current_roll, current_pitch, _ = math_utils.euler_xyz_from_quat(root_pose[:, 3:7])
    target_roll = current_roll if roll is None else torch.full_like(current_roll, roll)
    target_pitch = current_pitch if pitch is None else torch.full_like(current_pitch, pitch)
    root_pose[:, 3:7] = math_utils.quat_from_euler_xyz(target_roll, target_pitch, torch.full_like(current_roll, yaw))

    root_velocity = asset.data.root_vel_w[env_ids].clone()
    if zero_lateral_velocity:
        root_velocity[:, 1] = 0.0
    if zero_vertical_velocity:
        root_velocity[:, 2] = 0.0
    if zero_roll_pitch_velocity:
        root_velocity[:, 3] = 0.0
        root_velocity[:, 4] = 0.0
    if zero_yaw_velocity:
        root_velocity[:, 5] = 0.0

    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
