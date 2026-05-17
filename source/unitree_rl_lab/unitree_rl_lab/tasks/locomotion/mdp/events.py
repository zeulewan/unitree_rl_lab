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

            joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
            joint.GetExcludeFromArticulationAttr().Set(True)
            robot_local_pos = attachment.get("robot_local_pos")
            wheelchair_local_pos = attachment.get("wheelchair_local_pos")
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
    yaw: float = 0.0,
    zero_lateral_velocity: bool = True,
    zero_yaw_velocity: bool = True,
) -> None:
    """Keep an asset on a straight lateral/yaw rail while preserving forward motion."""

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)
    elif not hasattr(env_ids, "device"):
        env_ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)

    root_pose = torch.cat([asset.data.root_pos_w[env_ids].clone(), asset.data.root_quat_w[env_ids].clone()], dim=-1)
    root_pose[:, 1] = env.scene.env_origins[env_ids, 1] + lateral_position

    roll, pitch, _ = math_utils.euler_xyz_from_quat(root_pose[:, 3:7])
    root_pose[:, 3:7] = math_utils.quat_from_euler_xyz(roll, pitch, torch.full_like(roll, yaw))

    root_velocity = asset.data.root_vel_w[env_ids].clone()
    if zero_lateral_velocity:
        root_velocity[:, 1] = 0.0
    if zero_yaw_velocity:
        root_velocity[:, 5] = 0.0

    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
