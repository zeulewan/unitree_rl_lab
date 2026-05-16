"""Custom event helpers for locomotion tasks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

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

    local0 = body1_world * Gf.Transform(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body0_path)).GetInverse())
    local1 = body1_world * Gf.Transform(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body1_path)).GetInverse())

    joint.CreateLocalPos0Attr().Set(_vec3f(local0.GetTranslation()))
    joint.CreateLocalRot0Attr().Set(_quatf(local0.GetRotation().GetQuat()))
    joint.CreateLocalPos1Attr().Set(_vec3f(local1.GetTranslation()))
    joint.CreateLocalRot1Attr().Set(_quatf(local1.GetRotation().GetQuat()))


def _mask_collision_pair(stage, body0_path: str, body1_path: str) -> None:
    filtering_pairs = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(body0_path))
    rel = filtering_pairs.CreateFilteredPairsRel()
    target = Sdf.Path(body1_path)
    if target not in rel.GetTargets():
        rel.AddTarget(target)


def attach_wheelchair_hands_to_handles(
    env,
    env_ids,
    attachments: Sequence[dict[str, str]],
    robot_prim_name: str = "Robot",
    wheelchair_prim_name: str = "Wheelchair",
    joint_root_name: str = "HandHandleFixedJoints",
    joint_type: str = "spherical",
    mask_collisions: bool = True,
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
            _set_joint_frame_at_body1(stage, joint, body0_path, body1_path)

            if mask_collisions:
                _mask_collision_pair(stage, body0_path, body1_path)

    if missing_paths:
        preview = ", ".join(missing_paths[:8])
        suffix = "" if len(missing_paths) <= 8 else f", ... ({len(missing_paths)} missing paths total)"
        raise RuntimeError(f"Could not create wheelchair hand attachments; missing prim paths: {preview}{suffix}")
