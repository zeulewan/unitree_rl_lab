# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a simple G1 walking-policy demo with a procedural wheelchair prop.

This is a demo/visualization script, not a trained contact-manipulation task. The
wheelchair is kinematically kept at the robot's handle offset so an existing
walking policy can be shown "pushing" it forward for quick review videos.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play a G1 walking policy with a wheelchair push demo prop.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video in steps.")
parser.add_argument("--video-start-step", type=int, default=50, help="Simulation step to start the recorded video.")
parser.add_argument(
    "--video-folder",
    type=str,
    default=None,
    help="Output folder for recorded videos. Defaults to <checkpoint-run>/videos/wheelchair_push.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity", help="Name of the task.")
parser.add_argument("--command-x", type=float, default=0.55, help="Forward velocity command in m/s.")
parser.add_argument("--command-y", type=float, default=0.0, help="Lateral velocity command in m/s.")
parser.add_argument("--command-yaw", type=float, default=0.0, help="Yaw-rate command in rad/s.")
parser.add_argument(
    "--wheelchair-visual",
    choices=("procedural", "free3d"),
    default="procedural",
    help="Use the simple USD primitive chair or the locally downloaded Free3D wheelchair visual mesh.",
)
parser.add_argument(
    "--wheelchair-forward-offset",
    type=float,
    default=None,
    help="Wheelchair root offset ahead of the robot. Defaults depend on the selected visual.",
)
parser.add_argument(
    "--hide-hand-connectors",
    action="store_true",
    default=False,
    help="Hide the visual rods between the robot wrist links and wheelchair handles.",
)
parser.add_argument(
    "--hand-connector-radius",
    type=float,
    default=0.025,
    help="Radius of the visual hand-to-handle connector rods, in meters.",
)
parser.add_argument(
    "--camera-mode",
    choices=("fixed", "follow"),
    default="fixed",
    help="Use a fixed path-view camera or a chase camera.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import time
from pathlib import Path

import gymnasium as gym
import torch
from pxr import Gf, UsdGeom
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import omni.usd
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


class WheelchairProp:
    """Small USD-primitive wheelchair that follows a robot-relative handle offset."""

    default_forward_offset = 1.05

    def __init__(self, prim_path: str = "/World/WheelchairPushDemo"):
        self.stage = omni.usd.get_context().get_stage()
        self.root_path = prim_path
        self.left_handle_local = torch.tensor([-0.72, 0.24, 0.98])
        self.right_handle_local = torch.tensor([-0.72, -0.24, 0.98])
        self.root = UsdGeom.Xform.Define(self.stage, prim_path)
        root_xform = UsdGeom.Xformable(self.root.GetPrim())
        root_xform.ClearXformOpOrder()
        self.translate_op = root_xform.AddTranslateOp()
        self.rotate_op = root_xform.AddRotateZOp()

        self._add_cube("seat", (0.08, 0.08, 0.08), (0.0, 0.0, 0.55), (0.60, 0.50, 0.08))
        self._add_cube("backrest", (0.06, 0.06, 0.06), (-0.36, 0.0, 0.88), (0.08, 0.56, 0.55))
        self._add_cube("left_handle", (0.02, 0.02, 0.02), (-0.72, 0.24, 0.98), (0.38, 0.035, 0.035))
        self._add_cube("right_handle", (0.02, 0.02, 0.02), (-0.72, -0.24, 0.98), (0.38, 0.035, 0.035))
        self._add_cube("push_bar", (0.02, 0.02, 0.02), (-0.55, 0.0, 1.03), (0.035, 0.64, 0.035))
        self._add_cube("left_low_grip", (0.02, 0.02, 0.02), (-0.88, 0.22, 0.62), (0.45, 0.035, 0.035))
        self._add_cube("right_low_grip", (0.02, 0.02, 0.02), (-0.88, -0.22, 0.62), (0.45, 0.035, 0.035))
        self._add_cube("front_frame", (0.02, 0.02, 0.02), (0.25, 0.0, 0.42), (0.58, 0.045, 0.045))
        self._add_cube("left_side_frame", (0.02, 0.02, 0.02), (-0.02, 0.36, 0.42), (0.72, 0.035, 0.035))
        self._add_cube("right_side_frame", (0.02, 0.02, 0.02), (-0.02, -0.36, 0.42), (0.72, 0.035, 0.035))
        self._add_cube("left_footrest", (0.05, 0.05, 0.05), (0.58, 0.18, 0.18), (0.22, 0.12, 0.035))
        self._add_cube("right_footrest", (0.05, 0.05, 0.05), (0.58, -0.18, 0.18), (0.22, 0.12, 0.035))

        self._add_cylinder("left_wheel", (0.01, 0.01, 0.01), (-0.03, 0.43, 0.40), radius=0.36, height=0.065)
        self._add_cylinder("right_wheel", (0.01, 0.01, 0.01), (-0.03, -0.43, 0.40), radius=0.36, height=0.065)
        self._add_cylinder("left_rim", (0.75, 0.75, 0.75), (-0.03, 0.47, 0.40), radius=0.25, height=0.025)
        self._add_cylinder("right_rim", (0.75, 0.75, 0.75), (-0.03, -0.47, 0.40), radius=0.25, height=0.025)
        self._add_cylinder("left_caster", (0.01, 0.01, 0.01), (0.48, 0.32, 0.16), radius=0.13, height=0.045)
        self._add_cylinder("right_caster", (0.01, 0.01, 0.01), (0.48, -0.32, 0.16), radius=0.13, height=0.045)

    def _set_color(self, prim, color):
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])

    def _set_local_transform(self, prim, pos, scale):
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xform.AddScaleOp().Set(Gf.Vec3d(*scale))

    def _add_cube(self, name, color, pos, scale):
        cube = UsdGeom.Cube.Define(self.stage, f"{self.root_path}/{name}")
        cube.CreateSizeAttr(1.0)
        self._set_color(cube.GetPrim(), color)
        self._set_local_transform(cube.GetPrim(), pos, scale)

    def _add_cylinder(self, name, color, pos, radius, height):
        cylinder = UsdGeom.Cylinder.Define(self.stage, f"{self.root_path}/{name}")
        cylinder.CreateAxisAttr("Y")
        cylinder.CreateRadiusAttr(radius)
        cylinder.CreateHeightAttr(height)
        self._set_color(cylinder.GetPrim(), color)
        self._set_local_transform(cylinder.GetPrim(), pos, (1.0, 1.0, 1.0))

    def update_from_robot(self, robot, forward_offset: float):
        root_pos = robot.data.root_pos_w[0].detach().cpu()
        quat = robot.data.root_quat_w[0].detach().cpu()
        yaw = _quat_wxyz_to_yaw(quat)
        forward = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0])
        chair_pos = root_pos + forward * forward_offset
        chair_pos[2] = 0.0
        self.translate_op.Set(Gf.Vec3d(float(chair_pos[0]), float(chair_pos[1]), float(chair_pos[2])))
        self.rotate_op.Set(math.degrees(yaw))
        return chair_pos, yaw

    def handle_positions_world(self, chair_pos: torch.Tensor, yaw: float) -> tuple[torch.Tensor, torch.Tensor]:
        left_handle = self._local_point_to_world(self.left_handle_local, chair_pos, yaw)
        right_handle = self._local_point_to_world(self.right_handle_local, chair_pos, yaw)
        return left_handle, right_handle

    def _local_point_to_world(self, local_point: torch.Tensor, chair_pos: torch.Tensor, yaw: float) -> torch.Tensor:
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return torch.tensor(
            [
                float(chair_pos[0]) + cos_yaw * float(local_point[0]) - sin_yaw * float(local_point[1]),
                float(chair_pos[1]) + sin_yaw * float(local_point[0]) + cos_yaw * float(local_point[1]),
                float(chair_pos[2]) + float(local_point[2]),
            ]
        )


class Free3DWheelchairProp:
    """Downloaded Free3D active wheelchair mesh, kept at the same demo handle offset."""

    default_forward_offset = 0.75

    def __init__(self, prim_path: str = "/World/Free3DWheelchairPushDemo"):
        self.stage = omni.usd.get_context().get_stage()
        self.root_path = prim_path
        self.left_handle_local = torch.tensor([-0.40, 0.24, 0.88])
        self.right_handle_local = torch.tensor([-0.40, -0.24, 0.88])

        self.root = UsdGeom.Xform.Define(self.stage, prim_path)
        root_xform = UsdGeom.Xformable(self.root.GetPrim())
        root_xform.ClearXformOpOrder()
        self.translate_op = root_xform.AddTranslateOp()
        self.rotate_op = root_xform.AddRotateZOp()

        obj_path = (
            _repo_root()
            / "assets"
            / "objects"
            / "wheelchair"
            / "free3d_active_wheelchair"
            / "visual"
            / "active_wheelchair.obj"
        )
        if not obj_path.exists():
            raise FileNotFoundError(
                f"Missing Free3D wheelchair OBJ at {obj_path}. "
                "Run scripts/assets/import_free3d_active_wheelchair.py first."
            )
        self._add_obj_mesh(obj_path)

    def _add_obj_mesh(self, obj_path: Path):
        vertices, faces_by_material = _load_obj_mesh_groups(obj_path)
        material_colors = _load_mtl_colors(obj_path.with_suffix(".mtl"))

        for index, (material_name, faces) in enumerate(faces_by_material.items()):
            if not faces:
                continue
            mesh_path = f"{self.root_path}/mesh_{index:03d}_{_sanitize_prim_name(material_name)}"
            mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)
            points, face_counts, face_indices = _remap_faces(vertices, faces)
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr(face_counts)
            mesh.CreateFaceVertexIndicesAttr(face_indices)
            mesh.CreateSubdivisionSchemeAttr("none")
            color = material_colors.get(material_name, (0.55, 0.55, 0.55))
            UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])

    def update_from_robot(self, robot, forward_offset: float):
        root_pos = robot.data.root_pos_w[0].detach().cpu()
        quat = robot.data.root_quat_w[0].detach().cpu()
        yaw = _quat_wxyz_to_yaw(quat)
        forward = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0])
        chair_pos = root_pos + forward * forward_offset
        chair_pos[2] = 0.0
        self.translate_op.Set(Gf.Vec3d(float(chair_pos[0]), float(chair_pos[1]), float(chair_pos[2])))
        self.rotate_op.Set(math.degrees(yaw))
        return chair_pos, yaw

    def handle_positions_world(self, chair_pos: torch.Tensor, yaw: float) -> tuple[torch.Tensor, torch.Tensor]:
        left_handle = self._local_point_to_world(self.left_handle_local, chair_pos, yaw)
        right_handle = self._local_point_to_world(self.right_handle_local, chair_pos, yaw)
        return left_handle, right_handle

    def _local_point_to_world(self, local_point: torch.Tensor, chair_pos: torch.Tensor, yaw: float) -> torch.Tensor:
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return torch.tensor(
            [
                float(chair_pos[0]) + cos_yaw * float(local_point[0]) - sin_yaw * float(local_point[1]),
                float(chair_pos[1]) + sin_yaw * float(local_point[0]) + cos_yaw * float(local_point[1]),
                float(chair_pos[2]) + float(local_point[2]),
            ]
        )


class HandHandleConnectors:
    """Visible rods that mark the current wrist-to-handle attachment targets."""

    def __init__(self, prim_path: str = "/World/WheelchairPushDemoHandHandleConnectors", radius: float = 0.025):
        self.stage = omni.usd.get_context().get_stage()
        self.root_path = prim_path
        UsdGeom.Xform.Define(self.stage, prim_path)
        self.left = self._add_connector("left", (0.05, 0.35, 1.0), radius)
        self.right = self._add_connector("right", (1.0, 0.18, 0.08), radius)
        self.left_wrist_marker = self._add_marker("left_wrist_marker", (0.05, 0.35, 1.0), radius * 1.9)
        self.left_handle_marker = self._add_marker("left_handle_marker", (0.05, 0.35, 1.0), radius * 1.9)
        self.right_wrist_marker = self._add_marker("right_wrist_marker", (1.0, 0.18, 0.08), radius * 1.9)
        self.right_handle_marker = self._add_marker("right_handle_marker", (1.0, 0.18, 0.08), radius * 1.9)

    def _add_connector(self, name, color, radius):
        cylinder = UsdGeom.Cylinder.Define(self.stage, f"{self.root_path}/{name}")
        cylinder.CreateAxisAttr("Z")
        cylinder.CreateRadiusAttr(radius)
        height_attr = cylinder.CreateHeightAttr(0.001)
        gprim = UsdGeom.Gprim(cylinder.GetPrim())
        gprim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])

        xform = UsdGeom.Xformable(cylinder.GetPrim())
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        orient_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        return {"height": height_attr, "translate": translate_op, "orient": orient_op}

    def _add_marker(self, name, color, radius):
        sphere = UsdGeom.Sphere.Define(self.stage, f"{self.root_path}/{name}")
        sphere.CreateRadiusAttr(radius)
        gprim = UsdGeom.Gprim(sphere.GetPrim())
        gprim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])

        xform = UsdGeom.Xformable(sphere.GetPrim())
        xform.ClearXformOpOrder()
        return xform.AddTranslateOp()

    def update(self, left_wrist: torch.Tensor, left_handle: torch.Tensor, right_wrist: torch.Tensor, right_handle: torch.Tensor):
        self._update_connector(self.left, left_wrist, left_handle)
        self._update_connector(self.right, right_wrist, right_handle)
        self.left_wrist_marker.Set(_tensor_to_vec3d(left_wrist))
        self.left_handle_marker.Set(_tensor_to_vec3d(left_handle))
        self.right_wrist_marker.Set(_tensor_to_vec3d(right_wrist))
        self.right_handle_marker.Set(_tensor_to_vec3d(right_handle))

    def _update_connector(self, connector, start: torch.Tensor, end: torch.Tensor):
        start_vec = _tensor_to_vec3d(start)
        end_vec = _tensor_to_vec3d(end)
        delta = end_vec - start_vec
        length = max(delta.GetLength(), 1.0e-4)
        direction = delta / length
        midpoint = Gf.Vec3d(
            (start_vec[0] + end_vec[0]) * 0.5,
            (start_vec[1] + end_vec[1]) * 0.5,
            (start_vec[2] + end_vec[2]) * 0.5,
        )
        connector["height"].Set(length)
        connector["translate"].Set(midpoint)
        connector["orient"].Set(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), direction).GetQuat())


def _quat_wxyz_to_yaw(quat: torch.Tensor) -> float:
    w, x, y, z = [float(v) for v in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _tensor_to_vec3d(values: torch.Tensor) -> Gf.Vec3d:
    values = values.detach().cpu()
    return Gf.Vec3d(float(values[0]), float(values[1]), float(values[2]))


def _find_body_index(robot, body_name: str) -> int:
    if hasattr(robot, "find_bodies"):
        indices, _ = robot.find_bodies(body_name)
        if len(indices) > 0:
            return int(indices[0])

    for names in (
        getattr(robot, "body_names", None),
        getattr(getattr(robot, "data", None), "body_names", None),
        getattr(robot, "_body_names", None),
    ):
        if names and body_name in names:
            return names.index(body_name)

    raise ValueError(f"Could not find robot body '{body_name}'.")


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "assets" / "objects" / "wheelchair").exists():
            return parent
    raise RuntimeError("Could not locate unitree_rl_lab repository root.")


def _sanitize_prim_name(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    return cleaned or "default"


def _load_mtl_colors(mtl_path: Path) -> dict[str, tuple[float, float, float]]:
    colors: dict[str, tuple[float, float, float]] = {}
    current_material = None
    if not mtl_path.exists():
        return colors

    with mtl_path.open("r", encoding="utf-8", errors="replace") as mtl_file:
        for raw_line in mtl_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "newmtl" and len(parts) > 1:
                current_material = parts[1]
            elif parts[0] == "Kd" and current_material and len(parts) >= 4:
                colors[current_material] = tuple(float(value) for value in parts[1:4])
    return colors


def _load_obj_mesh_groups(obj_path: Path) -> tuple[list[tuple[float, float, float]], dict[str, list[list[int]]]]:
    vertices: list[tuple[float, float, float]] = []
    faces_by_material: dict[str, list[list[int]]] = {"default": []}
    current_material = "default"

    with obj_path.open("r", encoding="utf-8", errors="replace") as obj_file:
        for raw_line in obj_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "usemtl" and len(parts) > 1:
                current_material = parts[1]
                faces_by_material.setdefault(current_material, [])
            elif parts[0] == "f" and len(parts) >= 4:
                face = [_parse_obj_vertex_index(token, len(vertices)) for token in parts[1:]]
                faces_by_material.setdefault(current_material, []).append(face)

    return vertices, faces_by_material


def _parse_obj_vertex_index(token: str, vertex_count: int) -> int:
    index = int(token.split("/")[0])
    if index < 0:
        return vertex_count + index
    return index - 1


def _remap_faces(
    vertices: list[tuple[float, float, float]], faces: list[list[int]]
) -> tuple[list[Gf.Vec3f], list[int], list[int]]:
    remap: dict[int, int] = {}
    points: list[Gf.Vec3f] = []
    face_counts: list[int] = []
    face_indices: list[int] = []

    for face in faces:
        face_counts.append(len(face))
        for vertex_index in face:
            if vertex_index not in remap:
                remap[vertex_index] = len(points)
                points.append(Gf.Vec3f(*vertices[vertex_index]))
            face_indices.append(remap[vertex_index])

    return points, face_counts, face_indices


def _set_fixed_command(env):
    try:
        command = env.unwrapped.command_manager.get_command("base_velocity")
        command[:, 0] = args_cli.command_x
        command[:, 1] = args_cli.command_y
        command[:, 2] = args_cli.command_yaw
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
        if hasattr(command_term, "is_standing_env"):
            command_term.is_standing_env[:] = False
    except Exception as err:
        if not getattr(_set_fixed_command, "_warned", False):
            print(f"[WARN] Failed to set fixed velocity command: {err}")
            _set_fixed_command._warned = True


def _set_demo_camera(env, chair_pos: torch.Tensor, yaw: float):
    forward = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0])
    left = torch.tensor([-math.sin(yaw), math.cos(yaw), 0.0])
    if args_cli.camera_mode == "fixed":
        if getattr(_set_demo_camera, "_fixed_camera_set", False):
            return
        eye = chair_pos - forward * 2.8 + left * 4.2 + torch.tensor([0.0, 0.0, 2.0])
        target = chair_pos + forward * 3.2 + torch.tensor([0.0, 0.0, 0.65])
        _set_demo_camera._fixed_camera_set = True
    else:
        eye = chair_pos - forward * 3.2 + left * 2.6 + torch.tensor([0.0, 0.0, 1.9])
        target = chair_pos - forward * 0.15 + torch.tensor([0.0, 0.0, 0.75])
    env.unwrapped.sim.set_camera_view(eye.tolist(), target.tolist())


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Keep the demo deterministic and readable: flat floor, one forward command, no command arrows.
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.curriculum = None
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.command_x, args_cli.command_x)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.command_y, args_cli.command_y)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.command_yaw, args_cli.command_yaw)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env = env
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        base_env = env

    if args_cli.wheelchair_visual == "free3d":
        wheelchair = Free3DWheelchairProp()
    else:
        wheelchair = WheelchairProp()
    wheelchair_forward_offset = args_cli.wheelchair_forward_offset
    if wheelchair_forward_offset is None:
        wheelchair_forward_offset = wheelchair.default_forward_offset

    if args_cli.video:
        video_folder = args_cli.video_folder or os.path.join(log_dir, "videos", "wheelchair_push")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == args_cli.video_start_step,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording wheelchair push video.", flush=True)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None) or getattr(policy_nn, "student_obs_normalizer", None)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    robot = base_env.unwrapped.scene["robot"]
    hand_connectors = None
    left_wrist_body_idx = None
    right_wrist_body_idx = None
    if not args_cli.hide_hand_connectors:
        hand_connectors = HandHandleConnectors(radius=args_cli.hand_connector_radius)
        left_wrist_body_idx = _find_body_index(robot, "left_wrist_yaw_link")
        right_wrist_body_idx = _find_body_index(robot, "right_wrist_yaw_link")
    initial_root_pos = robot.data.root_pos_w[0, :2].detach().clone()
    _set_fixed_command(base_env)
    chair_pos, yaw = wheelchair.update_from_robot(robot, wheelchair_forward_offset)
    if hand_connectors is not None:
        left_handle, right_handle = wheelchair.handle_positions_world(chair_pos, yaw)
        hand_connectors.update(
            robot.data.body_pos_w[0, left_wrist_body_idx],
            left_handle,
            robot.data.body_pos_w[0, right_wrist_body_idx],
            right_handle,
        )
    _set_demo_camera(base_env, chair_pos, yaw)

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            _set_fixed_command(base_env)
            chair_pos, yaw = wheelchair.update_from_robot(robot, wheelchair_forward_offset)
            if hand_connectors is not None:
                left_handle, right_handle = wheelchair.handle_positions_world(chair_pos, yaw)
                hand_connectors.update(
                    robot.data.body_pos_w[0, left_wrist_body_idx],
                    left_handle,
                    robot.data.body_pos_w[0, right_wrist_body_idx],
                    right_handle,
                )
            _set_demo_camera(base_env, chair_pos, yaw)
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            _set_fixed_command(base_env)

        if args_cli.video:
            timestep += 1
            if timestep % 100 == 0:
                displacement = torch.linalg.norm(robot.data.root_pos_w[0, :2] - initial_root_pos).item()
                print(f"[INFO] Wheelchair demo step {timestep}: robot XY displacement {displacement:.3f} m", flush=True)
            if timestep >= args_cli.video_start_step + args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    final_displacement = torch.linalg.norm(robot.data.root_pos_w[0, :2] - initial_root_pos).item()
    print(f"[INFO] Wheelchair demo final robot XY displacement: {final_displacement:.3f} m", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
