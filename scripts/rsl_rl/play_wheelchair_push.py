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
parser.add_argument("--wheelchair-forward-offset", type=float, default=1.05, help="Wheelchair center offset ahead of the robot.")
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

    def __init__(self, prim_path: str = "/World/WheelchairPushDemo"):
        self.stage = omni.usd.get_context().get_stage()
        self.root_path = prim_path
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


def _quat_wxyz_to_yaw(quat: torch.Tensor) -> float:
    w, x, y, z = [float(v) for v in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


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

    wheelchair = WheelchairProp()

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
    initial_root_pos = robot.data.root_pos_w[0, :2].detach().clone()
    _set_fixed_command(base_env)
    chair_pos, yaw = wheelchair.update_from_robot(robot, args_cli.wheelchair_forward_offset)
    _set_demo_camera(base_env, chair_pos, yaw)

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            _set_fixed_command(base_env)
            chair_pos, yaw = wheelchair.update_from_robot(robot, args_cli.wheelchair_forward_offset)
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
