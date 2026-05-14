# Run trained G1 locomotion policy in a warehouse scene.
# Based on Isaac Lab's "Policy Inference in USD Environment" tutorial.
#
# Usage:
#   conda activate isaaclab
#   cd ~/GIT/unitree_rl_lab
#   python scripts/rsl_rl/play_warehouse.py

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 locomotion policy in warehouse")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robots")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to JIT checkpoint (.pt)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import io
import os
import sys
import pathlib

import torch
import omni

# Make sure unitree_rl_lab tasks are registered
sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401
sys.path.pop(0)

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg

# Import the PLAY config from our training task (exact same config play.py uses)
# Can't use normal import because "29dof" starts with a digit
import importlib
_vel_cfg = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg")
RobotPlayEnvCfg = _vel_cfg.RobotPlayEnvCfg


WAREHOUSE_USD = os.path.expanduser(
    "~/GIT/unitree_sim_isaaclab/assets/objects/small_warehouse/small_warehouse_digital_twin.usd"
)


def main():
    # Find checkpoint
    if args_cli.checkpoint:
        policy_path = os.path.abspath(args_cli.checkpoint)
    else:
        # Default: latest training run
        log_dir = os.path.expanduser(
            "~/GIT/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/"
        )
        runs = sorted(os.listdir(log_dir))
        latest_run = os.path.join(log_dir, runs[-1], "exported", "policy.pt")
        policy_path = latest_run
    print(f"[Warehouse] Loading policy from: {policy_path}")

    # Load JIT policy
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=args_cli.device)

    # Setup environment: training config + warehouse terrain
    env_cfg = RobotPlayEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.curriculum = None

    # Swap terrain from flat to warehouse
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=WAREHOUSE_USD,
    )

    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    # Create environment (uses the EXACT same obs/action pipeline as training)
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Run inference
    obs, _ = env.reset()
    print("[Warehouse] Running policy inference. Robot should walk.")
    with torch.inference_mode():
        while simulation_app.is_running():
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
    simulation_app.close()
