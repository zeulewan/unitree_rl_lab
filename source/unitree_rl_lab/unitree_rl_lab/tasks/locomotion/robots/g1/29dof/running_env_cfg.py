import math

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import RobotEnvCfg


@configclass
class RunningRobotEnvCfg(RobotEnvCfg):
    """Forward-running variant of the G1 velocity-tracking task."""

    def __post_init__(self):
        super().__post_init__()

        # Keep the policy interface identical to the walking task, but shift the
        # command curriculum and gait target toward faster forward locomotion.
        self.commands.base_velocity.rel_standing_envs = 0.01
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.8),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.1, 0.1),
        )
        self.commands.base_velocity.limit_ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 2.0),
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.3, 0.3),
        )

        self.rewards.track_lin_vel_xy.weight = 1.5
        self.rewards.track_lin_vel_xy.params["std"] = math.sqrt(0.36)
        self.rewards.track_ang_vel_z.weight = 0.25

        self.rewards.gait.weight = 0.75
        self.rewards.gait.params["period"] = 0.6
        self.rewards.gait.params["threshold"] = 0.5
        self.rewards.feet_clearance.params["target_height"] = 0.13

        self.rewards.action_rate.weight = -0.035
        self.rewards.energy.weight = -1.5e-5


@configclass
class RunningRobotPlayEnvCfg(RunningRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
