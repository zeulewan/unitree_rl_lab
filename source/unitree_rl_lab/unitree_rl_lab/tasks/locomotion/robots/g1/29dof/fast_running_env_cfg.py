from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .running_env_cfg import RunningRobotEnvCfg


@configclass
class FastRunningRobotEnvCfg(RunningRobotEnvCfg):
    """Higher-speed running variant that starts from the running task."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.8, 1.6),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.1, 0.1),
        )
        self.commands.base_velocity.limit_ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 3.0),
            lin_vel_y=(-0.15, 0.15),
            ang_vel_z=(-0.25, 0.25),
        )


@configclass
class FastRunningRobotPlayEnvCfg(FastRunningRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
