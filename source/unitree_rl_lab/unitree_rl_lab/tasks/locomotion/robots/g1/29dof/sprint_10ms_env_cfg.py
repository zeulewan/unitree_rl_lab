import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

from .fast_running_env_cfg import FastRunningRobotEnvCfg

SPRINT_TILE_SIZE = (80.0, 80.0)
SPRINT_TERRAIN_ROWS = 21
SPRINT_TERRAIN_COLS = 21


@configclass
class Sprint10msRobotEnvCfg(FastRunningRobotEnvCfg):
    """10 m/s sprint stress-test variant that starts from the fast running task."""

    def __post_init__(self):
        super().__post_init__()

        # A 20 second episode at 10 m/s can cover 200 m. Use a large flat map
        # and grid-spaced origins so failures are policy failures, not terrain
        # edge artifacts.
        self.scene.env_spacing = 8.0
        self.scene.terrain.use_terrain_origins = False
        self.scene.terrain.terrain_generator.size = SPRINT_TILE_SIZE
        self.scene.terrain.terrain_generator.border_width = 80.0
        self.scene.terrain.terrain_generator.num_rows = SPRINT_TERRAIN_ROWS
        self.scene.terrain.terrain_generator.num_cols = SPRINT_TERRAIN_COLS
        self.scene.terrain.terrain_generator.curriculum = False
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(2.0, 4.0),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.05, 0.05),
        )
        self.commands.base_velocity.limit_ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 10.0),
            lin_vel_y=(-0.10, 0.10),
            ang_vel_z=(-0.15, 0.15),
        )

        self.rewards.track_lin_vel_xy.weight = 2.0
        self.rewards.track_lin_vel_xy.params["std"] = math.sqrt(1.0)
        self.rewards.track_ang_vel_z.weight = 0.10

        self.rewards.gait.weight = 0.50
        self.rewards.gait.params["period"] = 0.45
        self.rewards.gait.params["threshold"] = 0.45
        self.rewards.feet_clearance.params["target_height"] = 0.16
        self.rewards.feet_slide.weight = -0.10

        self.rewards.action_rate.weight = -0.02
        self.rewards.energy.weight = -1.0e-5
        self.rewards.flat_orientation_l2.weight = -3.0
        self.terminations.bad_orientation.params["limit_angle"] = 1.0


@configclass
class Sprint10msRobotPlayEnvCfg(Sprint10msRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class Sprint10msCurriculumResumeEnvCfg(Sprint10msRobotEnvCfg):
    """Resume sprint curriculum from the latest observed paused range."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 6.5),
            lin_vel_y=(-0.10, 0.10),
            ang_vel_z=(-0.05, 0.05),
        )


@configclass
class Sprint10msCurriculumResumePlayEnvCfg(Sprint10msCurriculumResumeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class Sprint10msCurriculumResumePPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "unitree_g1_29dof_sprint_10ms"


@configclass
class Sprint10msGaitRobotEnvCfg(Sprint10msRobotEnvCfg):
    """Straight sprint variant that gives the policy more room to learn a running gait."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(5.5, 6.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        self.commands.base_velocity.limit_ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(5.5, 10.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )

        self.rewards.joint_deviation_waists.weight = -0.20
        self.rewards.joint_deviation_legs.weight = -0.35
        self.rewards.track_lin_vel_xy.weight = 2.25
        self.rewards.track_lin_vel_xy.params["std"] = math.sqrt(0.49)
        self.rewards.track_ang_vel_z.weight = 0.20

        self.rewards.gait.weight = 0.35
        self.rewards.gait.params["period"] = 0.38
        self.rewards.gait.params["threshold"] = 0.42
        self.rewards.feet_clearance.params["target_height"] = 0.17
        self.rewards.feet_slide.weight = -0.12

        self.rewards.action_rate.weight = -0.015
        self.rewards.energy.weight = -8.0e-6
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.base_height.weight = -6.0
        self.rewards.base_height.params["target_height"] = 0.74

        self.curriculum.lin_vel_cmd_levels = CurrTerm(
            func=mdp.stable_lin_vel_cmd_levels,
            params={
                "reward_success_ratio": 0.82,
                "max_failure_rate": 0.20,
                "min_episode_length_ratio": 0.75,
                "expand_lower": False,
                "expand_lateral": False,
            },
        )
        self.curriculum.lin_vel_cmd_stability = CurrTerm(
            func=mdp.lin_vel_cmd_stability,
            params={"failure_term_names": ("bad_orientation", "base_height")},
        )


@configclass
class Sprint10msGaitRobotPlayEnvCfg(Sprint10msGaitRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(5.5, 6.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )


@configclass
class Sprint10msGaitPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "unitree_g1_29dof_sprint_10ms_gait"
