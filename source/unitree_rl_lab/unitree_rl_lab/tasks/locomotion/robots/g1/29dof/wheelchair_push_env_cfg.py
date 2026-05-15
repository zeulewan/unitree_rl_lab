from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

from .velocity_env_cfg import RewardsCfg, RobotEnvCfg


WHEELCHAIR_HANDLE_TARGETS_B = [
    [0.35, 0.24, 0.18],
    [0.35, -0.24, 0.18],
]
"""Left/right handle-grip targets in the robot root frame, in meters."""


WHEELCHAIR_ARM_JOINT_POSE = {
    "left_shoulder_pitch_joint": -0.53,
    "left_shoulder_roll_joint": 0.41,
    "left_shoulder_yaw_joint": 0.06,
    "left_elbow_joint": 0.57,
    "left_wrist_roll_joint": 0.15,
    "left_wrist_pitch_joint": -0.13,
    "left_wrist_yaw_joint": 0.02,
    "right_shoulder_pitch_joint": -0.53,
    "right_shoulder_roll_joint": -0.41,
    "right_shoulder_yaw_joint": -0.06,
    "right_elbow_joint": 0.57,
    "right_wrist_roll_joint": -0.15,
    "right_wrist_pitch_joint": -0.13,
    "right_wrist_yaw_joint": -0.02,
}
"""Warm-start arm pose placing the wrist-yaw links near the wheelchair handle targets."""


@configclass
class WheelchairPushRewardsCfg(RewardsCfg):
    """Reward terms for walking forward while the hands stay on wheelchair handles."""

    hand_handle_position = RewTerm(
        func=mdp.hand_handle_position_error_exp,
        weight=2.0,
        params={
            "target_positions_b": WHEELCHAIR_HANDLE_TARGETS_B,
            "std": 0.08,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            ),
        },
    )

    hand_handle_position_l2 = RewTerm(
        func=mdp.hand_handle_position_error_l2,
        weight=-2.0,
        params={
            "target_positions_b": WHEELCHAIR_HANDLE_TARGETS_B,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            ),
        },
    )


@configclass
class WheelchairPushRobotEnvCfg(RobotEnvCfg):
    """G1 locomotion task with a fixed wheelchair-handle grip posture.

    This is the first trainable wheelchair-push proxy. It does not model chair
    wheel dynamics yet; instead it makes the arm posture and hand-handle
    attachment a policy objective while preserving the base walking policy's
    observation and action spaces for checkpoint warm-starting.
    """

    rewards: WheelchairPushRewardsCfg = WheelchairPushRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.init_state.joint_pos.pop(".*_shoulder_pitch_joint", None)
        self.scene.robot.init_state.joint_pos.pop(".*_elbow_joint", None)
        self.scene.robot.init_state.joint_pos.update(WHEELCHAIR_ARM_JOINT_POSE)

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum = None
        self.events.push_robot = None

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.85)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.35, 0.85)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (0.0, 0.0)

        self.rewards.track_ang_vel_z.weight = 0.25
        self.rewards.joint_deviation_arms.weight = -0.03
        self.rewards.joint_deviation_waists.weight = -0.5


@configclass
class WheelchairPushRobotPlayEnvCfg(WheelchairPushRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.commands.base_velocity.ranges.lin_vel_x = (0.55, 0.55)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.55, 0.55)


@configclass
class WheelchairPushPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "unitree_g1_29dof_wheelchair_push"
    max_iterations = 5000
