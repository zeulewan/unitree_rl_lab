from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.objects.wheelchair import ACTIVE_MANUAL_WHEELCHAIR_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

from .velocity_env_cfg import RewardsCfg, RobotEnvCfg, RobotSceneCfg


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


DYNAMIC_WHEELCHAIR_HANDLE_BODY_NAMES = [
    "left_handle_frame",
    "right_handle_frame",
]
"""Left/right wheelchair handle bodies in the passive wheelchair articulation."""


DYNAMIC_WHEELCHAIR_INVALID_CONTACT_SENSORS = [
    "wheelchair_base_robot_contact",
    "wheelchair_left_rear_wheel_robot_contact",
    "wheelchair_right_rear_wheel_robot_contact",
    "wheelchair_left_front_caster_robot_contact",
    "wheelchair_right_front_caster_robot_contact",
]
"""Wheelchair bodies where robot contact is penalized."""


DYNAMIC_WHEELCHAIR_ROBOT_CONTACT_FILTER_BODIES = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_rubber_hand",
    "right_rubber_hand",
]
"""Exact G1 rigid bodies used as wheelchair contact-filter targets."""


DYNAMIC_WHEELCHAIR_ROBOT_CONTACT_FILTERS = [
    f"{{ENV_REGEX_NS}}/Robot/{body_name}" for body_name in DYNAMIC_WHEELCHAIR_ROBOT_CONTACT_FILTER_BODIES
]
"""Exact robot prim paths for PhysX filtered wheelchair contact sensors."""


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


@configclass
class DynamicWheelchairPushSceneCfg(RobotSceneCfg):
    """Flat locomotion scene with a passive manual wheelchair articulation."""

    wheelchair = ACTIVE_MANUAL_WHEELCHAIR_CFG.replace(prim_path="{ENV_REGEX_NS}/Wheelchair")
    wheelchair.init_state.pos = (0.75, 0.0, 0.0)

    wheelchair_left_handle_robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Wheelchair/left_handle_frame",
        history_length=3,
        filter_prim_paths_expr=DYNAMIC_WHEELCHAIR_ROBOT_CONTACT_FILTERS,
    )
    wheelchair_right_handle_robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Wheelchair/right_handle_frame",
        history_length=3,
        filter_prim_paths_expr=DYNAMIC_WHEELCHAIR_ROBOT_CONTACT_FILTERS,
    )
    wheelchair_base_robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Wheelchair/base_link",
        history_length=3,
        filter_prim_paths_expr=DYNAMIC_WHEELCHAIR_ROBOT_CONTACT_FILTERS,
    )
    wheelchair_left_rear_wheel_robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Wheelchair/left_rear_wheel",
        history_length=3,
        filter_prim_paths_expr=DYNAMIC_WHEELCHAIR_ROBOT_CONTACT_FILTERS,
    )
    wheelchair_right_rear_wheel_robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Wheelchair/right_rear_wheel",
        history_length=3,
        filter_prim_paths_expr=DYNAMIC_WHEELCHAIR_ROBOT_CONTACT_FILTERS,
    )
    wheelchair_left_front_caster_robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Wheelchair/left_front_caster",
        history_length=3,
        filter_prim_paths_expr=DYNAMIC_WHEELCHAIR_ROBOT_CONTACT_FILTERS,
    )
    wheelchair_right_front_caster_robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Wheelchair/right_front_caster",
        history_length=3,
        filter_prim_paths_expr=DYNAMIC_WHEELCHAIR_ROBOT_CONTACT_FILTERS,
    )


@configclass
class DynamicWheelchairPushRewardsCfg(WheelchairPushRewardsCfg):
    """Rewards for physically pushing a passive wheelchair forward."""

    dynamic_hand_handle_position = RewTerm(
        func=mdp.dynamic_hand_handle_position_error_exp,
        weight=3.0,
        params={
            "std": 0.08,
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            ),
            "wheelchair_cfg": SceneEntityCfg("wheelchair", body_names=DYNAMIC_WHEELCHAIR_HANDLE_BODY_NAMES),
        },
    )

    dynamic_hand_handle_position_l2 = RewTerm(
        func=mdp.dynamic_hand_handle_position_error_l2,
        weight=-4.0,
        params={
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            ),
            "wheelchair_cfg": SceneEntityCfg("wheelchair", body_names=DYNAMIC_WHEELCHAIR_HANDLE_BODY_NAMES),
        },
    )

    wheelchair_track_forward_velocity = RewTerm(
        func=mdp.wheelchair_forward_velocity_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.25, "asset_cfg": SceneEntityCfg("wheelchair")},
    )

    wheelchair_forward_progress = RewTerm(
        func=mdp.wheelchair_forward_progress,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("wheelchair"), "max_velocity": 1.2},
    )

    wheelchair_lateral_velocity = RewTerm(
        func=mdp.wheelchair_lateral_velocity_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("wheelchair")},
    )

    wheelchair_yaw_velocity = RewTerm(
        func=mdp.wheelchair_yaw_velocity_l2,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("wheelchair")},
    )

    wheelchair_tilt = RewTerm(
        func=mdp.wheelchair_tilt_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("wheelchair")},
    )

    wheelchair_handle_contact = RewTerm(
        func=mdp.filtered_contact_presence,
        weight=0.5,
        params={
            "sensor_names": [
                "wheelchair_left_handle_robot_contact",
                "wheelchair_right_handle_robot_contact",
            ],
            "threshold": 1.0,
        },
    )

    wheelchair_invalid_contact = RewTerm(
        func=mdp.filtered_contact_force_penalty,
        weight=-0.02,
        params={"sensor_names": DYNAMIC_WHEELCHAIR_INVALID_CONTACT_SENSORS, "threshold": 3.0},
    )


@configclass
class DynamicWheelchairPushRobotEnvCfg(WheelchairPushRobotEnvCfg):
    """Trainable wheelchair-push task with a dynamic passive wheelchair in the scene.

    This keeps the policy observation and action spaces identical to the handle-grip proxy so the
    run can warm-start from the existing checkpoint. The wheelchair state is used by rewards only.
    """

    scene: DynamicWheelchairPushSceneCfg = DynamicWheelchairPushSceneCfg(num_envs=2048, env_spacing=4.0)
    rewards: DynamicWheelchairPushRewardsCfg = DynamicWheelchairPushRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_wheelchair = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("wheelchair"),
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )

        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.55)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.25, 0.75)
        self.rewards.hand_handle_position.weight = 0.0
        self.rewards.hand_handle_position_l2.weight = 0.0
        self.rewards.undesired_contacts.weight = 0.0

        for sensor_name in (
            "wheelchair_left_handle_robot_contact",
            "wheelchair_right_handle_robot_contact",
            *DYNAMIC_WHEELCHAIR_INVALID_CONTACT_SENSORS,
        ):
            getattr(self.scene, sensor_name).update_period = self.sim.dt


@configclass
class DynamicWheelchairPushRobotPlayEnvCfg(DynamicWheelchairPushRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.commands.base_velocity.ranges.lin_vel_x = (0.45, 0.45)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.45, 0.45)


@configclass
class DynamicWheelchairPushPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "unitree_g1_29dof_wheelchair_dynamic_push"
    max_iterations = 5000
