from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from unitree_rl_lab.assets.objects.wheelchair import ACTIVE_MANUAL_WHEELCHAIR_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

from .velocity_env_cfg import ObservationsCfg, RewardsCfg, RobotEnvCfg, RobotSceneCfg


WHEELCHAIR_HANDLE_TARGETS_B = [
    [0.328, 0.24, 0.12],
    [0.328, -0.24, 0.12],
]
"""Left/right handle-grip targets in the robot root frame, in meters."""


WHEELCHAIR_ARM_JOINT_POSE = {
    "left_shoulder_pitch_joint": -0.495,
    "left_shoulder_roll_joint": 0.374,
    "left_shoulder_yaw_joint": 0.043,
    "left_elbow_joint": 0.664,
    "left_wrist_roll_joint": 0.15,
    "left_wrist_pitch_joint": -0.088,
    "left_wrist_yaw_joint": 0.011,
    "right_shoulder_pitch_joint": -0.495,
    "right_shoulder_roll_joint": -0.374,
    "right_shoulder_yaw_joint": -0.043,
    "right_elbow_joint": 0.664,
    "right_wrist_roll_joint": -0.15,
    "right_wrist_pitch_joint": -0.088,
    "right_wrist_yaw_joint": -0.011,
}
"""Warm-start arm pose placing the rubber hand bodies near the wheelchair handle targets."""


DYNAMIC_WHEELCHAIR_INIT_POS = (0.728, 0.0, 0.0)
"""Wheelchair root pose that aligns the URDF handle frames with the G1 rubber-hand start pose."""


DYNAMIC_WHEELCHAIR_HANDLE_BODY_NAMES = [
    "left_handle_frame",
    "right_handle_frame",
]
"""Left/right wheelchair handle bodies in the passive wheelchair articulation."""


DYNAMIC_WHEELCHAIR_HAND_BODY_NAMES = [
    "left_rubber_hand",
    "right_rubber_hand",
]
"""Left/right G1 hand bodies that should align with the wheelchair handle bodies."""


DYNAMIC_WHEELCHAIR_HAND_HANDLE_ATTACHMENTS = [
    {
        "joint_name": "left_hand_to_handle_anchor_joint",
        "robot_body": "left_rubber_hand",
        "wheelchair_body": "left_handle_frame",
    },
    {
        "joint_name": "right_hand_to_handle_anchor_joint",
        "robot_body": "right_rubber_hand",
        "wheelchair_body": "right_handle_frame",
    },
]
"""Hand-handle joint pairs used by the attached-hands wheelchair task."""


DYNAMIC_WHEELCHAIR_WHEEL_BODY_NAMES = [
    "left_rear_wheel",
    "right_rear_wheel",
    "left_front_caster",
    "right_front_caster",
]
"""Wheelchair wheel/caster bodies that should remain near their ground-contact heights."""


DYNAMIC_WHEELCHAIR_WHEEL_GROUND_HEIGHTS = [
    0.31,
    0.31,
    0.075,
    0.075,
]
"""Nominal world-Z body center heights when all wheelchair wheels are on flat ground."""


DYNAMIC_WHEELCHAIR_INVALID_CONTACT_SENSORS = [
    "wheelchair_left_handle_invalid_contact",
    "wheelchair_right_handle_invalid_contact",
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


DYNAMIC_WHEELCHAIR_LEFT_HANDLE_ALLOWED_BODIES = [
    "left_rubber_hand",
]
"""Robot hand bodies allowed to contact the left wheelchair handle."""


DYNAMIC_WHEELCHAIR_RIGHT_HANDLE_ALLOWED_BODIES = [
    "right_rubber_hand",
]
"""Robot hand bodies allowed to contact the right wheelchair handle."""


DYNAMIC_WHEELCHAIR_LEFT_HANDLE_ALLOWED_FILTERS = [
    f"{{ENV_REGEX_NS}}/Robot/{body_name}" for body_name in DYNAMIC_WHEELCHAIR_LEFT_HANDLE_ALLOWED_BODIES
]
DYNAMIC_WHEELCHAIR_RIGHT_HANDLE_ALLOWED_FILTERS = [
    f"{{ENV_REGEX_NS}}/Robot/{body_name}" for body_name in DYNAMIC_WHEELCHAIR_RIGHT_HANDLE_ALLOWED_BODIES
]


DYNAMIC_WHEELCHAIR_LEFT_HANDLE_INVALID_FILTERS = [
    f"{{ENV_REGEX_NS}}/Robot/{body_name}"
    for body_name in DYNAMIC_WHEELCHAIR_ROBOT_CONTACT_FILTER_BODIES
    if body_name not in DYNAMIC_WHEELCHAIR_LEFT_HANDLE_ALLOWED_BODIES
]
DYNAMIC_WHEELCHAIR_RIGHT_HANDLE_INVALID_FILTERS = [
    f"{{ENV_REGEX_NS}}/Robot/{body_name}"
    for body_name in DYNAMIC_WHEELCHAIR_ROBOT_CONTACT_FILTER_BODIES
    if body_name not in DYNAMIC_WHEELCHAIR_RIGHT_HANDLE_ALLOWED_BODIES
]
"""Exact robot prim paths that are penalized if they touch the wheelchair handles."""


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
                body_names=DYNAMIC_WHEELCHAIR_HAND_BODY_NAMES,
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
                body_names=DYNAMIC_WHEELCHAIR_HAND_BODY_NAMES,
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
    wheelchair.init_state.pos = DYNAMIC_WHEELCHAIR_INIT_POS

    wheelchair_left_handle_robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Wheelchair/left_handle_frame",
        history_length=3,
        filter_prim_paths_expr=DYNAMIC_WHEELCHAIR_LEFT_HANDLE_ALLOWED_FILTERS,
    )
    wheelchair_right_handle_robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Wheelchair/right_handle_frame",
        history_length=3,
        filter_prim_paths_expr=DYNAMIC_WHEELCHAIR_RIGHT_HANDLE_ALLOWED_FILTERS,
    )
    wheelchair_left_handle_invalid_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Wheelchair/left_handle_frame",
        history_length=3,
        filter_prim_paths_expr=DYNAMIC_WHEELCHAIR_LEFT_HANDLE_INVALID_FILTERS,
    )
    wheelchair_right_handle_invalid_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Wheelchair/right_handle_frame",
        history_length=3,
        filter_prim_paths_expr=DYNAMIC_WHEELCHAIR_RIGHT_HANDLE_INVALID_FILTERS,
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
                body_names=DYNAMIC_WHEELCHAIR_HAND_BODY_NAMES,
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
                body_names=DYNAMIC_WHEELCHAIR_HAND_BODY_NAMES,
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
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("wheelchair")},
    )

    wheelchair_forward_line = RewTerm(
        func=mdp.wheelchair_forward_line_l2,
        weight=-1.5,
        params={"allowed_error": 0.05, "asset_cfg": SceneEntityCfg("wheelchair")},
    )

    wheelchair_yaw_velocity = RewTerm(
        func=mdp.wheelchair_yaw_velocity_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("wheelchair")},
    )

    wheelchair_tilt = RewTerm(
        func=mdp.wheelchair_tilt_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("wheelchair")},
    )

    wheelchair_wheel_ground_height = RewTerm(
        func=mdp.wheelchair_wheel_height_l2,
        weight=-50.0,
        params={
            "target_heights": DYNAMIC_WHEELCHAIR_WHEEL_GROUND_HEIGHTS,
            "allowed_error": 0.01,
            "asset_cfg": SceneEntityCfg("wheelchair", body_names=DYNAMIC_WHEELCHAIR_WHEEL_BODY_NAMES),
        },
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

    hand_handle_axis_alignment = RewTerm(
        func=mdp.dynamic_hand_handle_axis_alignment_l2,
        weight=-0.5,
        params={
            "axis": [1.0, 0.0, 0.0],
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=DYNAMIC_WHEELCHAIR_HAND_BODY_NAMES,
            ),
            "wheelchair_cfg": SceneEntityCfg("wheelchair", body_names=DYNAMIC_WHEELCHAIR_HANDLE_BODY_NAMES),
        },
    )

    wrist_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
            )
        },
    )

    wheelchair_invalid_contact = RewTerm(
        func=mdp.filtered_contact_force_penalty,
        weight=-0.02,
        params={"sensor_names": DYNAMIC_WHEELCHAIR_INVALID_CONTACT_SENSORS, "threshold": 3.0},
    )


@configclass
class DynamicWheelchairPushObservedObservationsCfg(ObservationsCfg):
    """Policy and critic observations that expose the passive wheelchair state."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        wheelchair_root_state = ObsTerm(
            func=mdp.wheelchair_root_state_b,
            clip=(-5.0, 5.0),
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "wheelchair_cfg": SceneEntityCfg("wheelchair"),
            },
        )
        wheelchair_handle_state = ObsTerm(
            func=mdp.wheelchair_handle_state_b,
            clip=(-3.0, 3.0),
            params={
                "robot_cfg": SceneEntityCfg(
                    "robot",
                    body_names=DYNAMIC_WHEELCHAIR_HAND_BODY_NAMES,
                ),
                "wheelchair_cfg": SceneEntityCfg("wheelchair", body_names=DYNAMIC_WHEELCHAIR_HANDLE_BODY_NAMES),
            },
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        wheelchair_root_state = ObsTerm(
            func=mdp.wheelchair_root_state_b,
            clip=(-5.0, 5.0),
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "wheelchair_cfg": SceneEntityCfg("wheelchair"),
            },
        )
        wheelchair_handle_state = ObsTerm(
            func=mdp.wheelchair_handle_state_b,
            clip=(-3.0, 3.0),
            params={
                "robot_cfg": SceneEntityCfg(
                    "robot",
                    body_names=DYNAMIC_WHEELCHAIR_HAND_BODY_NAMES,
                ),
                "wheelchair_cfg": SceneEntityCfg("wheelchair", body_names=DYNAMIC_WHEELCHAIR_HANDLE_BODY_NAMES),
            },
        )

    critic: CriticCfg = CriticCfg()


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


@configclass
class DynamicWheelchairPushObservedRobotEnvCfg(DynamicWheelchairPushRobotEnvCfg):
    """Dynamic wheelchair-push task where the policy can observe the wheelchair."""

    observations: DynamicWheelchairPushObservedObservationsCfg = DynamicWheelchairPushObservedObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.35)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.15, 0.55)


@configclass
class DynamicWheelchairPushObservedRobotPlayEnvCfg(DynamicWheelchairPushObservedRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.3)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.3, 0.3)


@configclass
class DynamicWheelchairPushObservedPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "unitree_g1_29dof_wheelchair_dynamic_push_observed"
    max_iterations = 5000
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.005,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.5,
    )


@configclass
class DynamicWheelchairPushAttachedRobotEnvCfg(DynamicWheelchairPushObservedRobotEnvCfg):
    """Dynamic wheelchair-push task with hands anchored to the wheelchair handles."""

    def __post_init__(self):
        super().__post_init__()

        self.events.attach_wheelchair_hands = EventTerm(
            func=mdp.attach_wheelchair_hands_to_handles,
            mode="startup",
            params={
                "attachments": DYNAMIC_WHEELCHAIR_HAND_HANDLE_ATTACHMENTS,
                "joint_type": "spherical",
                "mask_collisions": True,
            },
        )

        self.commands.base_velocity.ranges.lin_vel_x = (0.05, 0.25)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.05, 0.45)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.actions.JointPositionAction.scale = {
            ".*_hip_.*|waist_.*|.*_knee_joint|.*_ankle_.*": 0.25,
            ".*_shoulder_.*|.*_elbow_joint|.*_wrist_.*": 0.0,
        }

        self.rewards.dynamic_hand_handle_position.weight = 0.0
        self.rewards.dynamic_hand_handle_position_l2.weight = 0.0
        self.rewards.hand_handle_axis_alignment.weight = 0.0
        self.rewards.wheelchair_handle_contact.weight = 0.0
        self.rewards.wheelchair_invalid_contact.weight = -0.03
        self.rewards.wrist_joint_deviation.weight = -0.35
        self.rewards.action_rate.weight = 0.0


@configclass
class DynamicWheelchairPushAttachedRobotPlayEnvCfg(DynamicWheelchairPushAttachedRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.15)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.15, 0.15)


@configclass
class DynamicWheelchairPushAttachedPPORunnerCfg(DynamicWheelchairPushObservedPPORunnerCfg):
    experiment_name = "unitree_g1_29dof_wheelchair_dynamic_push_attached"
    max_iterations = 3000


@configclass
class DynamicWheelchairStandingAttachedRewardsCfg(DynamicWheelchairPushRewardsCfg):
    """Rewards for learning to stand with hands attached before pushing."""

    fall_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-200.0,
        params={"term_keys": ["bad_orientation", "base_height"]},
    )

    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.6,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    robot_xy_velocity = RewTerm(
        func=mdp.root_lin_vel_xy_l2,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    robot_yaw_velocity = RewTerm(
        func=mdp.root_ang_vel_z_l2,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    wheelchair_xy_velocity = RewTerm(
        func=mdp.root_lin_vel_xy_l2,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("wheelchair")},
    )


def _configure_dynamic_wheelchair_standing_pretrain(env_cfg):
    """Apply zero-motion standing settings shared by observed and attached stand pretrains."""

    env_cfg.episode_length_s = 10.0
    env_cfg.events.add_base_mass = None
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.terminations.base_height.params["minimum_height"] = 0.55
    env_cfg.terminations.bad_orientation.params["limit_angle"] = 0.55
    env_cfg.actions.JointPositionAction.scale = {
        ".*_hip_.*|waist_.*|.*_knee_joint|.*_ankle_.*": 0.12,
        ".*_shoulder_.*|.*_elbow_joint|.*_wrist_.*": 0.0,
    }

    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.limit_ranges.ang_vel_z = (0.0, 0.0)

    env_cfg.rewards.track_lin_vel_xy.weight = 2.0
    env_cfg.rewards.track_lin_vel_xy.params["std"] = 0.25
    env_cfg.rewards.track_ang_vel_z.weight = 1.0
    env_cfg.rewards.track_ang_vel_z.params["std"] = 0.25
    env_cfg.rewards.alive.weight = 1.0
    env_cfg.rewards.base_linear_velocity.weight = -2.5
    env_cfg.rewards.base_angular_velocity.weight = -0.2
    env_cfg.rewards.joint_vel.weight = -0.002
    env_cfg.rewards.joint_acc.weight = -1.0e-6
    env_cfg.rewards.action_rate.weight = 0.0
    env_cfg.rewards.joint_deviation_waists.weight = -1.0
    env_cfg.rewards.joint_deviation_legs.weight = -1.5
    env_cfg.rewards.flat_orientation_l2.weight = -20.0
    env_cfg.rewards.base_height.weight = -30.0
    env_cfg.rewards.base_height.params["target_height"] = 0.78
    env_cfg.rewards.gait.weight = 0.0
    env_cfg.rewards.feet_clearance.weight = 0.0
    env_cfg.rewards.feet_slide.weight = -0.3

    env_cfg.rewards.hand_handle_position.weight = 0.0
    env_cfg.rewards.hand_handle_position_l2.weight = 0.0
    env_cfg.rewards.dynamic_hand_handle_position.weight = 0.0
    env_cfg.rewards.dynamic_hand_handle_position_l2.weight = 0.0
    env_cfg.rewards.hand_handle_axis_alignment.weight = 0.0
    env_cfg.rewards.wheelchair_track_forward_velocity.weight = 2.0
    env_cfg.rewards.wheelchair_track_forward_velocity.params["std"] = 0.12
    env_cfg.rewards.wheelchair_forward_progress.weight = -1.0
    env_cfg.rewards.wheelchair_lateral_velocity.weight = -2.0
    env_cfg.rewards.wheelchair_forward_line.weight = -2.0
    env_cfg.rewards.wheelchair_yaw_velocity.weight = -1.0
    env_cfg.rewards.wheelchair_tilt.weight = -8.0
    env_cfg.rewards.wheelchair_wheel_ground_height.weight = -80.0
    env_cfg.rewards.wheelchair_invalid_contact.weight = -0.05
    env_cfg.rewards.wrist_joint_deviation.weight = -0.5


@configclass
class DynamicWheelchairStandingObservedRobotEnvCfg(DynamicWheelchairPushObservedRobotEnvCfg):
    """Standing-only pretrain with wheelchair observations but no hand-handle joint."""

    rewards: DynamicWheelchairStandingAttachedRewardsCfg = DynamicWheelchairStandingAttachedRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_dynamic_wheelchair_standing_pretrain(self)


@configclass
class DynamicWheelchairStandingObservedRobotPlayEnvCfg(DynamicWheelchairStandingObservedRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10


@configclass
class DynamicWheelchairStandingObservedPPORunnerCfg(DynamicWheelchairPushObservedPPORunnerCfg):
    experiment_name = "unitree_g1_29dof_wheelchair_dynamic_stand_observed"
    max_iterations = 1500
    num_steps_per_env = 48
    save_interval = 50
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.05,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.01,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=1.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0002,
        max_grad_norm=0.05,
    )


@configclass
class DynamicWheelchairStandingAttachedRobotEnvCfg(DynamicWheelchairPushAttachedRobotEnvCfg):
    """Standing-only pretraining task for the attached-hands wheelchair setup.

    This stage asks the robot to remain upright with both hands attached to the
    handles while the wheelchair stays still. It is intended as a warm-start
    before returning to the forward wheelchair-push task.
    """

    rewards: DynamicWheelchairStandingAttachedRewardsCfg = DynamicWheelchairStandingAttachedRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_dynamic_wheelchair_standing_pretrain(self)


@configclass
class DynamicWheelchairStandingAttachedRobotPlayEnvCfg(DynamicWheelchairStandingAttachedRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10


@configclass
class DynamicWheelchairStandingAttachedPPORunnerCfg(DynamicWheelchairPushObservedPPORunnerCfg):
    experiment_name = "unitree_g1_29dof_wheelchair_dynamic_stand_attached"
    max_iterations = 1500
    num_steps_per_env = 48
    save_interval = 50
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.05,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.01,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=1.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0002,
        max_grad_norm=0.05,
    )
