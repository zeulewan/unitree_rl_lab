from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from unitree_rl_lab.assets.objects.wheelchair import (
    ACTIVE_MANUAL_WHEELCHAIR_BRAKED_CFG,
    ACTIVE_MANUAL_WHEELCHAIR_CFG,
    ACTIVE_MANUAL_WHEELCHAIR_FIXED_BASE_CFG,
    ACTIVE_MANUAL_WHEELCHAIR_NO_COLLISION_CFG,
    ACTIVE_MANUAL_WHEELCHAIR_X_RAIL_NO_COLLISION_CFG,
)
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

from .velocity_env_cfg import ObservationsCfg, RewardsCfg, RobotEnvCfg, RobotSceneCfg


HAND_GRIP_LOCAL_POSITIONS = [
    [0.05414, -0.00372, 0.00502],
    [0.05414, 0.00372, 0.00502],
]
"""Left/right local offsets from each rubber-hand body origin to the intended palm grip point."""


WHEELCHAIR_HANDLE_TARGETS_B = [
    [0.382, 0.24, 0.12],
    [0.382, -0.24, 0.12],
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


G1_NEUTRAL_ARM_JOINT_POSE = {
    ".*_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    ".*_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "right_wrist_roll_joint": -0.15,
    ".*_wrist_pitch_joint": 0.0,
    ".*_wrist_yaw_joint": 0.0,
}
"""Default G1 arm pose used by the base walking/standing checkpoint."""


DYNAMIC_WHEELCHAIR_INIT_POS = (0.782, 0.0, 0.0)
"""Wheelchair root pose that aligns the URDF handle frames with the G1 palm grip points."""


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
        "robot_local_pos": HAND_GRIP_LOCAL_POSITIONS[0],
    },
    {
        "joint_name": "right_hand_to_handle_anchor_joint",
        "robot_body": "right_rubber_hand",
        "wheelchair_body": "right_handle_frame",
        "robot_local_pos": HAND_GRIP_LOCAL_POSITIONS[1],
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
            "body_local_positions": HAND_GRIP_LOCAL_POSITIONS,
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
            "body_local_positions": HAND_GRIP_LOCAL_POSITIONS,
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
class FixedBaseDynamicWheelchairPushSceneCfg(DynamicWheelchairPushSceneCfg):
    """Wheelchair scene with the chair root fixed for first attached-hands standing."""

    wheelchair = ACTIVE_MANUAL_WHEELCHAIR_FIXED_BASE_CFG.replace(prim_path="{ENV_REGEX_NS}/Wheelchair")
    wheelchair.init_state.pos = DYNAMIC_WHEELCHAIR_INIT_POS


@configclass
class BrakedDynamicWheelchairPushSceneCfg(DynamicWheelchairPushSceneCfg):
    """Wheelchair scene with passive wheel/caster braking but a free root."""

    wheelchair = ACTIVE_MANUAL_WHEELCHAIR_BRAKED_CFG.replace(prim_path="{ENV_REGEX_NS}/Wheelchair")
    wheelchair.init_state.pos = DYNAMIC_WHEELCHAIR_INIT_POS


@configclass
class NoCollisionDynamicWheelchairPushSceneCfg(DynamicWheelchairPushSceneCfg):
    """Wheelchair scene that preserves handles/state but disables chair collisions."""

    wheelchair = ACTIVE_MANUAL_WHEELCHAIR_NO_COLLISION_CFG.replace(prim_path="{ENV_REGEX_NS}/Wheelchair")
    wheelchair.init_state.pos = DYNAMIC_WHEELCHAIR_INIT_POS


@configclass
class PhysXRailNoCollisionDynamicWheelchairPushSceneCfg(DynamicWheelchairPushSceneCfg):
    """No-collision wheelchair scene with a real prismatic X-rail articulation joint."""

    wheelchair = ACTIVE_MANUAL_WHEELCHAIR_X_RAIL_NO_COLLISION_CFG.replace(prim_path="{ENV_REGEX_NS}/Wheelchair")
    wheelchair.init_state.pos = DYNAMIC_WHEELCHAIR_INIT_POS


@configclass
class DynamicWheelchairPushRewardsCfg(WheelchairPushRewardsCfg):
    """Rewards for physically pushing a passive wheelchair forward."""

    dynamic_hand_handle_position = RewTerm(
        func=mdp.dynamic_hand_handle_position_error_exp,
        weight=3.0,
        params={
            "std": 0.08,
            "robot_body_local_positions": HAND_GRIP_LOCAL_POSITIONS,
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
            "robot_body_local_positions": HAND_GRIP_LOCAL_POSITIONS,
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

    wheelchair_backward_velocity = RewTerm(
        func=mdp.wheelchair_backward_velocity_l2,
        weight=0.0,
        params={"max_velocity": 3.0, "asset_cfg": SceneEntityCfg("wheelchair")},
    )

    robot_forward_lean = RewTerm(
        func=mdp.root_forward_lean_exp,
        weight=0.0,
        params={"target": 0.17, "std": 0.20, "asset_cfg": SceneEntityCfg("robot")},
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

    wheelchair_rail_yaw_torque = RewTerm(
        func=mdp.body_incoming_joint_torque_axis_l2,
        weight=0.0,
        params={
            "axis": "z",
            "scale": 100.0,
            "max_abs_torque": 250.0,
            "asset_cfg": SceneEntityCfg("wheelchair", body_names="base_link"),
        },
    )

    wheelchair_root_heading = RewTerm(
        func=mdp.root_heading_lateral_l2,
        weight=0.0,
        params={"allowed_error": 0.03, "asset_cfg": SceneEntityCfg("wheelchair")},
    )

    wheelchair_forward_heading = RewTerm(
        func=mdp.root_forward_heading_l2,
        weight=0.0,
        params={"allowed_error": 0.005, "asset_cfg": SceneEntityCfg("wheelchair")},
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
        func=mdp.joint_position_l1,
        weight=-0.25,
        params={
            "target": 0.0,
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
                "robot_body_local_positions": HAND_GRIP_LOCAL_POSITIONS,
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
                "robot_body_local_positions": HAND_GRIP_LOCAL_POSITIONS,
                "robot_cfg": SceneEntityCfg(
                    "robot",
                    body_names=DYNAMIC_WHEELCHAIR_HAND_BODY_NAMES,
                ),
                "wheelchair_cfg": SceneEntityCfg("wheelchair", body_names=DYNAMIC_WHEELCHAIR_HANDLE_BODY_NAMES),
            },
        )

    critic: CriticCfg = CriticCfg()


@configclass
class DynamicWheelchairPushSoftAttachmentObservationsCfg(DynamicWheelchairPushObservedObservationsCfg):
    """Wheelchair observations plus soft hand-handle attachment load state."""

    @configclass
    class PolicyCfg(DynamicWheelchairPushObservedObservationsCfg.PolicyCfg):
        wheelchair_soft_attachment_state = ObsTerm(
            func=mdp.wheelchair_soft_attachment_state_b,
            clip=(-3.0, 3.0),
            params={
                "robot_body_local_positions": HAND_GRIP_LOCAL_POSITIONS,
                "robot_cfg": SceneEntityCfg("robot", body_names=DYNAMIC_WHEELCHAIR_HAND_BODY_NAMES),
                "wheelchair_cfg": SceneEntityCfg("wheelchair", body_names=DYNAMIC_WHEELCHAIR_HANDLE_BODY_NAMES),
                "stiffness": 2500.0,
                "damping": 75.0,
                "max_force": 350.0,
                "force_scale": 350.0,
            },
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(DynamicWheelchairPushObservedObservationsCfg.CriticCfg):
        wheelchair_soft_attachment_state = ObsTerm(
            func=mdp.wheelchair_soft_attachment_state_b,
            clip=(-3.0, 3.0),
            params={
                "robot_body_local_positions": HAND_GRIP_LOCAL_POSITIONS,
                "robot_cfg": SceneEntityCfg("robot", body_names=DYNAMIC_WHEELCHAIR_HAND_BODY_NAMES),
                "wheelchair_cfg": SceneEntityCfg("wheelchair", body_names=DYNAMIC_WHEELCHAIR_HANDLE_BODY_NAMES),
                "stiffness": 2500.0,
                "damping": 75.0,
                "max_force": 350.0,
                "force_scale": 350.0,
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
            mode="reset",
            params={
                "attachments": DYNAMIC_WHEELCHAIR_HAND_HANDLE_ATTACHMENTS,
                "joint_type": "spherical",
                "mask_collisions": True,
                "anchor_at_body_origins": False,
                "skip_existing": True,
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
class RelaxedDynamicWheelchairPushAttachedRobotEnvCfg(DynamicWheelchairPushAttachedRobotEnvCfg):
    """Walking phase after attached standing: free chair, attached hands, relaxed arm control."""

    def __post_init__(self):
        super().__post_init__()

        self.events.attach_wheelchair_hands.params["mask_collisions"] = True
        self.actions.JointPositionAction.scale = {
            ".*_hip_.*|waist_.*|.*_knee_joint|.*_ankle_.*": 0.18,
            ".*_shoulder_.*|.*_elbow_joint": 0.05,
            ".*_wrist_.*": 0.015,
        }

        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.22)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.05, 0.35)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (0.0, 0.0)

        self.terminations.base_height.params["minimum_height"] = 0.5
        self.terminations.bad_orientation.params["limit_angle"] = 0.65

        self.rewards.track_lin_vel_xy.weight = 1.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.3
        self.rewards.track_ang_vel_z.weight = 0.6
        self.rewards.track_ang_vel_z.params["std"] = 0.25
        self.rewards.base_linear_velocity.weight = -1.0
        self.rewards.base_angular_velocity.weight = -0.1
        self.rewards.action_rate.weight = -0.01
        self.rewards.joint_deviation_arms.weight = -0.02
        self.rewards.joint_deviation_waists.weight = -0.7
        self.rewards.joint_deviation_legs.weight = -1.0
        self.rewards.flat_orientation_l2.weight = -8.0
        self.rewards.base_height.weight = -15.0
        self.rewards.gait.weight = 0.35
        self.rewards.feet_clearance.weight = 0.5
        self.rewards.feet_slide.weight = -0.2

        self.rewards.wheelchair_track_forward_velocity.weight = 3.0
        self.rewards.wheelchair_track_forward_velocity.params["std"] = 0.18
        self.rewards.wheelchair_forward_progress.weight = 0.6
        self.rewards.wheelchair_forward_progress.params["max_velocity"] = 0.6
        self.rewards.wheelchair_lateral_velocity.weight = -3.0
        self.rewards.wheelchair_forward_line.weight = -4.0
        self.rewards.wheelchair_yaw_velocity.weight = -2.0
        self.rewards.wheelchair_tilt.weight = -10.0
        self.rewards.wheelchair_wheel_ground_height.weight = -100.0
        self.rewards.wheelchair_invalid_contact.weight = -0.02
        self.rewards.wrist_joint_deviation.weight = -0.25


@configclass
class RelaxedDynamicWheelchairPushAttachedRobotPlayEnvCfg(RelaxedDynamicWheelchairPushAttachedRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.commands.base_velocity.ranges.lin_vel_x = (0.18, 0.18)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.18, 0.18)


@configclass
class RelaxedDynamicWheelchairPushAttachedPPORunnerCfg(DynamicWheelchairPushObservedPPORunnerCfg):
    experiment_name = "unitree_g1_29dof_wheelchair_relaxed_push_attached"
    max_iterations = 2500
    num_steps_per_env = 48
    save_interval = 50
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.001,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0005,
        max_grad_norm=0.1,
    )


@configclass
class MinimalVelocityDynamicWheelchairPushAttachedRobotEnvCfg(RelaxedDynamicWheelchairPushAttachedRobotEnvCfg):
    """Diagnostic attached-hands push task with only wheelchair forward velocity reward active."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.18)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.05, 0.24)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False

        self.terminations.base_height.params["minimum_height"] = 0.30
        self.terminations.bad_orientation.params["limit_angle"] = 1.20

        self.actions.JointPositionAction.scale = {
            ".*_hip_.*|.*_knee_joint|.*_ankle_.*": 0.22,
            "waist_.*": 0.25,
            ".*_shoulder_.*|.*_elbow_joint": 0.08,
            ".*_wrist_.*": 0.02,
        }

        for reward_name in (
            "track_lin_vel_xy",
            "track_ang_vel_z",
            "alive",
            "base_linear_velocity",
            "base_angular_velocity",
            "joint_vel",
            "joint_acc",
            "action_rate",
            "dof_pos_limits",
            "energy",
            "joint_deviation_arms",
            "joint_deviation_waists",
            "joint_deviation_legs",
            "flat_orientation_l2",
            "base_height",
            "gait",
            "feet_slide",
            "feet_clearance",
            "undesired_contacts",
            "hand_handle_position",
            "hand_handle_position_l2",
            "dynamic_hand_handle_position",
            "dynamic_hand_handle_position_l2",
            "wheelchair_forward_progress",
            "wheelchair_backward_velocity",
            "wheelchair_lateral_velocity",
            "wheelchair_forward_line",
            "wheelchair_yaw_velocity",
            "wheelchair_root_heading",
            "wheelchair_forward_heading",
            "wheelchair_tilt",
            "wheelchair_wheel_ground_height",
            "wheelchair_handle_contact",
            "hand_handle_axis_alignment",
            "wrist_joint_deviation",
            "wheelchair_invalid_contact",
        ):
            if hasattr(self.rewards, reward_name):
                getattr(self.rewards, reward_name).weight = 0.0

        self.rewards.wheelchair_track_forward_velocity.weight = 6.0
        self.rewards.wheelchair_track_forward_velocity.params["std"] = 0.05


@configclass
class MinimalVelocityDynamicWheelchairPushAttachedRobotPlayEnvCfg(
    MinimalVelocityDynamicWheelchairPushAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.commands.base_velocity.ranges.lin_vel_x = (0.14, 0.14)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.14, 0.14)


@configclass
class MinimalVelocityDynamicWheelchairPushAttachedPPORunnerCfg(RelaxedDynamicWheelchairPushAttachedPPORunnerCfg):
    experiment_name = "unitree_g1_29dof_wheelchair_minimal_velocity_push_attached"
    max_iterations = 1000


@configclass
class MinimalStraightVelocityDynamicWheelchairPushAttachedRobotEnvCfg(
    MinimalVelocityDynamicWheelchairPushAttachedRobotEnvCfg
):
    """Minimal forward-push task that adds only straightness terms to prevent circular exploits."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.wheelchair_track_forward_velocity.weight = 6.0
        self.rewards.wheelchair_track_forward_velocity.params["std"] = 0.09
        self.rewards.wheelchair_forward_progress.weight = 2.0
        self.rewards.wheelchair_forward_progress.params["max_velocity"] = 0.3
        self.rewards.wheelchair_lateral_velocity.weight = -3.0
        self.rewards.wheelchair_forward_line.weight = -15.0
        self.rewards.wheelchair_forward_line.params["allowed_error"] = 0.03
        self.rewards.wheelchair_yaw_velocity.weight = -2.0
        self.rewards.wheelchair_root_heading.weight = -8.0
        self.rewards.wheelchair_root_heading.params["allowed_error"] = 0.03
        self.rewards.wheelchair_forward_heading.weight = -4.0
        self.rewards.wheelchair_forward_heading.params["allowed_error"] = 0.005


@configclass
class MinimalStraightVelocityDynamicWheelchairPushAttachedRobotPlayEnvCfg(
    MinimalStraightVelocityDynamicWheelchairPushAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.commands.base_velocity.ranges.lin_vel_x = (0.14, 0.14)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.14, 0.14)


@configclass
class MinimalStraightVelocityDynamicWheelchairPushAttachedPPORunnerCfg(
    MinimalVelocityDynamicWheelchairPushAttachedPPORunnerCfg
):
    experiment_name = "unitree_g1_29dof_wheelchair_minimal_straight_velocity_push_attached"
    max_iterations = 1000


@configclass
class MinimalYawLockedVelocityDynamicWheelchairPushAttachedRobotEnvCfg(
    MinimalVelocityDynamicWheelchairPushAttachedRobotEnvCfg
):
    """Minimal forward-push task with the wheelchair constrained to an X-axis rail."""

    def __post_init__(self):
        super().__post_init__()

        env_step_dt = self.decimation * self.sim.dt
        self.events.constrain_wheelchair_to_forward_rail = EventTerm(
            func=mdp.constrain_root_to_forward_rail,
            mode="interval",
            interval_range_s=(env_step_dt, env_step_dt),
            params={
                "asset_cfg": SceneEntityCfg("wheelchair"),
                "lateral_position": DYNAMIC_WHEELCHAIR_INIT_POS[1],
                "root_height": DYNAMIC_WHEELCHAIR_INIT_POS[2],
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "zero_lateral_velocity": True,
                "zero_vertical_velocity": True,
                "zero_roll_pitch_velocity": True,
                "zero_yaw_velocity": True,
            },
        )

        self.terminations.base_height.params["minimum_height"] = 0.08
        self.terminations.bad_orientation.params["limit_angle"] = 2.40

        self.actions.JointPositionAction.scale = {
            ".*_hip_.*|.*_knee_joint|.*_ankle_.*": 0.35,
            "waist_.*": 0.40,
            ".*_shoulder_.*|.*_elbow_joint": 0.20,
            ".*_wrist_.*": 0.08,
        }

        self.rewards.wheelchair_track_forward_velocity.weight = 6.0
        self.rewards.wheelchair_track_forward_velocity.params["std"] = 0.10
        self.rewards.wheelchair_forward_progress.weight = 1.0
        self.rewards.wheelchair_forward_progress.params["max_velocity"] = 0.35


@configclass
class MinimalYawLockedVelocityDynamicWheelchairPushAttachedRobotPlayEnvCfg(
    MinimalYawLockedVelocityDynamicWheelchairPushAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.commands.base_velocity.ranges.lin_vel_x = (0.14, 0.14)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.14, 0.14)


@configclass
class MinimalYawLockedVelocityDynamicWheelchairPushAttachedPPORunnerCfg(
    MinimalVelocityDynamicWheelchairPushAttachedPPORunnerCfg
):
    experiment_name = "unitree_g1_29dof_wheelchair_minimal_yaw_locked_velocity_push_attached"
    max_iterations = 1000


@configclass
class MinimalXRailProgressDynamicWheelchairPushAttachedRobotEnvCfg(
    MinimalYawLockedVelocityDynamicWheelchairPushAttachedRobotEnvCfg
):
    """X-rail wheelchair push scaffold with only forward progress rewarded."""

    scene: NoCollisionDynamicWheelchairPushSceneCfg = NoCollisionDynamicWheelchairPushSceneCfg(
        num_envs=2048,
        env_spacing=4.0,
    )

    def __post_init__(self):
        super().__post_init__()

        self.rewards.wheelchair_track_forward_velocity.weight = 0.0
        self.rewards.wheelchair_forward_progress.weight = 6.0
        self.rewards.wheelchair_forward_progress.params["max_velocity"] = 0.8


@configclass
class MinimalXRailProgressDynamicWheelchairPushAttachedRobotPlayEnvCfg(
    MinimalXRailProgressDynamicWheelchairPushAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.commands.base_velocity.ranges.lin_vel_x = (0.14, 0.14)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.14, 0.14)


@configclass
class MinimalXRailProgressDynamicWheelchairPushAttachedPPORunnerCfg(
    MinimalVelocityDynamicWheelchairPushAttachedPPORunnerCfg
):
    experiment_name = "unitree_g1_29dof_wheelchair_minimal_x_rail_progress_push_attached"
    max_iterations = 1000


@configclass
class MinimalXRailVelocityProgressDynamicWheelchairPushAttachedRobotEnvCfg(
    MinimalXRailProgressDynamicWheelchairPushAttachedRobotEnvCfg
):
    """X-rail wheelchair push scaffold with velocity tracking plus a backward-motion penalty."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.wheelchair_track_forward_velocity.weight = 6.0
        self.rewards.wheelchair_track_forward_velocity.params["std"] = 0.08
        self.rewards.wheelchair_forward_progress.weight = 1.0
        self.rewards.wheelchair_forward_progress.params["max_velocity"] = 0.35
        self.rewards.wheelchair_backward_velocity.weight = -10.0


@configclass
class MinimalXRailVelocityProgressDynamicWheelchairPushAttachedRobotPlayEnvCfg(
    MinimalXRailVelocityProgressDynamicWheelchairPushAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.commands.base_velocity.ranges.lin_vel_x = (0.14, 0.14)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.14, 0.14)


@configclass
class MinimalXRailVelocityProgressDynamicWheelchairPushAttachedPPORunnerCfg(
    MinimalVelocityDynamicWheelchairPushAttachedPPORunnerCfg
):
    experiment_name = "unitree_g1_29dof_wheelchair_minimal_x_rail_velocity_progress_push_attached"
    max_iterations = 1000


@configclass
class MinimalXRailFastVelocityProgressDynamicWheelchairPushAttachedRobotEnvCfg(
    MinimalXRailVelocityProgressDynamicWheelchairPushAttachedRobotEnvCfg
):
    """X-rail no-collision push scaffold targeting 2 m/s chair speed without pose shaping."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 2.0)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (2.0, 2.0)

        self.actions.JointPositionAction.scale = {
            ".*_hip_.*|.*_knee_joint|.*_ankle_.*": 0.50,
            "waist_.*": 0.50,
            ".*_shoulder_.*|.*_elbow_joint": 0.25,
            ".*_wrist_.*": 0.08,
        }

        self.rewards.wheelchair_track_forward_velocity.weight = 10.0
        self.rewards.wheelchair_track_forward_velocity.params["std"] = 0.8
        self.rewards.wheelchair_forward_progress.weight = 3.0
        self.rewards.wheelchair_forward_progress.params["max_velocity"] = 2.5
        self.rewards.wheelchair_backward_velocity.weight = -10.0


@configclass
class MinimalXRailFastVelocityProgressDynamicWheelchairPushAttachedRobotPlayEnvCfg(
    MinimalXRailFastVelocityProgressDynamicWheelchairPushAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10


@configclass
class MinimalXRailFastVelocityProgressDynamicWheelchairPushAttachedPPORunnerCfg(
    MinimalVelocityDynamicWheelchairPushAttachedPPORunnerCfg
):
    experiment_name = "unitree_g1_29dof_wheelchair_minimal_x_rail_fast_velocity_progress_push_attached"
    max_iterations = 1000
    save_interval = 50


@configclass
class MinimalXRailFastLeanVelocityProgressDynamicWheelchairPushAttachedRobotEnvCfg(
    MinimalXRailFastVelocityProgressDynamicWheelchairPushAttachedRobotEnvCfg
):
    """2 m/s X-rail scaffold with a small forward root-lean bias."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.robot_forward_lean.weight = 1.0
        self.rewards.robot_forward_lean.params["target"] = 0.17
        self.rewards.robot_forward_lean.params["std"] = 0.20


@configclass
class MinimalXRailFastLeanVelocityProgressDynamicWheelchairPushAttachedRobotPlayEnvCfg(
    MinimalXRailFastLeanVelocityProgressDynamicWheelchairPushAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10


@configclass
class MinimalXRailFastLeanVelocityProgressDynamicWheelchairPushAttachedPPORunnerCfg(
    MinimalVelocityDynamicWheelchairPushAttachedPPORunnerCfg
):
    experiment_name = "unitree_g1_29dof_wheelchair_minimal_x_rail_fast_lean_velocity_progress_push_attached"
    max_iterations = 1000
    save_interval = 50


@configclass
class MinimalPhysXRailFastLeanVelocityProgressDynamicWheelchairPushAttachedRobotEnvCfg(
    MinimalXRailFastLeanVelocityProgressDynamicWheelchairPushAttachedRobotEnvCfg
):
    """Playback diagnostic that replaces the kinematic X rail with a PhysX prismatic joint."""

    scene: PhysXRailNoCollisionDynamicWheelchairPushSceneCfg = PhysXRailNoCollisionDynamicWheelchairPushSceneCfg(
        num_envs=2048,
        env_spacing=4.0,
    )

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.spawn.rigid_props.max_linear_velocity = 20.0
        self.scene.robot.spawn.rigid_props.max_angular_velocity = 40.0
        env_step_dt = self.decimation * self.sim.dt
        soft_hand_attachment_params = {
            "robot_cfg": SceneEntityCfg("robot", body_names=DYNAMIC_WHEELCHAIR_HAND_BODY_NAMES),
            "wheelchair_cfg": SceneEntityCfg("wheelchair", body_names=DYNAMIC_WHEELCHAIR_HANDLE_BODY_NAMES),
            "robot_body_local_positions": HAND_GRIP_LOCAL_POSITIONS,
            "stiffness": 2500.0,
            "damping": 75.0,
            "max_force": 350.0,
        }
        self.events.attach_wheelchair_hands = None
        self.events.soft_attach_wheelchair_hands_reset = EventTerm(
            func=mdp.apply_soft_hand_handle_attachment,
            mode="reset",
            params=soft_hand_attachment_params,
        )
        self.events.soft_attach_wheelchair_hands = EventTerm(
            func=mdp.apply_soft_hand_handle_attachment,
            mode="interval",
            interval_range_s=(env_step_dt, env_step_dt),
            params=soft_hand_attachment_params,
        )
        self.events.constrain_wheelchair_to_forward_rail = None
        self.observations.policy.wheelchair_root_state.params["wheelchair_cfg"] = SceneEntityCfg(
            "wheelchair", body_names="base_link"
        )
        self.observations.critic.wheelchair_root_state.params["wheelchair_cfg"] = SceneEntityCfg(
            "wheelchair", body_names="base_link"
        )
        base_link_cfg = SceneEntityCfg("wheelchair", body_names="base_link")
        for reward_name in (
            "wheelchair_track_forward_velocity",
            "wheelchair_forward_progress",
            "wheelchair_backward_velocity",
            "wheelchair_lateral_velocity",
            "wheelchair_forward_line",
            "wheelchair_yaw_velocity",
            "wheelchair_root_heading",
            "wheelchair_forward_heading",
            "wheelchair_tilt",
        ):
            if hasattr(self.rewards, reward_name):
                getattr(self.rewards, reward_name).params["asset_cfg"] = base_link_cfg


@configclass
class MinimalPhysXRailFastLeanVelocityProgressDynamicWheelchairPushAttachedRobotPlayEnvCfg(
    MinimalPhysXRailFastLeanVelocityProgressDynamicWheelchairPushAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10


@configclass
class MinimalPhysXRailFastLeanVelocityProgressDynamicWheelchairPushAttachedPPORunnerCfg(
    MinimalXRailFastLeanVelocityProgressDynamicWheelchairPushAttachedPPORunnerCfg
):
    experiment_name = "unitree_g1_29dof_wheelchair_minimal_physx_rail_fast_lean_velocity_progress_push_attached"


@configclass
class MinimalPhysXRail1mpsYawTorqueDynamicWheelchairPushAttachedRobotEnvCfg(
    MinimalPhysXRailFastLeanVelocityProgressDynamicWheelchairPushAttachedRobotEnvCfg
):
    """1 m/s PhysX-rail push task with a yaw-torque penalty on the wheelchair rail joint."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (1.0, 1.0)

        self.rewards.wheelchair_track_forward_velocity.weight = 10.0
        self.rewards.wheelchair_track_forward_velocity.params["std"] = 0.4
        self.rewards.wheelchair_forward_progress.weight = 2.0
        self.rewards.wheelchair_forward_progress.params["max_velocity"] = 1.4
        self.rewards.wheelchair_backward_velocity.weight = -3.0
        self.rewards.robot_forward_lean.weight = 0.0
        self.rewards.wheelchair_rail_yaw_torque.weight = -0.05
        self.terminations.non_finite_wheelchair = DoneTerm(
            func=mdp.non_finite_asset_state,
            params={"asset_cfg": SceneEntityCfg("wheelchair", body_names="base_link")},
        )
        self.terminations.non_finite_robot = DoneTerm(
            func=mdp.non_finite_asset_state,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )


@configclass
class MinimalPhysXRail1mpsYawTorqueDynamicWheelchairPushAttachedRobotPlayEnvCfg(
    MinimalPhysXRail1mpsYawTorqueDynamicWheelchairPushAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10


@configclass
class MinimalPhysXRail1mpsYawTorqueDynamicWheelchairPushAttachedPPORunnerCfg(
    MinimalPhysXRailFastLeanVelocityProgressDynamicWheelchairPushAttachedPPORunnerCfg
):
    experiment_name = "unitree_g1_29dof_wheelchair_minimal_physx_rail_1mps_yaw_torque_push_attached"
    max_iterations = 1000
    save_interval = 50
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.0,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=1.0e-6,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=None,
        max_grad_norm=0.02,
    )


@configclass
class MinimalPhysXRail1mpsYawTorqueSoftObsDynamicWheelchairPushAttachedRobotEnvCfg(
    MinimalPhysXRail1mpsYawTorqueDynamicWheelchairPushAttachedRobotEnvCfg
):
    """Current 1 m/s PhysX-rail task with soft attachment state exposed to the policy."""

    observations: DynamicWheelchairPushSoftAttachmentObservationsCfg = (
        DynamicWheelchairPushSoftAttachmentObservationsCfg()
    )


@configclass
class MinimalPhysXRail1mpsYawTorqueSoftObsDynamicWheelchairPushAttachedRobotPlayEnvCfg(
    MinimalPhysXRail1mpsYawTorqueSoftObsDynamicWheelchairPushAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10


@configclass
class MinimalPhysXRail1mpsYawTorqueSoftObsDynamicWheelchairPushAttachedPPORunnerCfg(
    MinimalPhysXRail1mpsYawTorqueDynamicWheelchairPushAttachedPPORunnerCfg
):
    experiment_name = "unitree_g1_29dof_wheelchair_minimal_physx_rail_1mps_yaw_torque_softobs_push_attached"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.001,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=None,
        max_grad_norm=0.1,
    )


@configclass
class MinimalPhysXRail1mpsYawTorqueSoftObsStiffDynamicWheelchairPushAttachedRobotEnvCfg(
    MinimalPhysXRail1mpsYawTorqueSoftObsDynamicWheelchairPushAttachedRobotEnvCfg
):
    """SoftObs task with a stiffer bounded hand-handle spring-damper."""

    def __post_init__(self):
        super().__post_init__()

        soft_attachment = {
            "stiffness": 5000.0,
            "damping": 150.0,
            "max_force": 500.0,
        }
        for event_name in ("soft_attach_wheelchair_hands_reset", "soft_attach_wheelchair_hands"):
            event = getattr(self.events, event_name, None)
            if event is not None:
                event.params.update(soft_attachment)

        obs_attachment = {
            **soft_attachment,
            "force_scale": 500.0,
        }
        self.observations.policy.wheelchair_soft_attachment_state.params.update(obs_attachment)
        self.observations.critic.wheelchair_soft_attachment_state.params.update(obs_attachment)


@configclass
class MinimalPhysXRail1mpsYawTorqueSoftObsStiffDynamicWheelchairPushAttachedRobotPlayEnvCfg(
    MinimalPhysXRail1mpsYawTorqueSoftObsStiffDynamicWheelchairPushAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10


@configclass
class MinimalPhysXRail1mpsYawTorqueSoftObsStiffDynamicWheelchairPushAttachedPPORunnerCfg(
    MinimalPhysXRail1mpsYawTorqueSoftObsDynamicWheelchairPushAttachedPPORunnerCfg
):
    experiment_name = "unitree_g1_29dof_wheelchair_minimal_physx_rail_1mps_yaw_torque_softobs_stiff_push_attached"


@configclass
class StraightDynamicWheelchairPushAttachedRobotEnvCfg(RelaxedDynamicWheelchairPushAttachedRobotEnvCfg):
    """Straight-line correction phase for the attached wheelchair push task."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.06, 0.16)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.04, 0.22)

        self.rewards.track_ang_vel_z.weight = 1.0
        self.rewards.track_ang_vel_z.params["std"] = 0.15
        self.rewards.base_angular_velocity.weight = -0.2

        self.rewards.wheelchair_track_forward_velocity.weight = 2.0
        self.rewards.wheelchair_track_forward_velocity.params["std"] = 0.12
        self.rewards.wheelchair_forward_progress.weight = 0.35
        self.rewards.wheelchair_forward_progress.params["max_velocity"] = 0.3
        self.rewards.wheelchair_lateral_velocity.weight = -8.0
        self.rewards.wheelchair_forward_line.weight = -20.0
        self.rewards.wheelchair_forward_line.params["allowed_error"] = 0.02
        self.rewards.wheelchair_yaw_velocity.weight = -10.0
        self.rewards.wheelchair_root_heading.weight = -15.0
        self.rewards.wheelchair_root_heading.params["allowed_error"] = 0.02
        self.rewards.wheelchair_invalid_contact.weight = -0.03


@configclass
class StraightDynamicWheelchairPushAttachedRobotPlayEnvCfg(StraightDynamicWheelchairPushAttachedRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.commands.base_velocity.ranges.lin_vel_x = (0.12, 0.12)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.12, 0.12)


@configclass
class StraightDynamicWheelchairPushAttachedPPORunnerCfg(RelaxedDynamicWheelchairPushAttachedPPORunnerCfg):
    experiment_name = "unitree_g1_29dof_wheelchair_relaxed_push_attached_straight"
    max_iterations = 2000


@configclass
class YawConstrainedDynamicWheelchairPushAttachedRobotEnvCfg(StraightDynamicWheelchairPushAttachedRobotEnvCfg):
    """Forward-push curriculum that prevents the wheelchair from yawing off-line."""

    def __post_init__(self):
        super().__post_init__()

        env_step_dt = self.decimation * self.sim.dt
        self.events.constrain_wheelchair_to_forward_rail = EventTerm(
            func=mdp.constrain_root_to_forward_rail,
            mode="interval",
            interval_range_s=(env_step_dt, env_step_dt),
            params={
                "asset_cfg": SceneEntityCfg("wheelchair"),
                "lateral_position": DYNAMIC_WHEELCHAIR_INIT_POS[1],
                "yaw": 0.0,
            },
        )

        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.18)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.06, 0.24)

        self.rewards.wheelchair_track_forward_velocity.weight = 4.0
        self.rewards.wheelchair_track_forward_velocity.params["std"] = 0.1
        self.rewards.wheelchair_forward_progress.weight = 1.0
        self.rewards.wheelchair_forward_progress.params["max_velocity"] = 0.35
        self.rewards.wheelchair_lateral_velocity.weight = -1.0
        self.rewards.wheelchair_forward_line.weight = -1.0
        self.rewards.wheelchair_yaw_velocity.weight = -1.0
        self.rewards.wheelchair_root_heading.weight = -1.0


@configclass
class YawConstrainedDynamicWheelchairPushAttachedRobotPlayEnvCfg(
    YawConstrainedDynamicWheelchairPushAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.commands.base_velocity.ranges.lin_vel_x = (0.14, 0.14)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.14, 0.14)


@configclass
class YawConstrainedDynamicWheelchairPushAttachedPPORunnerCfg(StraightDynamicWheelchairPushAttachedPPORunnerCfg):
    experiment_name = "unitree_g1_29dof_wheelchair_relaxed_push_attached_yaw_constrained"
    max_iterations = 1500


@configclass
class UprightConstrainedDynamicWheelchairPushAttachedRobotEnvCfg(YawConstrainedDynamicWheelchairPushAttachedRobotEnvCfg):
    """Forward-push curriculum that keeps the chair planted while allowing robot lean."""

    def __post_init__(self):
        super().__post_init__()

        self.events.constrain_wheelchair_to_forward_rail.params.update(
            {
                "root_height": DYNAMIC_WHEELCHAIR_INIT_POS[2],
                "roll": 0.0,
                "pitch": 0.0,
                "zero_vertical_velocity": True,
                "zero_roll_pitch_velocity": True,
            }
        )

        self.rewards.wheelchair_tilt.weight = -1.0
        self.rewards.wheelchair_wheel_ground_height.weight = -5.0

        self.terminations.base_height.params["minimum_height"] = 0.45
        self.terminations.bad_orientation.params["limit_angle"] = 0.9

        self.actions.JointPositionAction.scale = {
            ".*_hip_.*|.*_knee_joint|.*_ankle_.*": 0.18,
            "waist_.*": 0.25,
            ".*_shoulder_.*|.*_elbow_joint": 0.08,
            ".*_wrist_.*": 0.015,
        }

        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.base_height.weight = -4.0
        self.rewards.base_height.params["target_height"] = 0.72
        self.rewards.base_linear_velocity.weight = -0.25
        self.rewards.base_angular_velocity.weight = -0.03
        self.rewards.joint_deviation_arms.weight = -0.01
        self.rewards.joint_deviation_waists.weight = -0.25


@configclass
class UprightConstrainedDynamicWheelchairPushAttachedRobotPlayEnvCfg(
    UprightConstrainedDynamicWheelchairPushAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.commands.base_velocity.ranges.lin_vel_x = (0.14, 0.14)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.14, 0.14)


@configclass
class UprightConstrainedDynamicWheelchairPushAttachedPPORunnerCfg(YawConstrainedDynamicWheelchairPushAttachedPPORunnerCfg):
    experiment_name = "unitree_g1_29dof_wheelchair_relaxed_push_attached_upright_constrained"
    max_iterations = 1000


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

    wheelchair_root_position = RewTerm(
        func=mdp.root_xy_position_l2,
        weight=0.0,
        params={
            "target_xy": [DYNAMIC_WHEELCHAIR_INIT_POS[0], DYNAMIC_WHEELCHAIR_INIT_POS[1]],
            "allowed_error": 0.03,
            "asset_cfg": SceneEntityCfg("wheelchair"),
        },
    )

    wheelchair_root_height = RewTerm(
        func=mdp.root_height_l2,
        weight=0.0,
        params={
            "target_height": DYNAMIC_WHEELCHAIR_INIT_POS[2],
            "allowed_error": 0.01,
            "asset_cfg": SceneEntityCfg("wheelchair"),
        },
    )

    wheelchair_root_heading = RewTerm(
        func=mdp.root_heading_lateral_l2,
        weight=0.0,
        params={"allowed_error": 0.03, "asset_cfg": SceneEntityCfg("wheelchair")},
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


def _restore_neutral_arm_start_pose(env_cfg):
    """Restore the base G1 neutral arm reset pose after wheelchair configs set handle arms."""

    for joint_name in WHEELCHAIR_ARM_JOINT_POSE:
        env_cfg.scene.robot.init_state.joint_pos.pop(joint_name, None)

    env_cfg.scene.robot.init_state.joint_pos.update(G1_NEUTRAL_ARM_JOINT_POSE)


@configclass
class DynamicWheelchairStandingObservedRobotEnvCfg(DynamicWheelchairPushObservedRobotEnvCfg):
    """Standing-only pretrain with wheelchair observations but no hand-handle joint."""

    rewards: DynamicWheelchairStandingAttachedRewardsCfg = DynamicWheelchairStandingAttachedRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_dynamic_wheelchair_standing_pretrain(self)


@configclass
class DynamicWheelchairStandingObservedNeutralRobotEnvCfg(DynamicWheelchairStandingObservedRobotEnvCfg):
    """Wheelchair-observed standing bridge that keeps the base neutral arm start pose."""

    def __post_init__(self):
        super().__post_init__()
        _restore_neutral_arm_start_pose(self)

        self.actions.JointPositionAction.scale = {
            ".*_hip_.*|waist_.*|.*_knee_joint|.*_ankle_.*": 0.18,
            ".*_shoulder_.*|.*_elbow_joint|.*_wrist_.*": 0.0,
        }
        self.rewards.joint_deviation_legs.weight = -1.0
        self.rewards.wheelchair_handle_contact.weight = 0.0
        self.rewards.wheelchair_invalid_contact.weight = 0.0
        self.rewards.wrist_joint_deviation.weight = 0.0


@configclass
class DynamicWheelchairStandingObservedNeutralRobotPlayEnvCfg(DynamicWheelchairStandingObservedNeutralRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10


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
class DynamicWheelchairStandingObservedNeutralPPORunnerCfg(DynamicWheelchairStandingObservedPPORunnerCfg):
    experiment_name = "unitree_g1_29dof_wheelchair_dynamic_stand_observed_neutral"


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
class FixedBaseDynamicWheelchairStandingAttachedRobotEnvCfg(DynamicWheelchairStandingAttachedRobotEnvCfg):
    """First holding primitive: hands attached to handles while the wheelchair base is fixed."""

    scene: FixedBaseDynamicWheelchairPushSceneCfg = FixedBaseDynamicWheelchairPushSceneCfg(
        num_envs=2048, env_spacing=4.0
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.wheelchair_track_forward_velocity.weight = 0.0
        self.rewards.wheelchair_forward_progress.weight = 0.0
        self.rewards.wheelchair_lateral_velocity.weight = 0.0
        self.rewards.wheelchair_forward_line.weight = 0.0
        self.rewards.wheelchair_yaw_velocity.weight = 0.0
        self.rewards.wheelchair_tilt.weight = 0.0
        self.rewards.wheelchair_wheel_ground_height.weight = 0.0
        self.rewards.wheelchair_xy_velocity.weight = 0.0


@configclass
class StationaryDynamicWheelchairStandingAttachedRobotEnvCfg(DynamicWheelchairStandingAttachedRobotEnvCfg):
    """Hands attached to a free wheelchair, biased to keep the chair stationary."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.wheelchair_track_forward_velocity.weight = 5.0
        self.rewards.wheelchair_track_forward_velocity.params["std"] = 0.05
        self.rewards.wheelchair_forward_progress.weight = 0.0
        self.rewards.wheelchair_lateral_velocity.weight = -12.0
        self.rewards.wheelchair_forward_line.weight = -10.0
        self.rewards.wheelchair_yaw_velocity.weight = -6.0
        self.rewards.wheelchair_tilt.weight = -20.0
        self.rewards.wheelchair_wheel_ground_height.weight = -120.0
        self.rewards.wheelchair_xy_velocity.weight = -12.0
        self.rewards.wheelchair_root_position.weight = -60.0
        self.rewards.wheelchair_root_height.weight = -40.0
        self.rewards.wheelchair_root_heading.weight = -20.0


@configclass
class BrakedStationaryDynamicWheelchairStandingAttachedRobotEnvCfg(
    StationaryDynamicWheelchairStandingAttachedRobotEnvCfg
):
    """Hands attached to a braked, free wheelchair with stationary rewards."""

    scene: BrakedDynamicWheelchairPushSceneCfg = BrakedDynamicWheelchairPushSceneCfg(num_envs=2048, env_spacing=4.0)


@configclass
class RelaxedBrakedDynamicWheelchairStandingAttachedRobotEnvCfg(DynamicWheelchairStandingAttachedRobotEnvCfg):
    """Hands attached to a braked free wheelchair with relaxed first-stage balance rewards."""

    scene: BrakedDynamicWheelchairPushSceneCfg = BrakedDynamicWheelchairPushSceneCfg(num_envs=2048, env_spacing=4.0)

    def __post_init__(self):
        super().__post_init__()

        self.actions.JointPositionAction.scale = {
            ".*_hip_.*|waist_.*|.*_knee_joint|.*_ankle_.*": 0.12,
            ".*_shoulder_.*|.*_elbow_joint": 0.05,
            ".*_wrist_.*": 0.03,
        }

        self.rewards.base_linear_velocity.weight = -1.0
        self.rewards.base_angular_velocity.weight = -0.1
        self.rewards.joint_deviation_arms.weight = -0.02
        self.rewards.joint_deviation_legs.weight = -1.0
        self.rewards.robot_xy_velocity.weight = -0.8
        self.rewards.robot_yaw_velocity.weight = -0.2

        self.rewards.wheelchair_track_forward_velocity.weight = 0.5
        self.rewards.wheelchair_track_forward_velocity.params["std"] = 0.2
        self.rewards.wheelchair_forward_progress.weight = 0.0
        self.rewards.wheelchair_lateral_velocity.weight = -0.5
        self.rewards.wheelchair_forward_line.weight = 0.0
        self.rewards.wheelchair_yaw_velocity.weight = -0.3
        self.rewards.wheelchair_tilt.weight = -2.0
        self.rewards.wheelchair_wheel_ground_height.weight = 0.0
        self.rewards.wheelchair_xy_velocity.weight = -0.5
        self.rewards.wheelchair_root_position.weight = 0.0
        self.rewards.wheelchair_root_height.weight = 0.0
        self.rewards.wheelchair_root_heading.weight = 0.0
        self.rewards.wheelchair_invalid_contact.weight = -0.01
        self.rewards.wrist_joint_deviation.weight = -0.05


@configclass
class LeftHandRelaxedBrakedDynamicWheelchairStandingAttachedRobotEnvCfg(
    RelaxedBrakedDynamicWheelchairStandingAttachedRobotEnvCfg
):
    """Diagnostic first stage with only the left hand attached to avoid a two-arm closed chain."""

    def __post_init__(self):
        super().__post_init__()
        self.events.attach_wheelchair_hands.params["attachments"] = [DYNAMIC_WHEELCHAIR_HAND_HANDLE_ATTACHMENTS[0]]


@configclass
class FixedBaseRelaxedWheelchairStandingAttachedRobotEnvCfg(
    RelaxedBrakedDynamicWheelchairStandingAttachedRobotEnvCfg
):
    """Hands attached to handles with relaxed arm control while the wheelchair root is fixed."""

    scene: FixedBaseDynamicWheelchairPushSceneCfg = FixedBaseDynamicWheelchairPushSceneCfg(
        num_envs=2048, env_spacing=4.0
    )

    def __post_init__(self):
        super().__post_init__()
        self.events.attach_wheelchair_hands.params["mask_collisions"] = True
        self.actions.JointPositionAction.scale[".*_wrist_.*"] = 0.015
        self.rewards.wheelchair_track_forward_velocity.weight = 0.0
        self.rewards.wheelchair_lateral_velocity.weight = 0.0
        self.rewards.wheelchair_yaw_velocity.weight = 0.0
        self.rewards.wheelchair_tilt.weight = 0.0
        self.rewards.wheelchair_xy_velocity.weight = 0.0
        self.rewards.wheelchair_invalid_contact.weight = 0.0
        self.rewards.wrist_joint_deviation.weight = -0.25


@configclass
class BrakedStationaryDynamicWheelchairStandingAttachedRobotPlayEnvCfg(
    BrakedStationaryDynamicWheelchairStandingAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10


@configclass
class RelaxedBrakedDynamicWheelchairStandingAttachedRobotPlayEnvCfg(
    RelaxedBrakedDynamicWheelchairStandingAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10


@configclass
class LeftHandRelaxedBrakedDynamicWheelchairStandingAttachedRobotPlayEnvCfg(
    LeftHandRelaxedBrakedDynamicWheelchairStandingAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10


@configclass
class FixedBaseRelaxedWheelchairStandingAttachedRobotPlayEnvCfg(
    FixedBaseRelaxedWheelchairStandingAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10


@configclass
class StationaryDynamicWheelchairStandingAttachedRobotPlayEnvCfg(
    StationaryDynamicWheelchairStandingAttachedRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10


@configclass
class FixedBaseDynamicWheelchairStandingAttachedRobotPlayEnvCfg(FixedBaseDynamicWheelchairStandingAttachedRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10


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


@configclass
class FixedBaseDynamicWheelchairStandingAttachedPPORunnerCfg(DynamicWheelchairStandingAttachedPPORunnerCfg):
    experiment_name = "unitree_g1_29dof_wheelchair_fixed_stand_attached"


@configclass
class StationaryDynamicWheelchairStandingAttachedPPORunnerCfg(DynamicWheelchairStandingAttachedPPORunnerCfg):
    experiment_name = "unitree_g1_29dof_wheelchair_stationary_stand_attached"


@configclass
class BrakedStationaryDynamicWheelchairStandingAttachedPPORunnerCfg(
    DynamicWheelchairStandingAttachedPPORunnerCfg
):
    experiment_name = "unitree_g1_29dof_wheelchair_braked_stationary_stand_attached"


@configclass
class RelaxedBrakedDynamicWheelchairStandingAttachedPPORunnerCfg(
    DynamicWheelchairStandingAttachedPPORunnerCfg
):
    experiment_name = "unitree_g1_29dof_wheelchair_relaxed_stand_attached"


@configclass
class FixedBaseRelaxedWheelchairStandingAttachedPPORunnerCfg(
    DynamicWheelchairStandingAttachedPPORunnerCfg
):
    experiment_name = "unitree_g1_29dof_wheelchair_fixed_relaxed_stand_attached"


@configclass
class LeftHandRelaxedBrakedDynamicWheelchairStandingAttachedPPORunnerCfg(
    DynamicWheelchairStandingAttachedPPORunnerCfg
):
    experiment_name = "unitree_g1_29dof_wheelchair_left_hand_relaxed_stand_attached"
