from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import RewardsCfg, RobotEnvCfg


@configclass
class StandingRewardsCfg(RewardsCfg):
    """Reward terms for a zero-command standing pretrain."""

    fall_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-200.0,
        params={"term_keys": ["bad_orientation", "base_height"]},
    )

    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.6,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
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


@configclass
class StandingRobotEnvCfg(RobotEnvCfg):
    """G1 zero-velocity standing stage before wheelchair-specific training."""

    rewards: StandingRewardsCfg = StandingRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum = None
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.base_height.params["minimum_height"] = 0.55
        self.terminations.bad_orientation.params["limit_angle"] = 0.55

        self.actions.JointPositionAction.scale = {
            ".*_hip_.*|waist_.*|.*_knee_joint|.*_ankle_.*": 0.18,
            ".*_shoulder_.*|.*_elbow_joint|.*_wrist_.*": 0.0,
        }

        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (0.0, 0.0)

        self.rewards.track_lin_vel_xy.weight = 2.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.25
        self.rewards.track_ang_vel_z.weight = 1.0
        self.rewards.track_ang_vel_z.params["std"] = 0.25
        self.rewards.alive.weight = 1.0
        self.rewards.base_linear_velocity.weight = -2.5
        self.rewards.base_angular_velocity.weight = -0.2
        self.rewards.joint_vel.weight = -0.002
        self.rewards.joint_acc.weight = -1.0e-6
        self.rewards.action_rate.weight = 0.0
        self.rewards.joint_deviation_waists.weight = -1.0
        self.rewards.joint_deviation_legs.weight = -1.0
        self.rewards.flat_orientation_l2.weight = -20.0
        self.rewards.base_height.weight = -30.0
        self.rewards.base_height.params["target_height"] = 0.78
        self.rewards.gait.weight = 0.0
        self.rewards.feet_clearance.weight = 0.0
        self.rewards.feet_slide.weight = -0.3


@configclass
class StandingRobotPlayEnvCfg(StandingRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10


@configclass
class StandingPPORunnerCfg(BasePPORunnerCfg):
    num_steps_per_env = 48
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_g1_29dof_stand"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.02,
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
