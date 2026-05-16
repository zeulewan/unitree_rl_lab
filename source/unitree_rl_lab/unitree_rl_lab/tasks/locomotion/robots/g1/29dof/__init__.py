import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Running",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.running_env_cfg:RunningRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.running_env_cfg:RunningRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Stand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standing_env_cfg:StandingRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.standing_env_cfg:StandingRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.standing_env_cfg:StandingPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Stand-Handle-Arms",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standing_env_cfg:StandingHandleArmsRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.standing_env_cfg:StandingHandleArmsRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.standing_env_cfg:StandingHandleArmsPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Stand-Reach-Arms",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standing_env_cfg:StandingReachArmsRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.standing_env_cfg:StandingReachArmsRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.standing_env_cfg:StandingReachArmsPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Running-Fast",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fast_running_env_cfg:FastRunningRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.fast_running_env_cfg:FastRunningRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Sprint-10ms",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sprint_10ms_env_cfg:Sprint10msRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.sprint_10ms_env_cfg:Sprint10msRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Sprint-10ms-Curriculum-Resume",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sprint_10ms_env_cfg:Sprint10msCurriculumResumeEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.sprint_10ms_env_cfg:Sprint10msCurriculumResumePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.sprint_10ms_env_cfg:Sprint10msCurriculumResumePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Sprint-10ms-Gait",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sprint_10ms_env_cfg:Sprint10msGaitRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.sprint_10ms_env_cfg:Sprint10msGaitRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.sprint_10ms_env_cfg:Sprint10msGaitPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Wheelchair-Push",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:WheelchairPushRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:WheelchairPushRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:WheelchairPushPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Wheelchair-Dynamic-Push",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairPushRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairPushRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairPushPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Wheelchair-Dynamic-Push-Observed",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairPushObservedRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairPushObservedRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairPushObservedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Wheelchair-Dynamic-Push-Attached",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairPushAttachedRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairPushAttachedRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairPushAttachedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Wheelchair-Relaxed-Push-Attached",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:RelaxedDynamicWheelchairPushAttachedRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:RelaxedDynamicWheelchairPushAttachedRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:RelaxedDynamicWheelchairPushAttachedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Wheelchair-Dynamic-Stand-Attached",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairStandingAttachedRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairStandingAttachedRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairStandingAttachedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Wheelchair-Fixed-Stand-Attached",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:FixedBaseDynamicWheelchairStandingAttachedRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:FixedBaseDynamicWheelchairStandingAttachedRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:FixedBaseDynamicWheelchairStandingAttachedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Wheelchair-Fixed-Relaxed-Stand-Attached",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:FixedBaseRelaxedWheelchairStandingAttachedRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:FixedBaseRelaxedWheelchairStandingAttachedRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:FixedBaseRelaxedWheelchairStandingAttachedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Wheelchair-Stationary-Stand-Attached",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:StationaryDynamicWheelchairStandingAttachedRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:StationaryDynamicWheelchairStandingAttachedRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:StationaryDynamicWheelchairStandingAttachedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Wheelchair-Braked-Stationary-Stand-Attached",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:BrakedStationaryDynamicWheelchairStandingAttachedRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:BrakedStationaryDynamicWheelchairStandingAttachedRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:BrakedStationaryDynamicWheelchairStandingAttachedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Wheelchair-Relaxed-Stand-Attached",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:RelaxedBrakedDynamicWheelchairStandingAttachedRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:RelaxedBrakedDynamicWheelchairStandingAttachedRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:RelaxedBrakedDynamicWheelchairStandingAttachedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Wheelchair-Left-Hand-Relaxed-Stand-Attached",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:LeftHandRelaxedBrakedDynamicWheelchairStandingAttachedRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:LeftHandRelaxedBrakedDynamicWheelchairStandingAttachedRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:LeftHandRelaxedBrakedDynamicWheelchairStandingAttachedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Wheelchair-Dynamic-Stand-Observed",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairStandingObservedRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairStandingObservedRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairStandingObservedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Wheelchair-Dynamic-Stand-Observed-Neutral",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairStandingObservedNeutralRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairStandingObservedNeutralRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.wheelchair_push_env_cfg:DynamicWheelchairStandingObservedNeutralPPORunnerCfg",
    },
)
