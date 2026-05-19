from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate as quat_apply
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _finite_tensor(value: torch.Tensor, replacement: float = 0.0) -> torch.Tensor:
    return torch.nan_to_num(value, nan=replacement, posinf=replacement, neginf=replacement)


def _finite_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = _finite_tensor(quat)
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    return torch.where(norm > 1.0e-6, quat / norm.clamp_min(1.0e-6), identity)


def _root_or_selected_body_state(
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return root state unless the cfg selects a specific body."""

    if isinstance(asset_cfg.body_ids, slice):
        return (
            _finite_tensor(asset.data.root_pos_w),
            _finite_quat(asset.data.root_quat_w),
            _finite_tensor(asset.data.root_lin_vel_w),
            _finite_tensor(asset.data.root_ang_vel_w),
        )

    body_id = asset_cfg.body_ids[0] if isinstance(asset_cfg.body_ids, list) else int(asset_cfg.body_ids)
    return (
        _finite_tensor(asset.data.body_pos_w[:, body_id, :]),
        _finite_quat(asset.data.body_quat_w[:, body_id, :]),
        _finite_tensor(asset.data.body_lin_vel_w[:, body_id, :]),
        _finite_tensor(asset.data.body_ang_vel_w[:, body_id, :]),
    )


"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


def joint_position_l1(
    env: ManagerBasedRLEnv,
    target: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize selected joints away from an explicit target position."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    target_tensor = torch.as_tensor(target, dtype=joint_pos.dtype, device=joint_pos.device)
    return torch.sum(torch.abs(joint_pos - target_tensor), dim=1)


def _body_points_w(
    asset: Articulation,
    body_ids: list[int],
    local_positions: list[list[float]] | None = None,
) -> torch.Tensor:
    body_pos_w = _finite_tensor(asset.data.body_pos_w[:, body_ids, :])
    if local_positions is None:
        return body_pos_w

    local_pos = torch.tensor(local_positions, device=body_pos_w.device, dtype=body_pos_w.dtype).unsqueeze(0)
    local_pos = local_pos.expand(body_pos_w.shape[0], -1, -1)
    body_quat_w = _finite_quat(asset.data.body_quat_w[:, body_ids, :])
    offsets_w = quat_apply(body_quat_w.reshape(-1, 4), local_pos.reshape(-1, 3)).reshape_as(body_pos_w)
    return body_pos_w + offsets_w


def _body_positions_in_root_frame(
    asset: Articulation,
    body_ids: list[int],
    local_positions: list[list[float]] | None = None,
) -> torch.Tensor:
    body_pos_w = _body_points_w(asset, body_ids, local_positions)
    body_pos_translated = body_pos_w - asset.data.root_pos_w[:, None, :]
    body_pos_b = torch.zeros_like(body_pos_translated)
    for body_index in range(body_pos_translated.shape[1]):
        body_pos_b[:, body_index, :] = quat_apply_inverse(asset.data.root_quat_w, body_pos_translated[:, body_index, :])
    return body_pos_b


def _body_axis_in_world(asset: Articulation, body_ids: list[int], axis: list[float]) -> torch.Tensor:
    body_quat_w = _finite_quat(asset.data.body_quat_w[:, body_ids, :])
    axis_b = torch.tensor(axis, device=body_quat_w.device, dtype=body_quat_w.dtype)
    axis_b = axis_b.expand(body_quat_w.shape[0], body_quat_w.shape[1], 3)
    axis_w = quat_apply(body_quat_w.reshape(-1, 4), axis_b.reshape(-1, 3))
    return axis_w.reshape(body_quat_w.shape[0], body_quat_w.shape[1], 3)


def hand_handle_position_error_exp(
    env: ManagerBasedRLEnv,
    target_positions_b: list[list[float]],
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_local_positions: list[list[float]] | None = None,
) -> torch.Tensor:
    """Reward keeping selected hand bodies close to fixed wheelchair-handle targets.

    The target positions are expressed in the robot root frame. This keeps the observation and
    action spaces identical to the base locomotion policy, so this task can be warm-started from
    the walking checkpoint while shifting the default arm pose into a handle-grip posture.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_b = _body_positions_in_root_frame(asset, asset_cfg.body_ids, body_local_positions)
    target_pos_b = torch.tensor(target_positions_b, device=env.device, dtype=body_pos_b.dtype).unsqueeze(0)
    position_error = torch.mean(torch.sum(torch.square(body_pos_b - target_pos_b), dim=-1), dim=-1)
    return torch.exp(-position_error / (std * std))


def hand_handle_position_error_l2(
    env: ManagerBasedRLEnv,
    target_positions_b: list[list[float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_local_positions: list[list[float]] | None = None,
) -> torch.Tensor:
    """Penalize distance between selected hand bodies and fixed handle targets."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_b = _body_positions_in_root_frame(asset, asset_cfg.body_ids, body_local_positions)
    target_pos_b = torch.tensor(target_positions_b, device=env.device, dtype=body_pos_b.dtype).unsqueeze(0)
    return torch.mean(torch.sum(torch.square(body_pos_b - target_pos_b), dim=-1), dim=-1)


def dynamic_hand_handle_position_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheelchair_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
    robot_body_local_positions: list[list[float]] | None = None,
    wheelchair_body_local_positions: list[list[float]] | None = None,
) -> torch.Tensor:
    """Reward selected robot hand bodies for staying close to moving wheelchair handle bodies."""
    robot: Articulation = env.scene[robot_cfg.name]
    wheelchair: Articulation = env.scene[wheelchair_cfg.name]
    hand_pos_w = _body_points_w(robot, robot_cfg.body_ids, robot_body_local_positions)
    handle_pos_w = _body_points_w(wheelchair, wheelchair_cfg.body_ids, wheelchair_body_local_positions)
    position_error = torch.mean(torch.sum(torch.square(hand_pos_w - handle_pos_w), dim=-1), dim=-1)
    return torch.exp(-position_error / (std * std))


def dynamic_hand_handle_position_error_l2(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheelchair_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
    robot_body_local_positions: list[list[float]] | None = None,
    wheelchair_body_local_positions: list[list[float]] | None = None,
) -> torch.Tensor:
    """Penalize distance between selected robot hand bodies and moving wheelchair handles."""
    robot: Articulation = env.scene[robot_cfg.name]
    wheelchair: Articulation = env.scene[wheelchair_cfg.name]
    hand_pos_w = _body_points_w(robot, robot_cfg.body_ids, robot_body_local_positions)
    handle_pos_w = _body_points_w(wheelchair, wheelchair_cfg.body_ids, wheelchair_body_local_positions)
    return torch.mean(torch.sum(torch.square(hand_pos_w - handle_pos_w), dim=-1), dim=-1)


def dynamic_hand_handle_axis_alignment_l2(
    env: ManagerBasedRLEnv,
    axis: list[float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheelchair_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
) -> torch.Tensor:
    """Penalize selected hand bodies being perpendicular to wheelchair handle frames."""
    robot: Articulation = env.scene[robot_cfg.name]
    wheelchair: Articulation = env.scene[wheelchair_cfg.name]
    hand_axis_w = _body_axis_in_world(robot, robot_cfg.body_ids, axis)
    handle_axis_w = _body_axis_in_world(wheelchair, wheelchair_cfg.body_ids, axis)
    alignment = torch.sum(hand_axis_w * handle_axis_w, dim=-1).abs().clamp(max=1.0)
    return torch.mean(1.0 - alignment, dim=-1)


def wheelchair_forward_velocity_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
) -> torch.Tensor:
    """Reward the wheelchair root/body for matching the commanded forward velocity in world X."""
    wheelchair: Articulation = env.scene[asset_cfg.name]
    _, _, lin_vel_w, _ = _root_or_selected_body_state(wheelchair, asset_cfg)
    command_x = env.command_manager.get_command(command_name)[:, 0]
    velocity_error = torch.square(lin_vel_w[:, 0] - command_x)
    return torch.exp(-velocity_error / (std * std))


def wheelchair_forward_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
    max_velocity: float = 1.2,
) -> torch.Tensor:
    """Reward positive wheelchair forward velocity."""
    wheelchair: Articulation = env.scene[asset_cfg.name]
    _, _, lin_vel_w, _ = _root_or_selected_body_state(wheelchair, asset_cfg)
    return torch.clamp(lin_vel_w[:, 0], min=0.0, max=max_velocity)


def wheelchair_backward_velocity_l2(
    env: ManagerBasedRLEnv,
    max_velocity: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
) -> torch.Tensor:
    """Penalize the wheelchair root/body for moving backward along world X."""
    wheelchair: Articulation = env.scene[asset_cfg.name]
    _, _, lin_vel_w, _ = _root_or_selected_body_state(wheelchair, asset_cfg)
    backward_velocity = torch.clamp(-lin_vel_w[:, 0], min=0.0, max=max_velocity)
    return torch.square(backward_velocity)


def wheelchair_lateral_velocity_l2(
    env: ManagerBasedRLEnv,
    max_velocity: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
) -> torch.Tensor:
    """Penalize the wheelchair drifting sideways."""
    wheelchair: Articulation = env.scene[asset_cfg.name]
    _, _, lin_vel_w, _ = _root_or_selected_body_state(wheelchair, asset_cfg)
    lateral_velocity = lin_vel_w[:, 1]
    if max_velocity is not None:
        lateral_velocity = torch.clamp(lateral_velocity, min=-max_velocity, max=max_velocity)
    return torch.square(lateral_velocity)


def wheelchair_forward_line_l2(
    env: ManagerBasedRLEnv,
    allowed_error: float = 0.05,
    max_error: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
) -> torch.Tensor:
    """Penalize the wheelchair root/body drifting away from the environment forward centerline."""
    wheelchair: Articulation = env.scene[asset_cfg.name]
    pos_w, _, _, _ = _root_or_selected_body_state(wheelchair, asset_cfg)
    lateral_position = pos_w[:, 1] - env.scene.env_origins[:, 1]
    lateral_error = torch.clamp(torch.abs(lateral_position) - allowed_error, min=0.0)
    if max_error is not None:
        lateral_error = torch.clamp(lateral_error, max=max_error)
    return torch.square(lateral_error)


def root_xy_position_l2(
    env: ManagerBasedRLEnv,
    target_xy: list[float],
    allowed_error: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
) -> torch.Tensor:
    """Penalize an articulation root/body drifting from a target XY offset in each env."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos_w, _, _, _ = _root_or_selected_body_state(asset, asset_cfg)
    target_xy_tensor = torch.tensor(target_xy, device=env.device, dtype=pos_w.dtype).unsqueeze(0)
    target_xy_w = env.scene.env_origins[:, :2] + target_xy_tensor
    position_error = torch.linalg.norm(pos_w[:, :2] - target_xy_w, dim=-1)
    position_error = torch.clamp(position_error - allowed_error, min=0.0)
    return torch.square(position_error)


def root_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    allowed_error: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
) -> torch.Tensor:
    """Penalize an articulation root/body drifting from a target world Z height."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos_w, _, _, _ = _root_or_selected_body_state(asset, asset_cfg)
    height_error = torch.abs(pos_w[:, 2] - (env.scene.env_origins[:, 2] + target_height))
    height_error = torch.clamp(height_error - allowed_error, min=0.0)
    return torch.square(height_error)


def root_heading_lateral_l2(
    env: ManagerBasedRLEnv,
    allowed_error: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
) -> torch.Tensor:
    """Penalize yaw drift from the world-X heading using the root/body forward axis."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, quat_w, _, _ = _root_or_selected_body_state(asset, asset_cfg)
    x_axis_b = torch.tensor([1.0, 0.0, 0.0], device=env.device, dtype=quat_w.dtype).expand(env.num_envs, 3)
    x_axis_w = quat_apply(quat_w, x_axis_b)
    heading_error = torch.clamp(torch.abs(x_axis_w[:, 1]) - allowed_error, min=0.0)
    return torch.square(heading_error)


def root_forward_heading_l2(
    env: ManagerBasedRLEnv,
    allowed_error: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
) -> torch.Tensor:
    """Penalize an articulation root/body facing away from the world-X forward direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, quat_w, _, _ = _root_or_selected_body_state(asset, asset_cfg)
    x_axis_b = torch.tensor([1.0, 0.0, 0.0], device=env.device, dtype=quat_w.dtype).expand(env.num_envs, 3)
    x_axis_w = quat_apply(quat_w, x_axis_b)
    heading_error = torch.clamp((1.0 - x_axis_w[:, 0]) - allowed_error, min=0.0)
    return torch.square(heading_error)


def wheelchair_yaw_velocity_l2(
    env: ManagerBasedRLEnv,
    max_angular_velocity: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
) -> torch.Tensor:
    """Penalize the wheelchair spinning while being pushed."""
    wheelchair: Articulation = env.scene[asset_cfg.name]
    _, _, _, ang_vel_w = _root_or_selected_body_state(wheelchair, asset_cfg)
    yaw_velocity = ang_vel_w[:, 2]
    if max_angular_velocity is not None:
        yaw_velocity = torch.clamp(yaw_velocity, min=-max_angular_velocity, max=max_angular_velocity)
    return torch.square(yaw_velocity)


def body_incoming_joint_torque_axis_l2(
    env: ManagerBasedRLEnv,
    axis: str = "z",
    scale: float = 100.0,
    max_abs_torque: float = 250.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
) -> torch.Tensor:
    """Penalize a selected body's incoming joint torque on one local axis."""
    asset: Articulation = env.scene[asset_cfg.name]
    axis_id = {"x": 0, "y": 1, "z": 2}[axis]
    body_id = asset_cfg.body_ids[0] if isinstance(asset_cfg.body_ids, list) else int(asset_cfg.body_ids)
    torque = _finite_tensor(asset.data.body_incoming_joint_wrench_b[:, body_id, 3 + axis_id])
    torque_abs = torch.clamp(torch.abs(torque), max=max_abs_torque)
    return torch.square(torque_abs / scale)


def wheelchair_tilt_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
) -> torch.Tensor:
    """Penalize wheelchair roll/pitch by using projected gravity in the chair body frame."""
    wheelchair: Articulation = env.scene[asset_cfg.name]
    if isinstance(asset_cfg.body_ids, slice):
        projected_gravity_b = wheelchair.data.projected_gravity_b
    else:
        _, quat_w, _, _ = _root_or_selected_body_state(wheelchair, asset_cfg)
        projected_gravity_b = quat_apply_inverse(quat_w, wheelchair.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=-1)


def wheelchair_wheel_height_l2(
    env: ManagerBasedRLEnv,
    target_heights: list[float],
    allowed_error: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wheelchair"),
) -> torch.Tensor:
    """Penalize wheelchair wheel/caster body heights drifting from their ground-contact height."""
    wheelchair: Articulation = env.scene[asset_cfg.name]
    wheel_heights = wheelchair.data.body_pos_w[:, asset_cfg.body_ids, 2]
    target_height_tensor = torch.tensor(target_heights, device=env.device, dtype=wheel_heights.dtype).unsqueeze(0)
    height_error = torch.clamp(torch.abs(wheel_heights - target_height_tensor) - allowed_error, min=0.0)
    return torch.mean(torch.square(height_error), dim=-1)


def root_lin_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize root XY linear velocity for any scene articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, lin_vel_w, _ = _root_or_selected_body_state(asset, asset_cfg)
    return torch.sum(torch.square(lin_vel_w[:, :2]), dim=-1)


def root_ang_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize root yaw angular velocity for any scene articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, _, ang_vel_w = _root_or_selected_body_state(asset, asset_cfg)
    return torch.square(ang_vel_w[:, 2])


def filtered_contact_presence(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward whether each filtered contact sensor sees contact above a force threshold."""
    reward = torch.zeros(env.num_envs, device=env.device)
    for sensor_name in sensor_names:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
        force_matrix = contact_sensor.data.force_matrix_w
        contact_force = torch.linalg.norm(force_matrix, dim=-1)
        reward += torch.any(contact_force > threshold, dim=(1, 2)).float()
    return reward / max(len(sensor_names), 1)


def filtered_contact_force_penalty(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize filtered contact forces above a threshold across multiple sensors."""
    penalty = torch.zeros(env.num_envs, device=env.device)
    for sensor_name in sensor_names:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
        force_matrix = contact_sensor.data.force_matrix_w
        contact_force = torch.linalg.norm(force_matrix, dim=-1)
        penalty += torch.sum(torch.clamp(contact_force - threshold, min=0.0), dim=(1, 2))
    return penalty


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def root_forward_lean_exp(
    env: ManagerBasedRLEnv,
    target: float = 0.17,
    std: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the root's local up axis leaning toward positive world X."""
    asset: RigidObject = env.scene[asset_cfg.name]
    root_quat_w = _finite_quat(asset.data.root_quat_w)
    local_up = torch.tensor([0.0, 0.0, 1.0], dtype=root_quat_w.dtype, device=root_quat_w.device)
    local_up = local_up.expand(root_quat_w.shape[0], -1)
    root_up_w = quat_apply(root_quat_w, local_up)
    lean_error = torch.square(root_up_w[:, 0] - target)
    return torch.exp(-lean_error / (std * std))


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
