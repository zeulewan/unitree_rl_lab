from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def stable_lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    reward_success_ratio: float = 0.8,
    max_failure_rate: float = 0.25,
    min_episode_length_ratio: float = 0.75,
    command_step: float = 0.1,
    expand_lower: bool = True,
    expand_lateral: bool = True,
    failure_term_names: Sequence[str] = ("bad_orientation", "base_height"),
) -> torch.Tensor:
    """Increase forward command range only when tracking and reset health are good."""
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    failure_rate = _recent_failure_rate(env, failure_term_names)
    episode_length_ratio = _episode_length_ratio(env, env_ids)

    if env.common_step_counter % env.max_episode_length == 0:
        should_expand = (
            reward > reward_term.weight * reward_success_ratio
            and failure_rate <= max_failure_rate
            and episode_length_ratio >= min_episode_length_ratio
        )
        if should_expand:
            lower_delta = -command_step if expand_lower else 0.0
            delta_command = torch.tensor([lower_delta, command_step], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()

            if expand_lateral:
                ranges.lin_vel_y = torch.clamp(
                    torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                    limit_ranges.lin_vel_y[0],
                    limit_ranges.lin_vel_y[1],
                ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def lin_vel_cmd_stability(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    failure_term_names: Sequence[str] = ("bad_orientation", "base_height"),
) -> dict[str, torch.Tensor]:
    """Log the quantities used by stable_lin_vel_cmd_levels."""
    command_term = env.command_manager.get_term("base_velocity")
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    return {
        "range_max": torch.tensor(command_term.cfg.ranges.lin_vel_x[1], device=env.device),
        "track_ratio": reward / reward_term.weight,
        "failure_rate": _recent_failure_rate(env, failure_term_names),
        "episode_length_ratio": _episode_length_ratio(env, env_ids),
    }


def _recent_failure_rate(env: ManagerBasedRLEnv, failure_term_names: Sequence[str]) -> torch.Tensor:
    """Return fraction of envs whose latest episode ended from a fall-like termination."""
    termination_manager = env.termination_manager
    last_episode_dones = getattr(termination_manager, "_last_episode_dones", None)
    term_name_to_idx = getattr(termination_manager, "_term_name_to_term_idx", {})

    if last_episode_dones is None or not term_name_to_idx:
        return torch.mean(termination_manager.terminated.float())

    selected_terms = []
    for term_name in failure_term_names:
        term_idx = term_name_to_idx.get(term_name)
        if term_idx is not None:
            selected_terms.append(last_episode_dones[:, term_idx])

    if not selected_terms:
        return torch.mean(termination_manager.terminated.float())

    return torch.stack(selected_terms, dim=1).any(dim=1).float().mean()


def _episode_length_ratio(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    if isinstance(env_ids, slice):
        episode_lengths = env.episode_length_buf[env_ids]
    elif len(env_ids) == 0:
        return torch.tensor(1.0, device=env.device)
    else:
        episode_lengths = env.episode_length_buf[env_ids]

    return torch.clamp(torch.mean(episode_lengths.float()) / env.max_episode_length, 0.0, 1.0)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
