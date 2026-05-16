"""Expand an RSL-RL actor/critic checkpoint for added observation terms."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _expand_weight(model_state_dict: dict[str, torch.Tensor], key: str, target_input_dim: int) -> None:
    weight = model_state_dict[key]
    if weight.ndim != 2:
        raise ValueError(f"{key} must be a matrix, got shape {tuple(weight.shape)}")

    current_input_dim = weight.shape[1]
    if current_input_dim == target_input_dim:
        print(f"{key}: already {current_input_dim} inputs")
        return

    expanded = weight.new_zeros((weight.shape[0], target_input_dim))
    copied_input_dim = min(current_input_dim, target_input_dim)
    expanded[:, :copied_input_dim] = weight[:, :copied_input_dim]
    model_state_dict[key] = expanded
    print(f"{key}: {current_input_dim} -> {target_input_dim} inputs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_checkpoint", type=Path)
    parser.add_argument("output_checkpoint", type=Path)
    parser.add_argument("--actor-input-dim", type=int, required=True)
    parser.add_argument("--critic-input-dim", type=int, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    model_state_dict = checkpoint["model_state_dict"]
    _expand_weight(model_state_dict, "actor.0.weight", args.actor_input_dim)
    _expand_weight(model_state_dict, "critic.0.weight", args.critic_input_dim)

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_checkpoint)
    print(f"wrote {args.output_checkpoint}")


if __name__ == "__main__":
    main()
