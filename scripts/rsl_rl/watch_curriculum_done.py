#!/usr/bin/env python3
"""Stop a tmux training session after the sprint velocity curriculum is complete.

The RSL-RL checkpoint does not persist the active command curriculum range, so the
continuation task starts from the latest observed range and this watcher stops
training only after the log reaches the target curriculum value and a checkpoint
at or after that iteration exists.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path


def tmux_alive(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def latest_run_dir(root: Path, run_name: str) -> Path | None:
    matches = [path for path in root.glob(f"*_{run_name}") if path.is_dir()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def latest_model_iter(run_dir: Path) -> int | None:
    values: list[int] = []
    for path in run_dir.glob("model_*.pt"):
        match = re.match(r"model_(\d+)\.pt$", path.name)
        if match:
            values.append(int(match.group(1)))
    return max(values) if values else None


def curriculum_done_iter(train_log: Path, target: float) -> tuple[int | None, float | None]:
    if not train_log.exists():
        return None, None

    current_iter = None
    done_iter = None
    last_value = None
    ansi = re.compile(r"\x1b\[[0-9;]*m")

    with train_log.open("r", errors="ignore") as log_file:
        for raw_line in log_file:
            line = ansi.sub("", raw_line)
            match = re.search(r"Learning iteration\s+(\d+)/(\d+)", line)
            if match:
                current_iter = int(match.group(1))
            match = re.search(r"Curriculum/lin_vel_cmd_levels:\s*([0-9.]+)", line)
            if match:
                last_value = float(match.group(1))
                if current_iter is not None and last_value >= target:
                    done_iter = current_iter

    return done_iter, last_value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-session", required=True)
    parser.add_argument("--train-log", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--target", type=float, default=10.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    print(f"[watch] Monitoring {args.train_log} for curriculum target {args.target:.1f}", flush=True)
    last_status = 0.0

    while tmux_alive(args.train_session):
        done_iter, last_value = curriculum_done_iter(args.train_log, args.target)
        run_dir = latest_run_dir(args.log_root, args.run_name)

        now = time.time()
        if now - last_status > 120:
            latest_model = latest_model_iter(run_dir) if run_dir else None
            print(
                f"[watch] last curriculum={last_value}, done_iter={done_iter}, latest_model={latest_model}",
                flush=True,
            )
            last_status = now

        if done_iter is not None and run_dir is not None:
            required_model = ((done_iter + 99) // 100) * 100
            latest_model = latest_model_iter(run_dir)
            if latest_model is not None and latest_model >= required_model:
                print(
                    "[watch] Curriculum reached "
                    f"{args.target:.1f} at iter {done_iter}; model_{latest_model}.pt exists. "
                    "Stopping training.",
                    flush=True,
                )
                subprocess.run(["tmux", "kill-session", "-t", args.train_session], check=False)
                break

        time.sleep(args.poll_seconds)

    print("[watch] Exiting.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
