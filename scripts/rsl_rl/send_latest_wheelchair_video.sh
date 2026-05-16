#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

TASK=${TASK:-Unitree-G1-29dof-Wheelchair-Dynamic-Push-Observed}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-logs/rsl_rl/unitree_g1_29dof_wheelchair_dynamic_push_observed}
RUN_GLOB=${RUN_GLOB:-*rubber_hands*}
RUN_DIR=${RUN_DIR:-}
CHECKPOINT=${CHECKPOINT:-}
NUM_ENVS=${NUM_ENVS:-1}
VIDEO_START_STEP=${VIDEO_START_STEP:-50}
VIDEO_LENGTH=${VIDEO_LENGTH:-300}
VIDEO_CAMERA_ORBIT_DEG=${VIDEO_CAMERA_ORBIT_DEG:-12.0}
VIDEO_EYE_OFFSET=${VIDEO_EYE_OFFSET:--4.5 -3.2 2.2}
VIDEO_TARGET_OFFSET=${VIDEO_TARGET_OFFSET:-0.5 0.0 0.9}
EMAIL_TO=${EMAIL_TO:-zeul@mordasiewicz.com}
SEND_EMAIL=${SEND_EMAIL:-1}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR=$(find "${EXPERIMENT_DIR}" -mindepth 1 -maxdepth 1 -type d -name "${RUN_GLOB}" | sort | tail -1)
fi

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR=$(find "${EXPERIMENT_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
fi

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "Could not find run directory under ${EXPERIMENT_DIR}" >&2
  exit 1
fi

if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT=$(find "${RUN_DIR}" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
fi

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "Could not find checkpoint in ${RUN_DIR}" >&2
  exit 1
fi

read -r -a EYE_OFFSET <<< "${VIDEO_EYE_OFFSET}"
read -r -a TARGET_OFFSET <<< "${VIDEO_TARGET_OFFSET}"
if [[ "${#EYE_OFFSET[@]}" -ne 3 || "${#TARGET_OFFSET[@]}" -ne 3 ]]; then
  echo "VIDEO_EYE_OFFSET and VIDEO_TARGET_OFFSET must each contain three numbers." >&2
  exit 1
fi

echo "Rendering checkpoint: ${CHECKPOINT}"
echo "Task: ${TASK}"

TERM=xterm conda run -n isaaclab python scripts/rsl_rl/play.py \
  --headless \
  --enable_cameras \
  --task "${TASK}" \
  --num_envs "${NUM_ENVS}" \
  --checkpoint "${CHECKPOINT}" \
  --video \
  --video-start-step "${VIDEO_START_STEP}" \
  --video_length "${VIDEO_LENGTH}" \
  --video-follow-robot \
  --video-camera-eye-offset "${EYE_OFFSET[@]}" \
  --video-camera-target-offset "${TARGET_OFFSET[@]}" \
  --video-camera-orbit-deg "${VIDEO_CAMERA_ORBIT_DEG}"

PLAY_VIDEO_DIR="$(dirname "${CHECKPOINT}")/videos/play"
PLAY_VIDEO=$(find "${PLAY_VIDEO_DIR}" -maxdepth 1 -name '*.mp4' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
if [[ -z "${PLAY_VIDEO}" || ! -f "${PLAY_VIDEO}" ]]; then
  echo "Could not find rendered video under ${PLAY_VIDEO_DIR}" >&2
  exit 1
fi

OUT_DIR=${OUT_DIR:-logs/demos/wheelchair_latest_${TIMESTAMP}}
mkdir -p "${OUT_DIR}"
OUT_VIDEO="${OUT_DIR}/$(basename "${CHECKPOINT}" .pt)_latest.mp4"
cp "${PLAY_VIDEO}" "${OUT_VIDEO}"
echo "Copied video: ${OUT_VIDEO}"

if [[ "${SEND_EMAIL}" == "1" ]]; then
  if ! command -v gog >/dev/null 2>&1; then
    echo "gog CLI was not found; video was rendered but not emailed." >&2
    exit 1
  fi
  gog send \
    --to "${EMAIL_TO}" \
    --subject "Latest wheelchair policy video ($(basename "${CHECKPOINT}" .pt))" \
    --body "Attached is the latest wheelchair policy playback rendered from ${CHECKPOINT}." \
    --attach "${OUT_VIDEO}"
fi

echo "Done."
