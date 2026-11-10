#!/usr/bin/env bash
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail
: "${LOOPWAM_REPO:?Set LOOPWAM_REPO to the clean upstream LoopWAM checkout}"
: "${LOOPWAM_DATA_ROOT:?Set LOOPWAM_DATA_ROOT to the RoboTwin dataset root}"
: "${LOOPWAM_TEXT_CACHE_ROOT:?Set LOOPWAM_TEXT_CACHE_ROOT to the text embedding cache}"
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPA_SOURCE="$(cd "${EXAMPLE_DIR}/../.." && pwd)"
cd "${LOOPWAM_REPO}"
export PYTHONPATH="${TPA_SOURCE}:${LOOPWAM_REPO}/src:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${LOOPWAM_REPO}/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MACHINE_RANK="${MACHINE_RANK:-0}"
for value in "$NNODES" "$NPROC_PER_NODE" "$MACHINE_RANK"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "Node and rank counts must be integers" >&2; exit 2; }
done
(( NNODES > 0 && NPROC_PER_NODE > 0 && MACHINE_RANK < NNODES )) || exit 2
if (( NNODES > 1 )); then
  : "${MASTER_ADDR:?Set the rendezvous address for multi-node training}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${LOOPWAM_REPO}/runs/turbophysai_$(date +%Y%m%d_%H%M%S)}"
exec python -m turbo_physai.cli run --model loopwam --log-report \
  accelerate launch --config_file "${EXAMPLE_DIR}/accelerate.yaml" \
  --deepspeed_config_file "${EXAMPLE_DIR}/deepspeed_zero1.json" \
  --num_processes "$((NNODES * NPROC_PER_NODE))" --num_machines "${NNODES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MASTER_ADDR:-127.0.0.1}" --main_process_port "${MASTER_PORT:-29500}" \
  --deepspeed_multinode_launcher standard \
  scripts/train.py --config-dir "${EXAMPLE_DIR}/configs" \
  task=robotwin_quality_score_3low_1cam_stitched_384_1e-4_action_weighted \
  data=robotwin_mixed_quality_score_3low_stitched_ssd \
  model.video_cross_attend_proprio=false \
  data.train.num_frames=65 data.val.num_frames=65 \
  data.train.action_video_freq_ratio=8 data.val.action_video_freq_ratio=8 \
  batch_size=8 num_workers=2 max_steps=300 log_every=1 save_every=0 eval_every=0 \
  "output_dir=${OUTPUT_DIR}" "$@"
