#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/debug/find_nan_source.sh ckpts/pretrained.pth
#
# This script runs a binary-disable workflow to localize upstream NaN/Inf source.
# It prints the exact commands and executes them one by one.

PRETRAINED_MODEL="${1:-ckpts/pretrained.pth}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TOOLS_DIR="${REPO_ROOT}/tools"
CFG_FILE="cfgs/mambafusion_models/mamba_fusion.yaml"
NGPUS=4
COMMON_SET_ARGS=(OPTIMIZATION.BATCH_SIZE_PER_GPU 1 OPTIMIZATION.ACCUMULATION_STEPS 6)
if [[ "${PRETRAINED_MODEL}" = /* ]]; then
  PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL}"
else
  PRETRAINED_MODEL_PATH="${REPO_ROOT}/${PRETRAINED_MODEL}"
fi

run_case() {
  local tag="$1"
  shift
  echo "=================================================="
  echo "[RUN] ${tag}"
  echo "=================================================="
  (
    cd "${TOOLS_DIR}"
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
    bash scripts/dist_train.sh ${NGPUS} \
      --cfg_file ${CFG_FILE} \
      --sync_bn \
      --pretrained_model "${PRETRAINED_MODEL_PATH}" \
      --use_amp \
      --logger_iter_interval 200 \
      --extra_tag "${tag}" \
      --set "${COMMON_SET_ARGS[@]}" \
      "$@"
  )
}

echo "[INFO] Step-1 baseline"
run_case "nanloc-base"

echo "[INFO] Step-2 disable both heavy FUSER switches"
run_case "nanloc-fuser-off" \
  MODEL.FUSER.USE_GATED_FUSION False MODEL.FUSER.USE_SPARSE_DISTILL False

echo "[INFO] Step-3A enable only gated fusion"
run_case "nanloc-only-gated" \
  MODEL.FUSER.USE_GATED_FUSION True MODEL.FUSER.USE_SPARSE_DISTILL False

echo "[INFO] Step-3B enable only sparse distill"
run_case "nanloc-only-distill" \
  MODEL.FUSER.USE_GATED_FUSION False MODEL.FUSER.USE_SPARSE_DISTILL True

echo "[DONE] Compare logs for:"
echo "  - first non-finite iteration"
echo "  - bad_tb_keys frequency"
echo "  - b_time(avg), OOM, and Hungarian errors"
