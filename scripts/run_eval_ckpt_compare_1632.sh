#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/shared/healthinfolab/phz24002/PocketXMol}"
PYTHON_BIN="${PYTHON_BIN:-/shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol/bin/python}"
DEVICE="${DEVICE:-cuda:2}"
NUM_MOLS="${NUM_MOLS:-1000}"
SAMPLE_STEPS="${SAMPLE_STEPS:-1 2 4 8 16 32}"

CONFIG_DISTILL="${CONFIG_DISTILL:-outputs_distill_ve/consistency_distill_20260328_164246/sampling_standard_snr.yaml}"
CONFIG_TASK="${CONFIG_TASK:-configs/sample/test/sbdd_csd/simple.yml}"
CKPT_DIR="${CKPT_DIR:-outputs_distill_ve/consistency_distill_20260328_164246/checkpoints}"
OUT_ROOT="${OUT_ROOT:-outputs_consistency_eval_ckpt_compare_1632}"

STANDARD_SPLIT_BY_NAME_PATH="${STANDARD_SPLIT_BY_NAME_PATH:-/shared/healthinfolab/phz24002/AliDiff/data/split_by_name.pt}"
STANDARD_TEST_SET_ROOT="${STANDARD_TEST_SET_ROOT:-/shared/healthinfolab/phz24002/AliDiff/data/test_set}"

CHECKPOINT_STEPS=(50000 100000)

cd "$ROOT_DIR"
mkdir -p "$OUT_ROOT"

for STEP in "${CHECKPOINT_STEPS[@]}"; do
  CKPT_PATH="${CKPT_DIR}/step_${STEP}.pt"
  RUN_OUTDIR="${OUT_ROOT}/step_${STEP}"

  if [[ ! -f "$CKPT_PATH" ]]; then
    echo "Missing checkpoint: $CKPT_PATH" >&2
    exit 1
  fi

  mkdir -p "$RUN_OUTDIR"
  echo "Running evaluation for step_${STEP} on ${DEVICE}"
  "$PYTHON_BIN" scripts/eval_consistency.py \
    --config_distill "$CONFIG_DISTILL" \
    --consistency_ckpt "$CKPT_PATH" \
    --config_task "$CONFIG_TASK" \
    --sample_steps ${SAMPLE_STEPS} \
    --num_mols "$NUM_MOLS" \
    --device "$DEVICE" \
    --outdir "$RUN_OUTDIR" \
    --standard_eval \
    --standard_split_by_name_path "$STANDARD_SPLIT_BY_NAME_PATH" \
    --standard_test_set_root "$STANDARD_TEST_SET_ROOT"
done
