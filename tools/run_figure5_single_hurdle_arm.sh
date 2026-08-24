#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 {pancreas1|breast1|lung1} PHYSICAL_GPU RUN_BASE" >&2
  exit 2
fi

arm="$1"
physical_gpu="$2"
run_base="$(realpath -m "$3")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${GHIST_PYTHON:-python}"

: "${GHIST_DATA_ROOT:?Set GHIST_DATA_ROOT to the GHIST data directory}"

case "$arm" in
  pancreas1|breast1|lung1)
    ;;
  *)
    echo "Unknown arm: $arm" >&2
    exit 2
    ;;
esac

config_rel="configs/figure5_single_hurdle/${arm}.json"
config_path="$repo_root/$config_rel"
arm_root="$run_base/$arm"
run_id="figure5_${arm}_hurdle_seed20260807"
experiment_path="$arm_root/results/$run_id"
log_path="$arm_root/train.log"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Missing GHIST Python: $python_bin" >&2
  exit 1
fi
if [[ ! -f "$config_path" ]]; then
  echo "Missing config: $config_path" >&2
  exit 1
fi
if [[ -e "$arm_root/started_at_utc.txt" || -e "$experiment_path" ]]; then
  echo "Refusing to reuse an existing arm root: $arm_root" >&2
  exit 1
fi

mkdir -p "$arm_root/cache" "$arm_root/mplconfig" "$arm_root/tmp" "$arm_root/provenance"
cp "$config_path" "$arm_root/provenance/${arm}.json"
date -u +%Y-%m-%dT%H:%M:%SZ > "$arm_root/started_at_utc.txt"
git -C "$repo_root" rev-parse HEAD > "$arm_root/provenance/git_head.txt"
git -C "$repo_root" status --short > "$arm_root/provenance/git_status.txt"
git -C "$repo_root" diff --binary > "$arm_root/provenance/working_tree.patch"
sha256sum "$config_path" "$repo_root/train.py" > "$arm_root/provenance/launch_sha256.txt"
"$python_bin" -m pip freeze > "$arm_root/provenance/pip_freeze.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader > "$arm_root/provenance/gpus_at_launch.csv"

export OUTPUT_ROOT="$arm_root"
export CACHE_ROOT="$arm_root/cache"
export RUN_ID="$run_id"
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export GHIST_RUN_ROLE="FULL"
export GHIST_FATAL_LOG_ERRORS="1"
export FORCE_REIMPUTE="1"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export PYTHONUNBUFFERED="1"
export PYTHONDONTWRITEBYTECODE="1"
export MPLCONFIGDIR="$arm_root/mplconfig"
export TMPDIR="$arm_root/tmp"
unset GHIST_SVG_COHORT_MANIFEST

{
  echo "arm=$arm"
  echo "physical_gpu=$physical_gpu"
  echo "logical_gpu=0"
  echo "run_id=$run_id"
  echo "repo_root=$repo_root"
  echo "config=$config_path"
  echo "output_root=$OUTPUT_ROOT"
  echo "cache_root=$CACHE_ROOT"
} > "$arm_root/provenance/launch.env"

set +e
(
  cd "$repo_root"
  "$python_bin" -u train.py \
    --config_file "$config_path" \
    --resume_epoch 0 \
    --fold_id 1 \
    --gpu_id 0
) >> "$log_path" 2>&1
status=$?
set -e

echo "$status" > "$arm_root/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$arm_root/finished_at_utc.txt"
exit "$status"
