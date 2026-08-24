#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 PHYSICAL_GPU RUN_BASE ARM [ARM ...]" >&2
  exit 2
fi

physical_gpu="$1"
run_base="$2"
shift 2
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for arm in "$@"; do
  "$repo_root/tools/run_figure5_single_hurdle_arm.sh" \
    "$arm" "$physical_gpu" "$run_base"
done
