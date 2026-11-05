#!/usr/bin/env bash
# Confirm the GPU, the data disk, and the cache paths before training.
set -euo pipefail

if [[ -f /etc/profile.d/vaaku.sh ]]; then
  # shellcheck disable=SC1091
  . /etc/profile.d/vaaku.sh
fi

ROOT="${VAAKKU_DATA_ROOT:-/home/ubuntu/vaaku}"

echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "kernel: $(uname -r)"
python3 --version
ffmpeg -version | awk 'NR == 1 {print}'
echo

nvidia-smi
echo

df -hT / "${ROOT}"
echo
echo "VAAKKU_DATA_DIR=${VAAKKU_DATA_DIR:-unset}"
echo "VAAKKU_ARTIFACT_DIR=${VAAKKU_ARTIFACT_DIR:-unset}"
echo "HF_HOME=${HF_HOME:-unset}"
echo "TMPDIR=${TMPDIR:-unset}"

test -d "${ROOT}/data/raw"
test -d "${ROOT}/data/prepared"
test -d "${ROOT}/artifacts"
test -d "${ROOT}/cache/huggingface"
test -d "${ROOT}/tmp"

avail_kb="$(df -Pk "${ROOT}" | awk 'NR == 2 {print $4}')"
if [[ "${avail_kb}" -lt 20000000 ]]; then
  echo "Data disk has under 20 GiB free." >&2
  exit 1
fi

echo
echo "GPU and data disk look ready."
