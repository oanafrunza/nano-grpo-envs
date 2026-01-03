#!/usr/bin/env bash
set -euo pipefail

# Monitor completion of current adapt suite and then launch LEN512 suite via runner.

cd "$(dirname "$0")"

BASE_DIR="exp_output/science2_phase_adapt_suite"
SEEDS=${SEEDS:-"1 2"}
NAMES=("phase_adapt_softmask" "phase_adapt_diversity" "phase_split_masking")

FOLLOWUP_MARKER="${BASE_DIR}/.len512_followup_launched"

if [ -f "$FOLLOWUP_MARKER" ]; then
  echo "[Followup] LEN512 suite already launched (marker present). Exiting."
  exit 0
fi

all_completed() {
  for s in $SEEDS; do
    for n in "${NAMES[@]}"; do
      if [ ! -f "${BASE_DIR}/${n}_seed${s}.completed" ]; then
        return 1
      fi
    done
  done
  return 0
}

echo "[Followup] Watching adapt suite for completion... (seeds: $SEEDS)"
while true; do
  if all_completed; then
    echo "[Followup] All standard adapt runs completed. Launching LEN512 suite via runner..."
    touch "$FOLLOWUP_MARKER"
    bash run_science2_phase_adapt.sh
    echo "[Followup] LEN512 suite launch finished."
    exit 0
  fi
  sleep 60
done
