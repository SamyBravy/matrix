#!/usr/bin/env bash
set -euo pipefail

THRESHOLD=${1:-80} # default 80%
LCOV_FILE=${2:-lcov.info}

if [[ ! -f "$LCOV_FILE" ]]; then
  echo "Missing $LCOV_FILE. Run coverage first (cargo llvm-cov --workspace --lcov --output-path lcov.info)." >&2
  exit 2
fi

LF=$(grep -h '^LF:' "$LCOV_FILE" | awk -F: '{s+=$2} END {print s+0}')
LH=$(grep -h '^LH:' "$LCOV_FILE" | awk -F: '{s+=$2} END {print s+0}')
PCT=0
if [[ $LF -gt 0 ]]; then
  PCT=$(( 100 * LH / LF ))
fi

if [[ $PCT -lt $THRESHOLD ]]; then
  echo "Coverage ${PCT}% is below threshold ${THRESHOLD}%" >&2
  exit 1
fi

echo "Coverage ${PCT}% >= threshold ${THRESHOLD}% ✔"
