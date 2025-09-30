#!/usr/bin/env bash
set -euo pipefail

# Generates a simple coverage percentage badge (SVG) from lcov.info using cargo-llvm-cov.
# Requirements: cargo-llvm-cov installed and lcov.info up to date.
# Usage: ./scripts/coverage_badge.sh [lcov_file] [output_svg]

LCOV_FILE=${1:-lcov.info}
OUT=${2:-coverage.svg}

if [[ ! -f "$LCOV_FILE" ]]; then
  echo "LCOV file '$LCOV_FILE' not found. Run: cargo llvm-cov --workspace --lcov --output-path lcov.info" >&2
  exit 1
fi

# Extract metrics (LF: total lines found, LH: lines hit)
LF=$(grep -h '^LF:' "$LCOV_FILE" | awk -F: '{s+=$2} END {print s+0}')
LH=$(grep -h '^LH:' "$LCOV_FILE" | awk -F: '{s+=$2} END {print s+0}')
if [[ $LF -eq 0 ]]; then
  PCT=0
else
  PCT=$(( 100 * LH / LF ))
fi

COLOR=red
if   [[ $PCT -ge 90 ]]; then COLOR=brightgreen
elif [[ $PCT -ge 80 ]]; then COLOR=green
elif [[ $PCT -ge 70 ]]; then COLOR=yellowgreen
elif [[ $PCT -ge 60 ]]; then COLOR=yellow
elif [[ $PCT -ge 50 ]]; then COLOR=orange
fi

cat > "$OUT" <<SVG
<svg xmlns="http://www.w3.org/2000/svg" width="140" height="20" role="img" aria-label="coverage:$PCT%">
  <linearGradient id="a" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <rect rx="3" width="140" height="20" fill="#555"/>
  <rect rx="3" x="70" width="70" height="20" fill="#4c1"/>
  <rect rx="3" width="140" height="20" fill="url(#a)"/>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="35" y="15" fill="#010101" fill-opacity=".3">coverage</text>
    <text x="35" y="14">coverage</text>
    <text x="105" y="15" fill="#010101" fill-opacity=".3">$PCT%</text>
    <text x="105" y="14">$PCT%</text>
  </g>
</svg>
SVG

# Patch color in second rect (after computing) for chosen color
sed -i "s/fill=\"#4c1\"/fill=\"$COLOR\"/" "$OUT"

echo "Generated $OUT with coverage $PCT% ($LH/$LF)"
