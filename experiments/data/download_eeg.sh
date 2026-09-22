#!/usr/bin/env bash
# Download the PhysioNet EEG Motor Movement/Imagery baseline runs (R01 = eyes open, R02 = eyes closed)
# used by eeg_data.py.  ~450MB for all 109 subjects.
#
# Usage:
#   ./data/download_eeg.sh                 # all 109 subjects
#   ./data/download_eeg.sh 10              # first 10 subjects only (quick smoke test)
#
# Data source: Schalk, G. EEG Motor Movement/Imagery dataset. PhysioNet, 2009.
# https://physionet.org/content/eegmmidb/1.0.0/
set -euo pipefail
cd "$(dirname "$0")"

N_SUBJECTS="${1:-109}"
OUT=eeg
PARALLEL=6

mkdir -p "$OUT"
FILELIST=$(mktemp)
trap 'rm -f "$FILELIST"' EXIT

for s in $(seq 1 "$N_SUBJECTS"); do
  S=$(printf "S%03d" "$s")
  for r in R01 R02; do
    echo "${S}/${S}${r}.edf"
  done
done > "$FILELIST"

echo "Downloading $((N_SUBJECTS * 2)) runs into $OUT/ ..."
xargs -P "$PARALLEL" -I{} sh -c '
  f=$(basename {})
  if [ -s "'"$OUT"'/$f" ]; then
    echo "skip $f (already present)"
  else
    curl -sS -m 120 -o "'"$OUT"'/$f" "https://physionet.org/files/eegmmidb/1.0.0/{}" \
      && echo "ok   $f" \
      || echo "FAIL $f"
  fi
' < "$FILELIST"

echo "Done. $(ls "$OUT"/*.edf 2>/dev/null | wc -l | tr -d ' ') .edf files in $OUT/."
