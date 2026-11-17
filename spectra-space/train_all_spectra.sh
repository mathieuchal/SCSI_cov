#!/bin/bash
# Runs both spectral_follmer production pipelines (iw_ri_d8_inf then iso_diag_d8_inf) back to back --
# see train_iw_ri_d8.sh / train_iso_diag_d8.sh for what each one does. Pass a J value as $1 to override
# the big-J oracle/log-score default (2e5) for both. Finishes with toy_report_spectral_follmer.py's
# combined log-score/W1 curve figure (figs/fig_spectral_follmer.png) and summary table
# (figs/table_spectral_follmer.txt) -- run only here, not in the per-toy scripts, since it plots both
# toys side by side and needs both to have already completed.
set -e
cd "$(dirname "$0")"

J=${1:-200000}

echo "=== train_all_spectra: starting iw_ri_d8_inf (J=$J) ==="
./train_iw_ri_d8.sh "$J"
echo "=== train_all_spectra: iw_ri_d8_inf done ==="

echo "=== train_all_spectra: starting iso_diag_d8_inf (J=$J) ==="
./train_iso_diag_d8.sh "$J"
echo "=== train_all_spectra: iso_diag_d8_inf done ==="

echo "=== train_all_spectra: generating figure + summary table ==="
source .venv/bin/activate
python3 toy_report_spectral_follmer.py
echo "=== train_all_spectra: ALL DONE ==="
