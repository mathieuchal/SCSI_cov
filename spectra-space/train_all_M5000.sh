#!/bin/bash
# Runs both finite-data (M=5000) training pipelines (iw_ri_d8 then iso_diag_d8) back to back -- see
# train_iw_ri_d8_M5000.sh / train_iso_diag_d8_M5000.sh for what each one does. Training only, no
# evaluation yet (see train_all_spectra.sh for the M=-1 campaign's full training+eval+report pipeline,
# once we're ready to add the same for M=5000).
set -e
cd "$(dirname "$0")"

echo "=== train_all_M5000: starting iw_ri_d8 ==="
./train_iw_ri_d8_M5000.sh
echo "=== train_all_M5000: iw_ri_d8 done ==="

echo "=== train_all_M5000: starting iso_diag_d8 ==="
./train_iso_diag_d8_M5000.sh
echo "=== train_all_M5000: iso_diag_d8 done ==="

echo "=== train_all_M5000: ALL DONE ==="
