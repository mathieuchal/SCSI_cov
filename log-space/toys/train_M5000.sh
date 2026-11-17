#!/bin/bash
# Finite-data (M=5000) counterparts of three of the d=8 production runs (iw_nonri, factor, diag): same toys (prior, N=40),
# same architectures, one fixed training ensemble of M=5000 noisy Ce. One training stage only: k=0..700 with the cosine outer
# lr (no constant-lr extension to k=1000). Training only (no evaluation). Outputs in results/toys_M=5000/, logs in
# ../logs/M=5000/. One run at a time.
#
#   bash train_M5000.sh        (from log-space/toys; master log ../logs/M=5000/train_M5000.log)
set -euo pipefail
cd "$(dirname "$0")"
source ../.venv/bin/activate
O=results/toys_M=5000; L=../logs/M=5000
mkdir -p $O $L
exec > >(tee -a $L/train_M5000.log) 2>&1

run () {   # toy tag [extra toy_train args]
  T=$1; TAG=$2; shift 2; S=${T}_${TAG}
  if [ -f $O/${S}_train.json ]; then echo "=== $S already trained, skipped"; return; fi
  echo "=== $S (k=0..700)  $(date -u)"
  python toy_train.py --toy $T --tag $TAG "$@" > $L/${S}_train.log 2>&1
  echo "=== $S done  $(date -u)"
}

echo "=== start $(date -u)"
run iw_nonri_d8_M5000 mlp_w512_d4
run factor_d8_M5000 mlp_w512_d4
run diag_d8_M5000 mlp_w512_d4
echo "=== all done $(date -u)"
