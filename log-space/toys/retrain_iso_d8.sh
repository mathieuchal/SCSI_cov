#!/bin/bash
# Retrain the d=8 iso-diagonal production run (iso_diag_d8_inf, MLP-4 w1024) with the current, fully seeded code, then
# rebuild everything derived from it: the W1 trajectory and {toy}_W1.json, the J=2e6 log-score trajectory and
# {toy}_logscore.json, the W1 tables and the trajectory figures.
#
# Why: the run of 2026-11-11 was trained (and its log-score computed) before the trunc_normal RNG fix (11-11 15:24 UTC):
# torch.nn.init.trunc_normal_ ignored the generator, so its training data and validation pairs are not reproducible.
# The iso IW-ML fit cache {toy}_iwfit_ref.pt (2026-11-09) also predates the fix and is rebuilt. The toy_eval
# reference {toy}_ref.pt (8 HMC chains, built 2026-11-16 on the fixed test set) is still valid and is kept.
# Run on 2026-11-17 (00:36-03:32 UTC); steps 3-5 were then finished with the current scripts below.
#
#   bash retrain_iso_d8.sh        (from log-space/toys; ~3h GPU; master log ../logs/retrain_iso_d8.log)
# Steps: 0 archive old outputs -> results/toys_M=-1/archive_iso_oldrng/   1 train k0-700 (cosine 5e-4 -> 1e-5)
#        2 resume k700-1000 (constant 5e-5)   3 W1: trajectory (toy_W1_mean.py) and best_k (toy_eval.py)
#        4 log-score J=2e6 (toy_oracle_log_score.py)   5 tables and figures
set -euo pipefail
cd "$(dirname "$0")"
source ../.venv/bin/activate
T=iso_diag_d8_inf; TAG=mlp_w1024_d4; STEM=${T}_${TAG}; R=results/toys_M=-1; L=../logs; A=$R/archive_iso_oldrng
exec > >(tee -a $L/retrain_iso_d8.log) 2>&1
echo "=== start $(date -u)"

# 0. archive: the old run's checkpoints/evals/model/train json, the iso log-score and W1 files, the IW-ML fit cache
[ -e "$A" ] && { echo "$A already exists: archive done before? remove or rename it first"; exit 1; }
mkdir -p "$A"
for f in $R/${STEM}_* $R/${T}_iwfit_ref.pt $R/${T}_logscore.json $R/${T}_W1.json; do [ -e "$f" ] && mv "$f" "$A"/; done
echo "archived $(ls "$A" | wc -l) files to $A"

# 1-2. training (production protocol, defaults of toy_train.py except --hidden 1024)
python toy_train.py --toy $T --tag $TAG --hidden 1024 > $L/${STEM}_train.log 2>&1
echo "stage 1 done $(date -u)"
python toy_train.py --toy $T --tag $TAG --n_outer 1000 --resume $R/${STEM}_k700.pt > $L/${STEM}_train_k700-1000.log 2>&1
echo "stage 2 done $(date -u)"

# 3. W1: trajectory on the evaluation grid (-> _train.json) and the run's best_k with oracle floor and IW-ML (-> _W1.json)
python toy_W1_mean.py --toys $T > $L/toy_W1_mean_iso.log 2>&1
python toy_eval.py --toy $T > $L/${STEM}_bestk_eval.log 2>&1
echo "W1 done $(date -u)"

# 4. predictive log-score at J=2e6: oracle (HMC), IW-ML, SC-SI on the grid + best_k (-> _logscore.json, _train.json)
python toy_oracle_log_score.py --toys $T > $L/logscore_d8_J2000000_iso_diag.log 2>&1
echo "log-score done $(date -u)"

# 5. tables and figures (all toys; only the iso inputs changed)
python toy_w1_tables.py > $L/toy_w1_tables.log 2>&1
python toy_logscore_w1_curve.py > $L/toy_logscore_w1_curve.log 2>&1
python toy_logscore_w1_curve.py --no_hetero >> $L/toy_logscore_w1_curve.log 2>&1
python toy_logscore_w1_curve.py --mean >> $L/toy_logscore_w1_curve.log 2>&1
echo "=== done $(date -u)"
