#!/bin/bash
# Finite-data (M=5000) counterpart of train_iw_ri_d8.sh: same architecture/hyperparameters (encoding=
# spectral, sampler=follmer, n_outer=120, steps_first=4000, steps_outer=800) applied to the iw_ri_d8 toy
# (toys.py: IWToy("iw_ri_d8", 8, 40, 5000, "ri", 16.0) -- same d/N/ri_mode/kappa0 as iw_ri_d8_inf, only M
# differs), to see how much finite data degrades performance relative to the M=-1 (infinite-data) case.
# Output lands under results/toys_M=5000/ (--results_dir), kept apart from the M=-1 results.
#
# TRAINING ONLY -- no oracle/log-score/W1 evaluation here yet (see train_iw_ri_d8.sh for what that would
# look like once we're ready to evaluate this run).
set -e
cd "$(dirname "$0")"

TOY=iw_ri_d8
TAG=spectral_follmer
RESULTS_DIR=results/toys_M=5000

echo "[$TOY] activating venv"
source .venv/bin/activate

echo "[$TOY] training start: encoding=spectral sampler=follmer n_outer=120 steps_first=4000 steps_outer=800 (M=5000)"
python3 toy_train.py --toy "$TOY" --encoding spectral --sampler follmer --tag "$TAG" \
  --results_dir "$RESULTS_DIR" --n_outer 120 --steps_first 4000 --steps_outer 800
echo "[$TOY] training done (checkpoints up to k=120)"
