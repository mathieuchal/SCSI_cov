#!/bin/bash
# Reproduces the production iw_ri_d8_inf spectral_follmer SC-SI run end-to-end: training with the exact
# hyperparameters used for the current results/toys/iw_ri_d8_inf_spectral_follmer_* checkpoints (recovered
# from this session's history), then the big-J oracle/log-score full-curve rescore and the per-checkpoint
# W1-mean eval, at the same J (default 2e5) used in production. Re-running this script from a clean
# results/toys/ is the reproducibility check: iw_ri_d8_inf is exactly conjugate, so the oracle/IW-ML side
# is deterministic; the SC-SI side should land on the same checkpoints/scores modulo training-seed effects
# already fixed by toy_train.py's default --seed 0.
set -e
cd "$(dirname "$0")"

J=${1:-200000}
TOY=iw_ri_d8_inf
TAG=spectral_follmer

echo "[$TOY] activating venv"
source .venv/bin/activate

echo "[$TOY] training start: encoding=spectral sampler=follmer n_outer=120 steps_first=4000 steps_outer=800"
python3 toy_train.py --toy "$TOY" --encoding spectral --sampler follmer --tag "$TAG" \
  --n_outer 120 --steps_first 4000 --steps_outer 800
echo "[$TOY] training done (checkpoints up to k=120)"

echo "[$TOY] oracle/IW-ML log-score starting (J=$J)"
python3 -c "
from log_score_oracle import eval_oracle_IW_log_score
eval_oracle_IW_log_score('$TOY', J=$J, seeds=(4321,))
"
echo "[$TOY] oracle/IW-ML log-score done"

echo "[$TOY] SC-SI checkpoint full-curve rescore starting (J=$J)"
python3 -c "
from log_score_scsi import eval_scsi_checkpoints_log_score
eval_scsi_checkpoints_log_score('$TOY', '$TAG', J=$J)
"
echo "[$TOY] SC-SI checkpoint rescore done"

echo "[$TOY] W1-mean per-checkpoint eval starting"
python3 -c "
from toy_W1_mean import w1_mean_dict
w1_mean_dict(['$TOY'], '$TAG')
"
echo "[$TOY] W1 done"

echo "[$TOY] ALL DONE"
