#!/bin/bash
# Reproduces the production iso_diag_d8_inf spectral_follmer SC-SI run end-to-end. Unlike iw_ri, this run
# was trained in TWO stages in production (n_outer=120, then resumed to n_outer=240 -- see
# train_iw_ri_d8.sh's header for the shared methodology note): reproducing both stages matters because
# resuming restarts the cosine LR schedule's floor for the remaining iterations (lr0 halves on resume in
# the original run's log), which a single n_outer=240 call from scratch would NOT reproduce. Then the
# same big-J oracle/log-score full-curve rescore and per-checkpoint W1-mean eval as iw_ri, at the same J
# (default 2e5). iso_diag_d8_inf is NOT exactly conjugate (HMC-based oracle) -- so its oracle number is
# expected to have a little run-to-run noise even at a matched seed, unlike iw_ri's.
set -e
cd "$(dirname "$0")"

J=${1:-200000}
TOY=iso_diag_d8_inf
TAG=spectral_follmer

echo "[$TOY] activating venv"
source .venv/bin/activate

echo "[$TOY] training stage 1/2 start: encoding=spectral sampler=follmer n_outer=120 steps_first=4000 steps_outer=800"
python3 toy_train.py --toy "$TOY" --encoding spectral --sampler follmer --tag "$TAG" \
  --n_outer 120 --steps_first 4000 --steps_outer 800
echo "[$TOY] training stage 1/2 done (checkpoints up to k=120)"

echo "[$TOY] training stage 2/2 start: resuming from k=120 up to n_outer=240"
python3 toy_train.py --toy "$TOY" --encoding spectral --sampler follmer --tag "$TAG" \
  --resume "results/toys/${TOY}_${TAG}_k120.pt" \
  --n_outer 240 --steps_outer 800 --batch 1024 --hidden 256 --depth 4 --arch mlp \
  --n_val 300 --verbose True
echo "[$TOY] training stage 2/2 done (checkpoints up to k=240)"

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
