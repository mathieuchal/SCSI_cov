#!/bin/bash
# Finite-data (M=5000) counterpart of train_iso_diag_d8.sh: same two-stage architecture/hyperparameters
# applied to the iso_diag_d8 toy (toys.py: IsoDiagToy("iso_diag_d8", 8, 40, 5000) -- same d/N as
# iso_diag_d8_inf, only M differs). Two stages (n_outer=120, then resumed to n_outer=240) for the same
# reason as train_iso_diag_d8.sh: resuming restarts the cosine LR schedule's floor for the remaining
# iterations, which a single n_outer=240 call from scratch would NOT reproduce. Output lands under
# results/toys_M=5000/ (--results_dir), kept apart from the M=-1 results.
#
# PAUSED: a first attempt crashed during stage 2 (~k=180, Cholesky non-PD), shortly after the lr jump at
# the stage 1/2 boundary described above -- plausible cause, not confirmed. Not pursuing a fix right now.
#
# TRAINING ONLY -- no oracle/log-score/W1 evaluation here yet (see train_iso_diag_d8.sh for what that
# would look like once we're ready to evaluate this run).
set -e
cd "$(dirname "$0")"

TOY=iso_diag_d8
TAG=spectral_follmer
RESULTS_DIR=results/toys_M=5000

echo "[$TOY] activating venv"
source .venv/bin/activate

echo "[$TOY] training stage 1/2 start: encoding=spectral sampler=follmer n_outer=120 steps_first=4000 steps_outer=800 (M=5000)"
python3 toy_train.py --toy "$TOY" --encoding spectral --sampler follmer --tag "$TAG" \
  --results_dir "$RESULTS_DIR" --n_outer 120 --steps_first 4000 --steps_outer 800
echo "[$TOY] training stage 1/2 done (checkpoints up to k=120)"

echo "[$TOY] training stage 2/2 start: resuming from k=120 up to n_outer=240"
python3 toy_train.py --toy "$TOY" --encoding spectral --sampler follmer --tag "$TAG" \
  --results_dir "$RESULTS_DIR" \
  --resume "${RESULTS_DIR}/${TOY}_${TAG}_k120.pt" \
  --n_outer 240 --steps_outer 800 --batch 1024 --hidden 256 --depth 4 --arch mlp \
  --n_val 300 --verbose True
echo "[$TOY] training stage 2/2 done (checkpoints up to k=240)"
