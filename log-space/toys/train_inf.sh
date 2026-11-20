#!/bin/bash
# The seven d=8 production runs with infinite data (M=inf), stage by stage (step 1 of README.md):
#   stage 1: k=0..700, cosine outer lr          stage 2: k=700..1000, constant lr 5e-5 (--resume)
#   AR(1) only, stage 3: k=1000..2000, constant lr 5e-5 (--resume)
# Architectures from paths.py (ARCH): MLP-4 w512, except iso_diag (MLP-4 w1024) and ar1 (ResNet-4 w1024).
# Outputs in results/toys_M=-1/, logs in ../logs/. One run at a time; a stage whose last checkpoint is already on disk
# is skipped, so the script can be relaunched after an interruption (an interrupted stage restarts from its beginning).
#
#   bash train_inf.sh                  (from log-space/toys; master log ../logs/train_inf.log)
#   bash train_inf.sh factor_d8_inf    (only the toys given)
#   DRY=1 bash train_inf.sh            (print the commands, run nothing)
set -euo pipefail
cd "$(dirname "$0")"
source ../.venv/bin/activate
R=results/toys_M=-1; L=../logs
mkdir -p $R $L
[ -z "${DRY:-}" ] && exec > >(tee -a $L/train_inf.log) 2>&1

stage () {   # toy tag last_k log_suffix [toy_train args]
  T=$1; TAG=$2; K=$3; SUF=$4; shift 4; S=${T}_${TAG}
  if [ -f $R/${S}_k${K}.pt ]; then echo "=== $S up to k=$K already trained, skipped"; return; fi
  CMD="python toy_train.py --toy $T --tag $TAG $*"
  if [ -n "${DRY:-}" ]; then echo "$CMD > $L/${S}_train${SUF}.log 2>&1"; return; fi
  echo "=== $S up to k=$K  $(date -u)"
  $CMD > $L/${S}_train${SUF}.log 2>&1
}

run () {   # toy tag [architecture args of stage 1]
  T=$1; TAG=$2; shift 2; S=${T}_${TAG}
  stage $T $TAG 700 "" "$@"
  stage $T $TAG 1000 _k700-1000 --n_outer 1000 --resume $R/${S}_k700.pt
  if [ "$T" = ar1_d8_inf ]; then stage $T $TAG 2000 _k1000-2000 --n_outer 2000 --resume $R/${S}_k1000.pt; fi
}

TOYS=${*:-"iw_ri_d8_inf iw_nonri_d8_inf factor_d8_inf iso_diag_d8_inf ar1_d8_inf diag_d8_inf hetero_d8_inf"}
echo "=== start $(date -u)"
for T in $TOYS; do
  TAG=$(python -c "import paths; print(paths.tag('$T'))")
  case $T in
    iso_diag_d8_inf) run $T $TAG --hidden 1024 ;;
    ar1_d8_inf)      run $T $TAG --arch resnet --hidden 1024 ;;
    *)               run $T $TAG ;;
  esac
done
echo "=== all done $(date -u)"
