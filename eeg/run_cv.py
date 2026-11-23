"""The paper's EEG run, eeg_w4_s0_k1.0_e0.65_rv, end to end: the five folds of the subject-disjoint CV, then pooling and figures.

Stages (in this order; --stages picks a subset):
  exp        eeg_exp.py per fold: training (skipped if the checkpoints exist, unless --retrain), log-score checkpoint
             selection on the validation subjects, next-window evaluation and classification on the test subjects
  splithalf  eeg_exp2.py per fold, at the checkpoint selected by `exp` (read from {tag}_results.json, key kbest)
  pool       eeg_cv_classif.py and eeg_cv_pool_calib.py: pools the held-out folds (needs all five folds)
  figs       paper_figs.py fig_eeg, figs_classif.py 4, eeg_figs.py

Usage (from eeg/):
  python run_cv.py                                  # everything (GPU for exp and splithalf)
  python run_cv.py --folds 2 3 --stages exp splithalf   # some folds only, e.g. one per GPU
  python run_cv.py --stages pool figs               # pooling and figures only (no GPU)
"""
import argparse, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
K = 5
BASE = "eeg_w4_s0_k1.0_e0.65"                     # eeg_w{win}_s{seed}_k{kappa}_e{neff_scale}
RV, RVLS = f"{BASE}_rv", f"{BASE}_rvlogscore"
STAGES = ["exp", "splithalf", "pool", "figs"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--folds", type=int, nargs="+", default=list(range(K)), help="folds for exp and splithalf (default: all)")
ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
ap.add_argument("--retrain", action="store_true", help="retrain even if the checkpoints exist")
ap.add_argument("--device", default=None, help="passed to eeg_exp.py and eeg_exp2.py (default: cuda if available)")
a = ap.parse_args()


def run(*cmd):
    cmd = [sys.executable, *map(str, cmd)]
    print("\n>>", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)


def fold_tag(prefix, f):
    return f"{prefix}_cv{K}f{f}_wide"


dev = ["--device", a.device] if a.device else []
res = lambda *p: os.path.join(HERE, "results", *p)

for f in a.folds:
    if "exp" in a.stages:
        run("eeg_exp.py", "--win", 4, "--neff_scale", 0.65, "--hidden", 512, "--n_outer", 30,
            "--cv_folds", K, "--cv_fold", f, "--tag", f"_rv_cv{K}f{f}_wide", *(["--retrain"] if a.retrain else []), *dev)
    if "splithalf" in a.stages:
        kbest = json.load(open(res(f"{fold_tag(RV, f)}_results.json")))["kbest"]
        run("eeg_exp2.py", fold_tag(RV, f), kbest, 0.65, "--cv_folds", K, "--cv_fold", f, *dev)

if "pool" in a.stages:
    # the pooling scripts read the per-fold files under the _rvlogscore prefix (identical copies: log-score is the default rule)
    for f in range(K):
        for suf in ["results.json", "arrays.npz"]:
            shutil.copyfile(res(f"{fold_tag(RV, f)}_{suf}"), res(f"{fold_tag(RVLS, f)}_{suf}"))
    run("eeg_cv_classif.py", "--prefix", RVLS, "--folds", K)
    shutil.copyfile(res(f"{RVLS}_cv{K}_wide_classification.json"), res(f"{BASE}_cv{K}_wide_classification.json"))
    run("eeg_cv_pool_calib.py", "--results_prefix", RVLS, "--splithalf_prefix", RV, "--folds", K)

if "figs" in a.stages:
    run("paper_figs.py", "fig_eeg")
    run("figs_classif.py", 4)
    run("eeg_figs.py")
