"""Pool the per-fold next-window (eeg_exp.py) and split-half (eeg_exp2.py) evaluations of a 5-fold CV run over all
subjects, in the same JSON layout paper_figs.fig_eeg reads for panels (a) and (b).

  * coverages: every subject contributes the same number of cases in every fold, so the pooled coverage is the
    subject-count-weighted mean of the per-fold coverages (identical to recomputing it on the pooled cases);
  * log-score gains vs sample covariance: per-case scores pooled over folds, 95% CI by bootstrap over subjects
    (covutils.bootstrap_ci), exactly as eeg_exp.py / eeg_exp2.py do within one split.

Usage: python eeg_cv_pool_calib.py [results_prefix] [splithalf_prefix] [K]
   e.g. python eeg_cv_pool_calib.py eeg_w4_s0_k1.0_e0.65_rvlogscore eeg_w4_s0_k1.0_e0.65_rv 5
Writes results/{results_prefix}_cv{K}_nextwindow.json and results/{results_prefix}_cv{K}_splithalf.json.
"""
import json, sys
import numpy as np
from covutils import bootstrap_ci

RP = sys.argv[1] if len(sys.argv) > 1 else "eeg_w4_s0_k1.0_e0.65_rvlogscore"
SP = sys.argv[2] if len(sys.argv) > 2 else "eeg_w4_s0_k1.0_e0.65_rv"
K = int(sys.argv[3]) if len(sys.argv) > 3 else 5
L = lambda p: json.load(open(p))


def wmean(dicts, w):
    """weighted mean of a list of {level: value} dicts."""
    return {lv: float(np.average([d[lv] for d in dicts], weights=w)) for lv in dicts[0]}


# ------------------------------------------------------------------ next window (eeg_exp.py)
R = [L(f"results/{RP}_cv{K}f{f}_wide_results.json") for f in range(K)]
A = [np.load(f"results/{RP}_cv{K}f{f}_wide_arrays.npz", allow_pickle=True) for f in range(K)]
w = [r["n_test_subjects"] for r in R]
names = list(A[0]["score_names"])
score = np.concatenate([a["score"] for a in A], 1)
subj = np.concatenate([a["meta"][:, 0] for a in A])
base = score[names.index("Sample cov.")]
nw = {"source": f"{RP}_cv{K}f*_wide pooled over {K} folds", "kbest": [r["kbest"] for r in R], "n_test_subjects": int(len(np.unique(subj))),
      "n_test_cases": int(len(subj)), "methods": {}}
for i, n in enumerate(names):
    m, lo, hi = bootstrap_ci(score[i], subj); dm, dlo, dhi = bootstrap_ci(score[i] - base, subj)
    cov = [R[f]["methods"][n]["coverage"] for f in range(K)]
    nw["methods"][n] = {"logscore": m, "logscore_ci": [lo, hi], "gain_vs_scm": dm, "gain_ci": [dlo, dhi],
                        "coverage": None if cov[0] is None else wmean(cov, w)}
json.dump(nw, open(f"results/{RP}_cv{K}_nextwindow.json", "w"), indent=1)

# ------------------------------------------------------------------ split half (eeg_exp2.py)
S, SA = [], []
for f in range(K):
    k = R[f]["kbest"]
    S.append(L(f"results/{SP}_cv{K}f{f}_wide_splithalf_k{k}.json"))
    SA.append(np.load(f"results/{SP}_cv{K}f{f}_wide_splithalf_k{k}_arrays.npz", allow_pickle=True))
w2 = [s["n_test_subjects"] for s in S]
sn = list(SA[0]["score_names"])
sc = np.concatenate([a["scores"] for a in SA], 1)
ssub = np.concatenate([a["meta"][:, 0] for a in SA])
sbase = sc[sn.index("Sample cov.")]
sh = {"source": f"{SP}_cv{K}f*_wide_splithalf pooled over {K} folds", "n_test_subjects": int(len(np.unique(ssub))), "n_cases": int(len(ssub)),
      "logscore": {}, "dir_cov": {}, "functional": {}, "distance": {}}
for i, n in enumerate(sn):
    sh["logscore"][n] = {"mean": bootstrap_ci(sc[i], ssub), "gain": bootstrap_ci(sc[i] - sbase, ssub)}
    sh["dir_cov"][n] = wmean([s["dir_cov"][n] for s in S], w2)
    sh["distance"][n] = {"cov": wmean([s["distance"][n]["cov"] for s in S], w2)}
for fn in S[0]["functional"]:
    sh["functional"][fn] = {n: {"cov": wmean([s["functional"][fn][n]["cov"] for s in S], w2)} for n in sn}
json.dump(sh, open(f"results/{RP}_cv{K}_splithalf.json", "w"), indent=1)

# ------------------------------------------------------------------ summary
print(f"next window: {nw['n_test_subjects']} subjects, {nw['n_test_cases']} cases; split-half: {sh['n_test_subjects']} subjects, {sh['n_cases']} cases")
print(f"{'method':28s} {'gain next-window':>24s} {'gain split-half':>24s} {'cov80 NW':>8s} {'dir cov80 SH':>12s}")
for n in sn:
    g = nw["methods"][n]; h = sh["logscore"][n]["gain"]
    print(f"{n:28s} {g['gain_vs_scm']:+7.2f} [{g['gain_ci'][0]:+6.2f},{g['gain_ci'][1]:+6.2f}] {h[0]:+7.2f} [{h[1]:+6.2f},{h[2]:+6.2f}] "
          f"{g['coverage']['0.8']:8.3f} {sh['dir_cov'][n]['0.8']:12.3f}")
print("split-half 80% coverage of functionals:")
for fn in sh["functional"]:
    print(f"  {fn:22s} " + "  ".join(f"{n.split(' ')[0]}={sh['functional'][fn][n]['cov']['0.8']:.3f}" for n in sn))
print(f"saved results/{RP}_cv{K}_nextwindow.json, results/{RP}_cv{K}_splithalf.json")
