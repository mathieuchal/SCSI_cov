"""Eyes-open vs eyes-closed classification, cross-validated over ALL subjects.

Reads the per-fold outputs of `eeg_exp.py --cv_folds K --cv_fold F --select worstks` (each fold: an SC-SI prior and a logistic
regression trained on other subjects, checkpoint chosen on 10 validation subjects, then tested on the fold's held-out subjects),
pools the held-out predictions (every subject is tested exactly once) and reports:
  * pooled accuracy / NLL / ECE / selective accuracy,
  * per-fold accuracy (does the effect replicate across folds?),
  * per-subject accuracies and paired subject-level tests (bootstrap over subjects, Wilcoxon signed-rank, sign counts).

Usage: python eeg_cv_classif.py [tag_prefix] [n_folds]     e.g. eeg_w4_s0_k1.0_e0.65 5
"""
import json, sys
import numpy as np
from scipy import stats

B = sys.argv[1] if len(sys.argv) > 1 else "eeg_w4_s0_k1.0_e0.65"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
OURS = "Posterior-trained LR, posterior-averaged (ours)"
BASE = ["SCM -> LR (plug-in)", "OAS -> LR (plug-in)", "NLS -> LR (plug-in)"]

P, Y, S, FO, KB = [], [], [], [], []
for f in range(K):
    tag = f"{B}_cv{K}f{f}_wide"
    z = np.load(f"results/{tag}_arrays.npz", allow_pickle=True)
    names = list(z["prob_names"])
    P.append(np.clip(z["prob"], 1e-4, 1 - 1e-4)); Y.append(z["te_y"]); S.append(z["te_sub"]); FO.append(np.full(len(z["te_y"]), f))
    KB.append(json.load(open(f"results/{tag}_results.json"))["kbest"])
P, Y, S, FO = np.concatenate(P, 1), np.concatenate(Y), np.concatenate(S), np.concatenate(FO)
subs = np.unique(S); by = {s: np.where(S == s)[0] for s in subs}
print(f"folds={K}, checkpoints selected per fold: {KB}; subjects={len(subs)}, windows={len(Y)} ({len(Y)//len(subs)} per subject), all subjects tested once: {len(subs) == len(set(S))}")

M = {n: i for i, n in enumerate(names)}
short = {"SCM -> LR (plug-in)": "sample cov.", "OAS -> LR (plug-in)": "OAS", "NLS -> LR (plug-in)": "LW-NLS",
         "SCM-trained LR, posterior-averaged (ours)": "SC-SI (SCM-trained LR)", OURS: "SC-SI"}


def acc(p, idx, frac=1.0):
    o = idx[np.argsort(-np.abs(p[idx] - 0.5))]; k = max(int(frac * len(o)), 1)
    return ((p[o[:k]] > 0.5).astype(int) == Y[o[:k]]).mean()


def nll(p, idx): return -np.mean(np.where(Y[idx] == 1, np.log(p[idx]), np.log(1 - p[idx])))


def ece(p, idx, bins=10):
    conf = np.where(p[idx] > 0.5, p[idx], 1 - p[idx]); ok = ((p[idx] > 0.5).astype(int) == Y[idx]).astype(float); e = 0.0
    for a, b in zip(np.linspace(0.5, 1, bins + 1)[:-1], np.linspace(0.5, 1, bins + 1)[1:]):
        m = (conf > a) & (conf <= b)
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e


allidx = np.arange(len(Y))
print("\n== pooled over all subjects ==")
print(f"{'method':26s} {'acc':>7s} {'acc@80%':>8s} {'acc@50%':>8s} {'NLL':>7s} {'ECE':>7s}")
for n, lab in short.items():
    p = P[M[n]]; print(f"{lab:26s} {acc(p, allidx):7.3f} {acc(p, allidx, .8):8.3f} {acc(p, allidx, .5):8.3f} {nll(p, allidx):7.3f} {ece(p, allidx):7.3f}")

print("\n== per-fold accuracy (each fold = different held-out subjects and a different trained prior) ==")
print(f"{'':26s}" + "".join(f"  fold{f}" for f in range(K)))
for n, lab in short.items():
    print(f"{lab:26s}" + "".join(f"  {acc(P[M[n]], np.where(FO == f)[0]):.3f}" for f in range(K)))
gain = [[acc(P[M[OURS]], np.where(FO == f)[0]) - acc(P[M[b]], np.where(FO == f)[0]) for f in range(K)] for b in BASE]
print("SC-SI minus plug-in, folds where SC-SI is ahead:", {short[b]: f"{int((np.array(g) > 0).sum())}/{K}" for b, g in zip(BASE, gain)})

print("\n== paired, subject-level (ours = SC-SI posterior-averaged; positive acc / negative NLL = better) ==")
rng = np.random.default_rng(0); NB = 5000
per_sub = {n: np.array([((P[M[n]][by[s]] > 0.5).astype(int) == Y[by[s]]).mean() for s in subs]) for n in names}
out = {"folds": K, "kbest": KB, "n_subjects": int(len(subs)), "n_windows": int(len(Y)), "methods": {}}
for b in BASE:
    d = per_sub[OURS] - per_sub[b]
    w = stats.wilcoxon(d[d != 0]) if (d != 0).any() else None
    bs = [acc(P[M[OURS]], idx := np.concatenate([by[s] for s in rng.choice(subs, len(subs))]), 1.0) - acc(P[M[b]], idx, 1.0) for _ in range(NB)]
    bs50 = [acc(P[M[OURS]], idx, .5) - acc(P[M[b]], idx, .5) for idx in (np.concatenate([by[s] for s in rng.choice(subs, len(subs))]) for _ in range(NB))]
    bsn = [nll(P[M[OURS]], idx) - nll(P[M[b]], idx) for idx in (np.concatenate([by[s] for s in rng.choice(subs, len(subs))]) for _ in range(NB))]
    ci = lambda v: f"{np.mean(v):+.3f} [{np.percentile(v, 2.5):+.3f},{np.percentile(v, 97.5):+.3f}]"
    print(f"vs {short[b]:12s} acc {ci(bs)}  | acc@50% {ci(bs50)}  | NLL {ci(bsn)}  | subjects better/worse/tied {int((d > 0).sum())}/{int((d < 0).sum())}/{int((d == 0).sum())}  Wilcoxon p={w.pvalue:.1e}")
    out["methods"][short[b]] = {"acc_diff": [float(np.mean(bs)), *map(float, np.percentile(bs, [2.5, 97.5]))], "acc50_diff": [float(np.mean(bs50)), *map(float, np.percentile(bs50, [2.5, 97.5]))],
                                "nll_diff": [float(np.mean(bsn)), *map(float, np.percentile(bsn, [2.5, 97.5]))], "wilcoxon_p": float(w.pvalue) if w else None,
                                "subjects_better": int((d > 0).sum()), "subjects_worse": int((d < 0).sum())}
print(f"\nper-subject accuracy, mean +- s.e. over {len(subs)} subjects: " + ", ".join(f"{short[n]} {per_sub[n].mean():.3f}+-{per_sub[n].std(ddof=1)/np.sqrt(len(subs)):.3f}" for n in short))

fr = np.linspace(0.2, 1.0, 33)
out["curves"] = {"fraction": fr.tolist(), **{short[n]: [float(acc(P[M[n]], allidx, f)) for f in fr] for n in short}}
out["pooled"] = {short[n]: {"acc": float(acc(P[M[n]], allidx)), "acc80": float(acc(P[M[n]], allidx, .8)), "acc50": float(acc(P[M[n]], allidx, .5)),
                            "nll": float(nll(P[M[n]], allidx)), "ece": float(ece(P[M[n]], allidx))} for n in short}
json.dump(out, open(f"results/{B}_cv{K}_wide_classification.json", "w"), indent=1)
print(f"saved results/{B}_cv{K}_wide_classification.json")
