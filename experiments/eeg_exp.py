"""EEG experiment: posterior over covariance vs point estimates for short-window alpha-band EEG.

Protocol (App. I.7 of the paper, "validation through future noisy observations"):
  * task = (subject, condition, 4-s window); one covariance prior for the whole population of
    training subjects, learned from empirical covariances ONLY (no clean targets).
  * held-out subjects: posterior computed from ONE short calibration window; evaluated on the NEXT
    window of the same run through the predictive law  P(Ce_val | Ce_cal) = int K_N(. | C) Q(dC | Ce_cal).
  * subject-disjoint train / validation / test split; checkpoint (outer iteration) chosen on validation.
"""
import argparse, json, math, os, time
import numpy as np
import torch

from scsi import SCSI, SCSIConfig, Sym, wishart_channel, DT
from covutils import (IWPrior, oas_shrink, nls_shrink, wishart_logscore, posterior_predictive_logscore, directional_pit,
                      coverage_from_pit, airm_distance, tangent_features, inv_sqrt, bootstrap_ci)

ap = argparse.ArgumentParser()
ap.add_argument("--win", type=int, default=4)
ap.add_argument("--n_outer", type=int, default=20)
ap.add_argument("--steps_first", type=int, default=4000)
ap.add_argument("--steps_outer", type=int, default=800)
ap.add_argument("--kappa", type=float, default=1.0)
ap.add_argument("--neff_scale", type=float, default=1.0)
ap.add_argument("--J", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--tag", default="")
ap.add_argument("--retrain", action="store_true")
args = ap.parse_args()
torch.set_num_threads(args.threads)
TAG = f"eeg_w{args.win}_s{args.seed}_k{args.kappa}_e{args.neff_scale}{args.tag}"
os.makedirs("results", exist_ok=True)

# ------------------------------------------------------------------------------------------------ #
# data, effective N, subject split
# ------------------------------------------------------------------------------------------------ #
z = np.load(f"data/eeg_scm_{args.win}s.npz")
scm_win = torch.tensor(z["scm_win"], dtype=DT)          # (S, 2, W, d, d)   cond 0 = open, 1 = closed
scm_full = torch.tensor(z["scm_full"], dtype=DT)        # (S, 2, d, d)
Sn, _, W, d, _ = scm_win.shape
n_samp = int(z["n_win"])
ratio = float(np.median(z["neff_ratio"]))
N = int(round(ratio * n_samp * args.neff_scale))
print(f"subjects={Sn} windows/run={W} d={d}  samples/window={n_samp}  N_eff/n={ratio:.4f} -> N={N} (N/d={N/d:.1f})")
rng = np.random.default_rng(args.seed)
perm = rng.permutation(Sn)
n_tr, n_va = int(0.65 * Sn), int(0.10 * Sn)
tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
print(f"split subjects: train={len(tr)} val={len(va)} test={len(te)}")
Ce_train = scm_win[tr].reshape(-1, d, d)                # unlabeled pool of noisy covariances, both conditions
print("training ensemble M =", Ce_train.shape[0])
S_ = Sym(d)


def make_cases(subj_idx, with_next=True):
    """(calibration window j, validation window j+1) pairs from the same subject/condition run."""
    cal, val, meta = [], [], []
    for s in subj_idx:
        for c in range(2):
            for j in range(W - 1):
                cal.append(scm_win[s, c, j]); val.append(scm_win[s, c, j + 1]); meta.append((int(s), c, j))
    return torch.stack(cal), torch.stack(val), np.array(meta)


# ------------------------------------------------------------------------------------------------ #
# fit SC-SI (checkpoints on the outer iterations) and the IW baseline
# ------------------------------------------------------------------------------------------------ #
cfg = SCSIConfig(N=N, d=d, kappa=args.kappa, n_outer=args.n_outer, steps_first=args.steps_first,
                 steps_outer=args.steps_outer, seed=args.seed, init="deconv", log_prior_inflate=1.5,
                 threads=args.threads, n_recon=8)
model = SCSI(cfg)
ck_dir = f"results/{TAG}_ckpt"
os.makedirs(ck_dir, exist_ok=True)
CK = [0, 2, 5, 8, 12, 16, 20, 25, 30]
CK = [k for k in CK if k <= args.n_outer]
Cal_va, Val_va, meta_va = make_cases(va)
val_curve = {}


def cb(m, k):
    if k in CK:
        torch.save(m.ema_net.state_dict(), f"{ck_dir}/k{k}.pt")
        Cs = m.sample_posterior(Cal_va, 64)
        val_curve[k] = float(posterior_predictive_logscore(Val_va, N, Cs).mean())
        print(f"   [val] outer {k}: predictive log-score = {val_curve[k]:.4f}", flush=True)


have = all(os.path.exists(f"{ck_dir}/k{k}.pt") for k in CK) and os.path.exists(f"results/{TAG}_val.json")
if have and not args.retrain:
    val_curve = {int(k): v for k, v in json.load(open(f"results/{TAG}_val.json")).items()}
    # rebuild network shell with same normalisation stats
    Ye = S_.encode(Ce_train, 0.0)
    from scsi import Drift
    model.ema_net = Drift(S_.p, Ye.mean(0), Ye.std(0).clamp_min(1e-3), cfg.hidden, cfg.depth)
    print("loaded cached checkpoints")
else:
    t0 = time.time()
    model.fit(Ce_train, callback=cb)
    json.dump(val_curve, open(f"results/{TAG}_val.json", "w"))
    print(f"SC-SI fit: {time.time()-t0:.0f}s")

kbest = max(val_curve, key=val_curve.get)
print("validation curve:", {k: round(v, 3) for k, v in val_curve.items()}, " -> selected outer iteration", kbest)
model.ema_net.load_state_dict(torch.load(f"{ck_dir}/k{kbest}.pt"))

iw = IWPrior(Ce_train, N, iters=800)
print(f"IW baseline fitted by marginal likelihood: nu0={iw.nu0:.2f}, trace(Psi0)/(nu0-d-1)={float(iw.Psi0.trace())/(iw.nu0-d-1):.2f}")

# ------------------------------------------------------------------------------------------------ #
# test cases: predictive scores + calibration of directional variances
# ------------------------------------------------------------------------------------------------ #
Cal, Val, meta = make_cases(te)
subj_id = meta[:, 0]
J = args.J
gen = torch.Generator().manual_seed(1234)
draws = {
    "SC-SI (ours)": model.sample_posterior(Cal, J, gen=gen),
    "IW conjugate (ML-fitted)": iw.sample_posterior(Cal, J, gen=gen),
}
plugins = {"Sample cov.": Cal, "Linear shrinkage (OAS)": oas_shrink(Cal, N),
            "Nonlinear shrinkage (LW)": nls_shrink(Cal, N)}
# long-run proxy: run-average of the other windows (excludes the calibration window and the validation window)
Cref_full = []
for (s, c, j) in meta:
    idx = [i for i in range(W) if i not in (j, j + 1)]
    Cref_full.append(scm_win[s, c, idx].mean(0))
Cref_full = torch.stack(Cref_full)

# directions: channel axes + top-3 pooled PCs of the training ensemble
evals, evecs = torch.linalg.eigh(Ce_train.mean(0))
dirs = torch.cat([torch.eye(d, dtype=DT), evecs[:, -3:].T], 0)          # (d+3, d)
dir_names = list(map(str, z["channels"])) + ["PC1", "PC2", "PC3"]

results = {"N": N, "d": d, "kbest": kbest, "val_curve": val_curve, "n_test_cases": int(len(meta)),
           "n_test_subjects": int(len(te)), "methods": {}}
per_case_score = {}
pit_all = {}
width_all = {}
for name in list(plugins) + list(draws):
    if name in plugins:
        Cs = plugins[name][:, None].expand(-1, J, -1, -1)
        score = wishart_logscore(Val, N, plugins[name])
    else:
        Cs = draws[name]
        score = posterior_predictive_logscore(Val, N, Cs)
    pit, pred, obs = directional_pit(None, Val, N, Cs, dirs, gen=gen)
    per_case_score[name] = score.numpy()
    pit_all[name] = pit.numpy()
    lo, hi = torch.quantile(pred, 0.1, dim=-1), torch.quantile(pred, 0.9, dim=-1)
    width_all[name] = torch.log(hi / lo).numpy()
    cov = coverage_from_pit(pit)
    results["methods"][name] = {"logscore": None, "coverage": cov}

# oracle-ish reference: plug-in with the run-average of the other windows (uses much more data; not causal)
per_case_score["Long-run proxy (reference)"] = wishart_logscore(Val, N, Cref_full).numpy()

base = per_case_score["Sample cov."]
for name, sc in per_case_score.items():
    m, lo, hi = bootstrap_ci(sc, subj_id)
    dm, dlo, dhi = bootstrap_ci(sc - base, subj_id)
    results["methods"].setdefault(name, {"coverage": None})
    results["methods"][name].update(logscore=m, logscore_ci=[lo, hi], gain_vs_scm=dm, gain_ci=[dlo, dhi])
for name in pit_all:
    results["methods"][name]["median_log_width80"] = float(np.median(width_all[name]))
    # per-direction coverage at 80% for a breakdown
    p = pit_all[name]
    results["methods"][name]["cov80_by_dir"] = {dn: float(((p[:, i] > 0.1) & (p[:, i] < 0.9)).mean()) for i, dn in enumerate(dir_names)}

print("\n=== held-out subjects: predictive log-score (higher is better), gain vs sample covariance ===")
for name, r in results["methods"].items():
    print(f"{name:32s} logscore={r['logscore']:.3f} [{r['logscore_ci'][0]:.3f},{r['logscore_ci'][1]:.3f}]  "
          f"gain={r['gain_vs_scm']:+.3f} [{r['gain_ci'][0]:+.3f},{r['gain_ci'][1]:+.3f}]   cov(50/80/95)="
          + ("-" if r["coverage"] is None else "/".join(f"{v:.3f}" for v in r["coverage"].values())))

# ------------------------------------------------------------------------------------------------ #
# point-estimation sanity check: distance to long-run proxy
# ------------------------------------------------------------------------------------------------ #
def stein(A, C):
    Ai = torch.linalg.inv(A)
    M = Ai @ C
    return torch.einsum("...ii->...", M) - torch.linalg.slogdet(M)[1] - d


pt = {"Sample cov.": Cal, "Linear shrinkage (OAS)": plugins["Linear shrinkage (OAS)"],
      "Nonlinear shrinkage (LW)": plugins["Nonlinear shrinkage (LW)"],
      "IW post. mean": draws["IW conjugate (ML-fitted)"].mean(1),
      "SC-SI post. mean": draws["SC-SI (ours)"].mean(1),
      "SC-SI Stein-optimal": torch.linalg.inv(torch.linalg.inv(draws["SC-SI (ours)"]).mean(1))}
results["point"] = {}
for name, Ch in pt.items():
    st = stein(Ch, Cref_full).numpy()
    ai = airm_distance(Ch, Cref_full).numpy()
    results["point"][name] = {"stein": bootstrap_ci(st, subj_id), "airm": bootstrap_ci(ai, subj_id)}
    print(f"{name:26s} Stein loss={st.mean():.3f}  AIRM to long-run={ai.mean():.3f}")

# ------------------------------------------------------------------------------------------------ #
# uncertainty on a nonlinear functional: eyes-open vs eyes-closed Riemannian distance, per window pair
# ------------------------------------------------------------------------------------------------ #
def dist_experiment():
    outs = []
    gen2 = torch.Generator().manual_seed(77)
    rows = []
    for s in te:
        for j in range(W):
            rows.append((int(s), j))
    Ao = torch.stack([scm_win[s, 0, j] for s, j in rows]); Ac = torch.stack([scm_win[s, 1, j] for s, j in rows])
    ref_o = torch.stack([scm_win[s, 0, [i for i in range(W) if i != j]].mean(0) for s, j in rows])
    ref_c = torch.stack([scm_win[s, 1, [i for i in range(W) if i != j]].mean(0) for s, j in rows])
    ref = airm_distance(ref_o, ref_c).numpy()
    sub = np.array([s for s, _ in rows])
    res = {"reference_mean": float(ref.mean())}
    D = {}
    D["Sample cov."] = airm_distance(Ao, Ac)[:, None].numpy()
    D["Linear shrinkage (OAS)"] = airm_distance(oas_shrink(Ao, N), oas_shrink(Ac, N))[:, None].numpy()
    D["Nonlinear shrinkage (LW)"] = airm_distance(nls_shrink(Ao, N), nls_shrink(Ac, N))[:, None].numpy()
    # parametric bootstrap around the plug-in (frequentist uncertainty baseline)
    bo = wishart_channel(Ao[:, None].expand(-1, J, -1, -1).contiguous(), N, gen2)
    bc = wishart_channel(Ac[:, None].expand(-1, J, -1, -1).contiguous(), N, gen2)
    D["Param. bootstrap (SCM)"] = airm_distance(bo, bc).numpy()
    for nm, sampler in (("IW conjugate (ML-fitted)", lambda A: iw.sample_posterior(A, J, gen=gen2)),
                        ("SC-SI (ours)", lambda A: model.sample_posterior(A, J, gen=gen2))):
        D[nm] = airm_distance(sampler(Ao), sampler(Ac)).numpy()
    for nm, dd in D.items():
        pt_est = np.median(dd, 1)
        err = np.abs(pt_est - ref)
        r = {"point_abs_err": bootstrap_ci(err, sub), "point_bias": bootstrap_ci(pt_est - ref, sub)}
        if dd.shape[1] > 1:
            for lv in (0.8, 0.95):
                lo, hi = np.quantile(dd, (1 - lv) / 2, 1), np.quantile(dd, 1 - (1 - lv) / 2, 1)
                r[f"cov{int(lv*100)}"] = bootstrap_ci(((ref >= lo) & (ref <= hi)).astype(float), sub)
                r[f"width{int(lv*100)}"] = float(np.median(hi - lo))
        res[nm] = r
        print(f"{nm:28s} bias={r['point_bias'][0]:+.3f} |err|={r['point_abs_err'][0]:.3f}"
              + (f"  cov80={r['cov80'][0]:.3f} cov95={r['cov95'][0]:.3f} width80={r['width80']:.2f}" if "cov80" in r else ""))
    return res, D, ref, sub, rows


print("\n=== open-vs-closed Riemannian distance (reference = leave-window-out run averages) ===")
results["distance"], Ddist, ref_dist, sub_dist, rows_dist = dist_experiment()

# ------------------------------------------------------------------------------------------------ #
# downstream decision: eyes open vs closed from a single window, plug-in vs posterior averaging
# ------------------------------------------------------------------------------------------------ #
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

Cref = Ce_train.mean(0)
Iref = inv_sqrt(Cref)


def feats(C):
    return tangent_features(C, Iref, S_).numpy()


tr_C = scm_win[tr].reshape(-1, d, d)
tr_y = np.tile(np.repeat(np.arange(2), W), len(tr))
te_C = scm_win[te].reshape(-1, d, d)
te_y = np.tile(np.repeat(np.arange(2), W), len(te))
te_sub = np.repeat(te, 2 * W)
Jc = 64
gen3 = torch.Generator().manual_seed(5)
te_draws = model.sample_posterior(te_C, Jc, gen=gen3)                    # (M,Jc,d,d)
tr_draws = model.sample_posterior(tr_C, 4, gen=gen3)                     # 4 draws per training window
scaler = StandardScaler().fit(feats(tr_C))
def fit_lr(X, y, C=1.0):
    return LogisticRegression(C=C, max_iter=2000).fit(scaler.transform(X), y)
lrA = fit_lr(feats(tr_C), tr_y)
lrB = fit_lr(feats(tr_draws.reshape(-1, d, d)), np.repeat(tr_y, 4))
def bma(lr, draws):
    M_, J_ = draws.shape[:2]
    p = lr.predict_proba(scaler.transform(feats(draws.reshape(-1, d, d))))[:, 1]
    return p.reshape(M_, J_).mean(1)
prob = {"SCM -> LR (plug-in)": lrA.predict_proba(scaler.transform(feats(te_C)))[:, 1],
        "OAS -> LR (plug-in)": lrA.predict_proba(scaler.transform(feats(oas_shrink(te_C, N))))[:, 1],
        "NLS -> LR (plug-in)": lrA.predict_proba(scaler.transform(feats(nls_shrink(te_C, N))))[:, 1],
        "SCM-trained LR, posterior-averaged (ours)": bma(lrA, te_draws),
        "Posterior-trained LR, posterior-averaged (ours)": bma(lrB, te_draws)}


def ece(p, y, bins=10):
    conf = np.where(p > 0.5, p, 1 - p); pred = (p > 0.5).astype(int); acc = (pred == y).astype(float)
    e = 0.0
    edges = np.linspace(0.5, 1.0, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def selective_acc(p, y, frac):
    conf = np.abs(p - 0.5); order = np.argsort(-conf)[: int(frac * len(p))]
    return ((p[order] > 0.5).astype(int) == y[order]).mean()


print("\n=== eyes-open vs eyes-closed, single 4-s window, held-out subjects ===")
results["classif"] = {}
for name, p in prob.items():
    p = np.clip(p, 1e-4, 1 - 1e-4)
    nll = -np.mean(np.where(te_y == 1, np.log(p), np.log(1 - p)))
    acc_case = ((p > 0.5).astype(int) == te_y).astype(float)
    r = dict(acc=bootstrap_ci(acc_case, te_sub), nll=bootstrap_ci(-np.where(te_y == 1, np.log(p), np.log(1 - p)), te_sub),
             ece=float(ece(p, te_y)), sel_acc_80=float(selective_acc(p, te_y, 0.8)), sel_acc_50=float(selective_acc(p, te_y, 0.5)))
    results["classif"][name] = r
    print(f"{name:50s} acc={r['acc'][0]:.3f} NLL={r['nll'][0]:.3f} ECE={r['ece']:.3f} acc@80%cov={r['sel_acc_80']:.3f} acc@50%cov={r['sel_acc_50']:.3f}")

json.dump(results, open(f"results/{TAG}_results.json", "w"), indent=1, default=float)

# save arrays needed for figures
np.savez(f"results/{TAG}_arrays.npz", pit=np.stack([pit_all[k] for k in pit_all]), pit_names=np.array(list(pit_all)),
         dir_names=np.array(dir_names), width=np.stack([width_all[k] for k in width_all]),
         meta=meta, score=np.stack([per_case_score[k] for k in per_case_score]), score_names=np.array(list(per_case_score)),
         dist_ref=ref_dist, dist_names=np.array(list(Ddist)), dist_sub=sub_dist,
         **{f"dist_{i}": v for i, v in enumerate(Ddist.values())},
         prob=np.stack(list(prob.values())), prob_names=np.array(list(prob)), te_y=te_y, te_sub=te_sub)
print("saved", TAG)
