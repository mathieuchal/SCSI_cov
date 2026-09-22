"""Finance experiment: dynamic selection of meaningful eigendirections from a covariance *posterior*.

Data   : Ken French 48 industry portfolios, daily value-weighted returns 1926-2026 (zero-mean model).
Task   : (63-day window, random basket of d=12 industries fully observed in the window).
Prior  : learned by SC-SI from PAST windows only (non-overlapping, several random baskets per window),
         refitted on three expanding folds; test months 1995-2004 / 2005-2014 / 2015-2026.
Decision: number K of factors of the PCA factor model  C_K = sum_{k<=K} l_k u_k u_k^T + s_K^2 (I - P_K)
          * plug-in rules : Gavish-Donoho hard threshold on the sample spectrum, fixed K in {1,2,3}, K = d (SCM)
          * Bayes rule    : K minimising the *posterior expected Stein loss*  E[L_S(C_K(Ce), C) | Ce]
          Also: posterior-mean / Stein-optimal covariance, IW conjugate baseline, OAS shrinkage.
Scores : out-of-sample (next 21 days) min-variance-portfolio volatility, predictive log-score, calibration.
"""
import argparse, json, math, os, time
import numpy as np, pandas as pd
import torch

from scsi import SCSI, SCSIConfig, Sym, Drift, wishart_channel, DT
from covutils import (IWPrior, oas_shrink, nls_shrink, wishart_logscore, posterior_predictive_logscore, directional_pit,
                      coverage_from_pit, bootstrap_ci)

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=12)
ap.add_argument("--nw", type=int, default=63)
ap.add_argument("--nf", type=int, default=21)
ap.add_argument("--paths", type=int, default=8)
ap.add_argument("--subsets_per_window", type=int, default=16)
ap.add_argument("--n_outer", type=int, default=15)
ap.add_argument("--steps_first", type=int, default=4000)
ap.add_argument("--steps_outer", type=int, default=800)
ap.add_argument("--neff_scale", type=float, default=None, help="if None, calibrated on validation fold with the IW baseline")
ap.add_argument("--J", type=int, default=256)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--folds", default="1995,2005,2015")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="")
args = ap.parse_args()
torch.set_num_threads(args.threads)
d, NW, NF, J = args.d, args.nw, args.nf, args.J
TAG = f"fin_d{d}_nw{NW}_s{args.seed}{args.tag}"
os.makedirs("results", exist_ok=True)

df = pd.read_pickle("data/ff48_vw_daily.pkl")
R = df.values.astype(np.float64)                      # % daily returns
dates = df.index
T, A = R.shape
S_ = Sym(d)
year = dates.year.values
first_idx = lambda y: int(np.searchsorted(year, y))    # first row of calendar year y


def window_ok(e, assets, nf=NF):
    seg = R[e - NW + 1: e + 1 + nf, assets]
    return not np.isnan(seg).any()


def scm(e, assets, lo=None, hi=None):
    X = R[e - NW + 1: e + 1, assets] if lo is None else R[lo:hi, assets]
    return X.T @ X / len(X)


def train_tasks(train_end_idx, rng):
    """non-overlapping windows ending before train_end_idx; several random baskets per window"""
    out = []
    for e in range(NW - 1, train_end_idx - 1, NW):
        seg = R[e - NW + 1: e + 1]
        avail = np.where(~np.isnan(seg).any(0))[0]
        if len(avail) < d:
            continue
        for _ in range(args.subsets_per_window):
            a = np.sort(rng.choice(avail, d, replace=False))
            out.append(scm(e, a))
    return torch.tensor(np.stack(out), dtype=DT)


def test_paths(train_end_idx, test_end_idx, rng):
    """B fixed baskets ('paths') drawn from assets with data over the whole fold; monthly forecasts."""
    lo = train_end_idx - 2 * 252
    cols = np.where(~np.isnan(R[lo:test_end_idx]).any(0))[0]
    baskets = [np.sort(rng.choice(cols, d, replace=False)) for _ in range(args.paths)]
    ends = list(range(train_end_idx, test_end_idx - NF, 21))
    return baskets, ends


# ------------------------------------------------------------------------------------------------ #
# factor-count rules
# ------------------------------------------------------------------------------------------------ #
def gd_omega(beta):
    return 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43


def gd_K(l, N):
    """Gavish-Donoho (2014) optimal hard threshold, unknown noise level.  l: (...,d) ascending eigenvalues."""
    beta = l.shape[-1] / N
    thr = gd_omega(beta) ** 2 * l.median(-1).values
    return (l > thr[..., None]).sum(-1)


def pca_factor_family(Ce):
    """all rank-K factor covariances C_K (K=0..d) built from Ce.  Returns (...,d+1,d,d)."""
    l, U = torch.linalg.eigh(Ce)                       # ascending
    l = l.flip(-1); U = U.flip(-1)                     # descending
    out = []
    for K in range(d + 1):
        if K == d:
            lam = l
        else:
            s2 = l[..., K:].mean(-1, keepdim=True)
            lam = torch.cat([l[..., :K], s2.expand(*s2.shape[:-1], d - K)], -1)
        out.append((U * lam.unsqueeze(-2)) @ U.transpose(-1, -2))
    return torch.stack(out, -3)


def stein_loss(Ahat, C):
    """L_S(Ahat, C) = tr(Ahat^-1 C) - log det(Ahat^-1 C) - d ; broadcasts."""
    M = torch.linalg.solve(Ahat, C)
    return torch.einsum("...ii->...", M) - torch.linalg.slogdet(M)[1] - d


def posterior_K(Cal, draws):
    """Bayes rule: argmin_K E[L_S(C_K(Ce), C) | Ce] using posterior draws (M,J,d,d)."""
    fam = pca_factor_family(Cal)                                   # (M,d+1,d,d)
    risk = torch.stack([stein_loss(fam[:, K:K + 1], draws).mean(1) for K in range(d + 1)], 1)   # (M,d+1)
    return risk.argmin(1), risk


def mv_weights(C):
    x = torch.linalg.solve(C, torch.ones(C.shape[-1], 1, dtype=C.dtype).expand(*C.shape[:-2], -1, -1))
    return (x / x.sum((-1, -2), keepdim=True)).squeeze(-1)


# ------------------------------------------------------------------------------------------------ #
# effective sample-size calibration (IW proxy; cheap) on a validation fold that precedes all test folds
# ------------------------------------------------------------------------------------------------ #
def make_cases(baskets, ends):
    cal, fwd, meta = [], [], []
    for b, a in enumerate(baskets):
        for e in ends:
            if not window_ok(e, a):
                continue
            cal.append(scm(e, a)); fwd.append(scm(e, a, e + 1, e + 1 + NF)); meta.append((b, e))
    return torch.tensor(np.stack(cal), dtype=DT), torch.tensor(np.stack(fwd), dtype=DT), np.array(meta)


def calibrate_neff(rng):
    tr_end = first_idx(1985); va_end = first_idx(1995)
    Ce_tr = train_tasks(tr_end, rng)
    baskets, ends = test_paths(tr_end, va_end, rng)
    Cal, Fwd, meta = make_cases(baskets, ends)
    best, table = None, {}
    for sc in (0.3, 0.45, 0.6, 0.8, 1.0):
        N = max(int(round(NW * sc)), d + 2); Nf = max(int(round(NF * sc)), 3)
        iw = IWPrior(Ce_tr, N, iters=300)
        Cs = iw.sample_posterior(Cal, 128, gen=torch.Generator().manual_seed(1))
        dirs = torch.eye(d, dtype=DT)
        pit, _, _ = directional_pit(None, Fwd, Nf, Cs, dirs, gen=torch.Generator().manual_seed(2))
        cov = coverage_from_pit(pit)
        score = float(posterior_predictive_logscore(Fwd, Nf, Cs).mean())
        err = sum(abs(cov[k] - k) for k in cov)
        table[sc] = dict(cov=cov, calib_err=err, logscore=score, N=N)
        print(f"  neff_scale={sc}: N={N} coverage={cov}  calib_err={err:.3f} logscore={score:.3f}", flush=True)
        if best is None or err < table[best]["calib_err"]:
            best = sc
    return best, table


rng = np.random.default_rng(args.seed)
if args.neff_scale is None:
    print("calibrating N_eff scale on validation fold (train <1985, val 1985-1994) with the IW proxy ...")
    scale, calib_table = calibrate_neff(np.random.default_rng(args.seed + 100))
else:
    scale, calib_table = args.neff_scale, {}
N = max(int(round(NW * scale)), d + 2); NfE = max(int(round(NF * scale)), 3)
print(f"N_eff scale = {scale}  ->  N = {N} (window {NW}), N_forward = {NfE};  N/d = {N/d:.1f}")

# ------------------------------------------------------------------------------------------------ #
# folds
# ------------------------------------------------------------------------------------------------ #
fold_years = [int(y) for y in args.folds.split(",")]
fold_bounds = [(fold_years[i], fold_years[i + 1] if i + 1 < len(fold_years) else 2027) for i in range(len(fold_years))]
all_rows = []          # per-case records
path_records = []      # time series for figures
names = ["SCM", "OAS shrinkage", "LW nonlinear shrinkage", "PCA K=1", "PCA K=3", "PCA K=GD (plug-in)", "PCA K=Bayes (SC-SI)", "PCA K=Bayes (IW)",
         "IW post. mean", "SC-SI post. mean", "SC-SI Stein-opt."]
out_cases = {n: {"var": [], "stein_fwd": []} for n in names}
Ks = {"GD": [], "post": [], "postIW": []}
meta_all, fold_all, ll_scores = [], [], {"SCM": [], "OAS shrinkage": [], "LW nonlinear shrinkage": [], "IW": [], "SC-SI": []}
pit_store = {"OAS shrinkage": [], "LW nonlinear shrinkage": [], "IW": [], "SC-SI": [], "SCM": []}
risk_store = []

for fi, (y0, y1) in enumerate(fold_bounds):
    t0 = time.time()
    tr_end, te_end = first_idx(y0), min(first_idx(y1), T)
    frng = np.random.default_rng(args.seed + 10 * fi)
    Ce_tr = train_tasks(tr_end, frng)
    print(f"\n=== fold {fi}: train < {y0}, test {y0}-{y1-1}:  M={len(Ce_tr)} training covariances", flush=True)
    baskets, ends = test_paths(tr_end, te_end, frng)
    Cal, Fwd, meta = make_cases(baskets, ends)
    print(f"    test cases: {len(Cal)}", flush=True)

    ck = f"results/{TAG}_fold{fi}.pt"
    cfg = SCSIConfig(N=N, d=d, kappa=1.0, n_outer=args.n_outer, steps_first=args.steps_first, steps_outer=args.steps_outer,
                     seed=args.seed + fi, init="deconv", log_prior_inflate=1.5, threads=args.threads, n_recon=6)
    model = SCSI(cfg)
    if os.path.exists(ck):
        Ye = S_.encode(Ce_tr, 0.0)
        model.ema_net = Drift(S_.p, Ye.mean(0), Ye.std(0).clamp_min(1e-3), cfg.hidden, cfg.depth)
        model.ema_net.load_state_dict(torch.load(ck))
        print("    loaded", ck)
    else:
        model.fit(Ce_tr)
        torch.save(model.ema_net.state_dict(), ck)
    iw = IWPrior(Ce_tr, N, iters=600)
    g = torch.Generator().manual_seed(100 + fi)
    D_si = model.sample_posterior(Cal, J, gen=g)
    D_iw = iw.sample_posterior(Cal, J, gen=g)
    print(f"    sampling done ({time.time()-t0:.0f}s total)", flush=True)

    # ---- estimators ------------------------------------------------------------------------- #
    fam = pca_factor_family(Cal)
    lcal = torch.linalg.eigvalsh(Cal)
    K_gd = gd_K(lcal, N)
    K_si, risk_si = posterior_K(Cal, D_si)
    K_iw, _ = posterior_K(Cal, D_iw)
    est = {
        "SCM": Cal,
        "OAS shrinkage": oas_shrink(Cal, N),
        "LW nonlinear shrinkage": nls_shrink(Cal, N),
        "PCA K=1": fam[:, 1], "PCA K=3": fam[:, 3],
        "PCA K=GD (plug-in)": fam[torch.arange(len(Cal)), K_gd.clamp(0, d)],
        "PCA K=Bayes (SC-SI)": fam[torch.arange(len(Cal)), K_si],
        "PCA K=Bayes (IW)": fam[torch.arange(len(Cal)), K_iw],
        "IW post. mean": D_iw.mean(1),
        "SC-SI post. mean": D_si.mean(1),
        "SC-SI Stein-opt.": torch.linalg.inv(torch.linalg.inv(D_si).mean(1)),
    }
    for n_, C_hat in est.items():
        w = mv_weights(C_hat)
        out_cases[n_]["var"].append(torch.einsum("mi,mij,mj->m", w, Fwd, w).numpy())
        out_cases[n_]["stein_fwd"].append(stein_loss(C_hat, Fwd).numpy())
    Ks["GD"].append(K_gd.numpy()); Ks["post"].append(K_si.numpy()); Ks["postIW"].append(K_iw.numpy())
    meta_all.append(meta); fold_all.append(np.full(len(Cal), fi))
    risk_store.append(risk_si.numpy())
    # predictive log-scores of the forward window
    ll_scores["SCM"].append(wishart_logscore(Fwd, NfE, Cal).numpy())
    ll_scores["OAS shrinkage"].append(wishart_logscore(Fwd, NfE, est["OAS shrinkage"]).numpy())
    ll_scores["LW nonlinear shrinkage"].append(wishart_logscore(Fwd, NfE, est["LW nonlinear shrinkage"]).numpy())
    ll_scores["IW"].append(posterior_predictive_logscore(Fwd, NfE, D_iw).numpy())
    ll_scores["SC-SI"].append(posterior_predictive_logscore(Fwd, NfE, D_si).numpy())
    # calibration of forward directional variances (assets + equal-weight)
    ew = torch.ones(1, d, dtype=DT) / math.sqrt(d)
    dirs = torch.cat([torch.eye(d, dtype=DT), ew], 0)
    for nm, Cs in (("SC-SI", D_si), ("IW", D_iw), ("SCM", Cal[:, None].expand(-1, J, -1, -1)),
                   ("OAS shrinkage", est["OAS shrinkage"][:, None].expand(-1, J, -1, -1)),
                   ("LW nonlinear shrinkage", est["LW nonlinear shrinkage"][:, None].expand(-1, J, -1, -1))):
        pit, _, _ = directional_pit(None, Fwd, NfE, Cs, dirs, gen=torch.Generator().manual_seed(9))
        pit_store[nm].append(pit.numpy())
    # time series for the first path (figure): posterior of eigenvalue shares and P(meaningful)
    m0 = meta[:, 0] == 0
    Dm = D_si[torch.tensor(m0)]
    ev = torch.linalg.eigvalsh(Dm).flip(-1)                            # (m,J,d) descending
    share = ev / ev.sum(-1, keepdim=True)
    rho = ev / ev.median(-1, keepdim=True).values                       # signal-to-bulk ratio of population eigenvalues
    l_plug = torch.linalg.eigvalsh(Cal[torch.tensor(m0)]).flip(-1)
    rho_plug = l_plug / l_plug.median(-1, keepdim=True).values
    path_records.append(dict(
        end=meta[m0, 1], share_mean=share.mean(1)[:, :4].numpy(), share_lo=torch.quantile(share, 0.1, 1)[:, :4].numpy(),
        share_hi=torch.quantile(share, 0.9, 1)[:, :4].numpy(),
        p_meaningful=(rho > 3).to(DT).mean(1)[:, :5].numpy(), plug_meaningful=(rho_plug > 3).to(DT)[:, :5].numpy(),
        K_gd=K_gd[torch.tensor(m0)].numpy(), K_post=K_si[torch.tensor(m0)].numpy(),
        plug_share=(l_plug / l_plug.sum(-1, keepdim=True))[:, :4].numpy()))
    print(f"    fold done in {time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------------------------------------------ #
# summary
# ------------------------------------------------------------------------------------------------ #
meta_all = np.concatenate(meta_all); fold_all = np.concatenate(fold_all)
month_id = meta_all[:, 1]                                 # cases in the same month are dependent -> block bootstrap over months
results = {"N": N, "scale": scale, "calib_table": {str(k): v for k, v in calib_table.items()}, "n_cases": int(len(meta_all))}
var = {n_: np.concatenate(v["var"]) for n_, v in out_cases.items()}
stf = {n_: np.concatenate(v["stein_fwd"]) for n_, v in out_cases.items()}
base = var["SCM"]
print("\n=== out-of-sample min-variance portfolio (unconstrained), 21-day realised variance ===")
results["mv"] = {}
for n_ in names:
    vol = math.sqrt(252 * var[n_].mean())
    rel = np.log(var[n_] / var["OAS shrinkage"])           # log variance ratio vs OAS (negative = better)
    m, lo, hi = bootstrap_ci(rel, month_id)
    results["mv"][n_] = dict(ann_vol=vol, log_var_ratio_vs_OAS=[m, lo, hi], stein_fwd=float(np.median(stf[n_])))
    print(f"{n_:24s} ann.vol={vol:6.3f}   mean log(var/var_OAS)={m:+.4f} [{lo:+.4f},{hi:+.4f}]   median fwd Stein={np.median(stf[n_]):.3f}")
Kg, Kp, Kiw = np.concatenate(Ks["GD"]), np.concatenate(Ks["post"]), np.concatenate(Ks["postIW"])
results["K"] = {"GD_mean": float(Kg.mean()), "post_mean": float(Kp.mean()), "postIW_mean": float(Kiw.mean()),
                "GD_hist": np.bincount(Kg, minlength=d + 1).tolist(), "post_hist": np.bincount(Kp, minlength=d + 1).tolist()}
# stability: month-to-month change of K along each path
def flips(K):
    ch = []
    for b in range(args.paths):
        for fi in range(len(fold_bounds)):
            m = (meta_all[:, 0] == b) & (fold_all == fi)
            k = K[m][np.argsort(meta_all[m, 1])]
            ch.append(np.abs(np.diff(k)).mean() if len(k) > 1 else 0)
    return float(np.mean(ch))
results["K"]["GD_mean_abs_change"] = flips(Kg); results["K"]["post_mean_abs_change"] = flips(Kp)
print(f"\nK selected: GD mean={Kg.mean():.2f} (|dK|/month={results['K']['GD_mean_abs_change']:.2f}) ; "
      f"Bayes(SC-SI) mean={Kp.mean():.2f} (|dK|/month={results['K']['post_mean_abs_change']:.2f}) ; Bayes(IW) mean={Kiw.mean():.2f}")

print("\n=== predictive log-score of the forward 21-day covariance (vs SCM plug-in) ===")
results["logscore"] = {}
ls = {k: np.concatenate(v) for k, v in ll_scores.items()}
for k in ls:
    m, lo, hi = bootstrap_ci(ls[k] - ls["SCM"], month_id)
    results["logscore"][k] = [m, lo, hi]
    print(f"{k:16s} gain vs SCM = {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
print("\n=== calibration of forward directional variances (asset axes + equal weight) ===")
results["coverage"] = {}
for k, v in pit_store.items():
    p = np.concatenate(v)
    results["coverage"][k] = coverage_from_pit(torch.tensor(p))
    print(f"{k:16s} coverage 50/80/95 = " + "/".join(f"{x:.3f}" for x in results['coverage'][k].values()))
json.dump(results, open(f"results/{TAG}_results.json", "w"), indent=1, default=float)
np.savez(f"results/{TAG}_arrays.npz", meta=meta_all, fold=fold_all, Kg=Kg, Kp=Kp, Kiw=Kiw, risk=np.concatenate(risk_store),
         dates=dates.values.astype("datetime64[D]").astype(np.int64),
         **{f"var_{i}": var[n_] for i, n_ in enumerate(names)}, names=np.array(names),
         **{f"path{fi}_{k}": v for fi, rec in enumerate(path_records) for k, v in rec.items()})
print("saved", TAG)
