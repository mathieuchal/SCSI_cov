"""Evaluate one checkpoint of an SC-SI run (and the fitted inverse-Wishart baseline) against the reference posterior of
a toy model.

Statistics of C (all computed from posterior draws): logdet, log condition number, top-eigenvalue share,
log directional variance along the top / bottom empirical eigenvector of Ce, correlation C01/sqrt(C00 C11).
Metrics per statistic: normalised Wasserstein-1 distance to the reference posterior, dispersion ratio, mean shift,
KS distance of the reference-PIT, and simulation-based-calibration (SBC) ranks of the true value.

    python toy_eval.py --toy iw_ri_d8_inf              # the production run TAGS[toy], at its best_k
    python toy_eval.py --toy iw_ri_d8_inf --k 1000     # another checkpoint of that run

Writes the diagnostics results/toys_M=-1/{toy}_{tag}_k{k}eval_{metrics.json,stats.npz}, and -- when k is the run's best_k
(toy_train.py: the last checkpoint with --last_checkpoint, imposed in infinite-data mode; else the best callback validation) -- results/toys_M=-1/{toy}_W1.json:
  toy, n_test, n_ref, J, N, d, and for agg in ("mean", "median"): agg: {oracle, iw_ml, scsi}, each
  {value, detail: {stat: {w1, std}}} (w1_summary); oracle = sampling floor (W1 between the two halves of the reference
  draws), scsi = the run ("tag") at its best_k ("k"). The W1 trajectory of the run (every checkpoint of the grid) is
  computed by toy_W1_mean.py, with the same settings and the same w1_summary.
"""
import argparse, json, math, os, time
import numpy as np, torch
from scipy import stats as sps

from scsi import SCSI, SCSIConfig, Drift, Sym, wishart_channel, DT
from model import make_drift
from covutils import IWPrior
from toys import make_toy, IsoDiagToy

STATS = ["logdet", "logcond", "top_share", "ldv_top", "ldv_bot", "corr01"]
LABELS = {"logdet": "log det C", "logcond": "log condition number", "top_share": "top-eigenvalue share",
          "ldv_top": "log dir. variance (top emp. eigvec)", "ldv_bot": "log dir. variance (bottom emp. eigvec)",
          "corr01": "correlation C01"}


def make_testset(toy, n=128, seed=777):
    g = torch.Generator().manual_seed(seed)
    C = toy.sample_prior(n, g)
    Ce = wishart_channel(C, toy.N, g)
    return C, Ce


def stats_of(C, Ce):
    """C: (M,J,d,d), Ce: (M,d,d) -> dict of (M,J) tensors."""
    w, V = torch.linalg.eigh(Ce)
    #Top and bottom eigenvectors
    vt, vb = V[..., -1], V[..., 0]
    #Eigenvalues
    we = torch.linalg.eigvalsh(C)
    #Contraction of C against v
    dv = lambda v: torch.einsum("mi,mjik,mk->mj", v, C, v)
    return {"logdet": torch.linalg.slogdet(C)[1], "logcond": torch.log(we[..., -1] / we[..., 0]),
            "top_share": we[..., -1] / we.sum(-1), "ldv_top": torch.log(dv(vt)), "ldv_bot": torch.log(dv(vb)),
            "corr01": C[..., 0, 1] / torch.sqrt(C[..., 0, 0] * C[..., 1, 1])}


def stats_np(C, Ce):
    """stats_of on the GPU when there is one (batched 16x16 eigh: ~0.1 s vs ~2.5 s per learner on the CPU), as numpy."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {s: v.cpu().numpy() for s, v in stats_of(C.to(dev), Ce.to(dev)).items()}


def fit_iw_cached(Ce_tr, N, cache, iters=800):
    """IW-ML baseline fit, cached per toy: its training ensemble is seeded (1000) and identical for every
    checkpoint of a toy, so refitting it at every evaluation (~30 s, 800 Adam steps on the CPU) is pure waste."""
    if os.path.exists(cache):
        z = torch.load(cache, map_location="cpu")
        if z["M"] == Ce_tr.shape[0] and z["iters"] == iters:
            iw = IWPrior.__new__(IWPrior)
            iw.N, iw.d, iw.Psi0, iw.nu0, iw.loglik = N, Ce_tr.shape[-1], z["Psi0"], z["nu0"], z["loglik"]
            return iw
    iw = IWPrior(Ce_tr, N, iters=iters)
    torch.save({"M": Ce_tr.shape[0], "iters": iters, "Psi0": iw.Psi0, "nu0": iw.nu0, "loglik": iw.loglik}, cache)
    return iw


def rhat(x):
    nk = x.shape[-1]
    m = x.mean(-1); W = x.var(-1, unbiased=True).mean(-1); B = nk * m.var(-1, unbiased=True)
    return torch.sqrt(((nk - 1) / nk * W + B / nk) / W)


# HMC warmup=1500 (toys.py's default) does not converge for the "flat" (no Stiefel/Givens) parametrizations of
# FactorToy/HeteroskedasticToy -- see the comment in toy_oracle_log_scores.py for the diagnostic that found this.
WARMUP = {}  # the "needs more warmup" finding traced back to a buggy generic-LAPACK torch build (fixed by
             # venv_paper: torch==2.2.2+MKL, see requirements.txt) -- default warmup=1500 works fine now


def get_reference(toy, Ce, n_ref, cache):
    """Calculate stats by sampling true posterior."""
    if os.path.exists(cache):
        z = torch.load(cache, map_location="cpu")
        return z["Cs"].to(DT).to(Ce.device), z.get("diag", {})
    orig_device = Ce.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Ce = Ce.to(device)
    g = torch.Generator().manual_seed(4242)
    t0 = time.time()
    diag = {}
    if toy.exact:
        Cs = toy.ref_posterior(Ce, n_ref, g)
    else:
        n_chains = 8
        warmup = WARMUP.get(toy.name, 1500)
        Cs, diag = toy.hmc(Ce, n_chains=n_chains, n_keep=int(math.ceil(n_ref / n_chains)), thin=4, warmup=warmup, gen=g, return_diag=True)
        diag["warmup"] = warmup
        S = stats_of(Cs, Ce)
        diag["rhat"] = {k: dict(max=float(rhat(v.reshape(v.shape[0], n_chains, -1)).max()), median=float(rhat(v.reshape(v.shape[0], n_chains, -1)).median()))
                        for k, v in S.items()}
        idx = torch.randperm(Cs.shape[1], generator=g).to(Cs.device); Cs = Cs[:, idx]      # mix chains (RHS metrics already computed)
    diag["sec"] = time.time() - t0
    torch.save({"Cs": Cs.float().cpu(), "diag": diag}, cache)
    return Cs.to(orig_device), diag


def make_pd_toy(kind, p, N):
    """Construct an IsoDiagToy directly from CLI params (--toy iso_diag with --p/--N), bypassing the
    TOYS registry -- handy for a quick parameter sweep without editing toys.py. `load_scsi` below
    already works for these (IsoDiagToy aliases `.d = p`)."""
    name = f"{kind}_d{p}_inf"
    return IsoDiagToy(name, p, N, -1)


def load_scsi(toy, path, threads=2):
    """Load SCSI model"""
    z = torch.load(path, map_location="cpu")
    #Create config file -- hidden/depth from the checkpoint if present (models trained with
    #non-default capacity, e.g. via toy_train.py --hidden/--depth), else the SCSIConfig defaults.
    arch, nhead = z.get("arch", "mlp"), z.get("nhead", 8)
    cfg = SCSIConfig(N=toy.N, d=toy.d, kappa=1.0, threads=threads, hidden=z.get("hidden", 256), depth=z.get("depth", 4),
                      arch=arch, nhead=nhead)

    m = SCSI(cfg)
    m.ema_net = make_drift(arch, Sym(toy.d).p, z["mu"], z["sd"], hidden=cfg.hidden, depth=cfg.depth, d=toy.d, nhead=nhead, N=toy.N).to(m.device)
    m.ema_net.load_state_dict(z["state"])
    m.ema_net.eval()
    return m, z["k"]


def w1_norm(x, ref):                           # per-observation W1 / sd(ref)
    return np.array([sps.wasserstein_distance(x[m], ref[m]) / ref[m].std() for m in range(x.shape[0])])


# d=8 production runs: toy -> training tag (checkpoints results/toys_M=-1/{toy}_{tag}_k{k}.pt)
TAGS = {"iw_ri_d8_inf": "mlp_w512_d4", "iw_nonri_d8_inf": "mlp_w512_d4", "factor_d8_inf": "mlp_w512_d4",
        "hetero_d8_inf": "mlp_w512_d4", "diag_d8_inf": "mlp_w512_d4", "iso_diag_d8_inf": "mlp_w1024_d4",
        "ar1_d8_inf": "resnet_w1024_d4"}
GRID = [2, 4, 6, 8, 10, 12, 16, 20, 24, 60, *range(100, 1000, 40), 1000, *range(1100, 2001, 100)]   # evaluation grid
AGGS = ("mean", "median")


def w1_summary(stats, ref_stats):
    """W1 of a method against the reference, both {stat: (n_test, draws) array} (stats_np): per statistic, the per-task
    W1/sd(ref) (w1_norm; tasks with a non-finite value -- zero reference spread, e.g. corr01 for the diagonal toy --
    left out) aggregated by the mean and by the median, with the std over the tasks; value = average over the
    statistics that have tasks.  -> {agg: {"value", "detail": {stat: {"w1", "std"}}}}"""
    pt = {}
    for s in STATS:
        v = w1_norm(stats[s], ref_stats[s])
        pt[s] = v[np.isfinite(v)]
    res = {}
    for agg in AGGS:
        f = np.mean if agg == "mean" else np.median
        det = {s: {"w1": float(f(v)) if v.size else float("nan"), "std": float(v.std()) if v.size else float("nan")}
               for s, v in pt.items()}
        res[agg] = {"value": float(np.nanmean([e["w1"] for e in det.values()])), "detail": det}
    return res


def nanmedian0(v):                             # median over test cases, ignoring degenerate (sd(ref)=0 -> inf/nan) ones
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def pit_vs_ref(x, ref):
    out = np.empty_like(x, dtype=np.float64)
    for m in range(x.shape[0]):
        s = np.sort(ref[m]); out[m] = np.searchsorted(s, x[m], side="right") / len(s)
    return out


def ks_unif(u):
    u = np.sort(np.asarray(u).ravel()); n = len(u)
    return float(np.max(np.maximum(np.abs(u - np.arange(1, n + 1) / n), np.abs(u - np.arange(0, n) / n))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", required=True, help="TOYS registry name, or 'iso_diag' for a direct "
                                                   "(--p/--N) construction bypassing the registry")
    ap.add_argument("--p", type=int, default=8, help="dimension (only used with --toy iso_diag)")
    ap.add_argument("--N", type=int, default=40, help="(only used with --toy iso_diag)")
    ap.add_argument("--n_test", type=int, default=128)
    ap.add_argument("--n_ref", type=int, default=2048)
    ap.add_argument("--J", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=26)
    ap.add_argument("--ref_only", action="store_true")
    ap.add_argument("--chunk", type=int, default=131072, help="SC-SI sampler batch (draws per forward pass)")
    ap.add_argument("--m_inf", type=int, default=5000, help="IW-fit ensemble size when toy.M == -1 (infinite-data mode)")
    ap.add_argument("--tag", default="", help="toy_train.py --tag of the run to evaluate (default: the production run "
                                                "TAGS[toy]). The reference cache {toy}_ref.pt is keyed by the toy only.")
    ap.add_argument("--k", type=int, default=None, help="checkpoint to evaluate (default: the run's best_k)")
    ap.add_argument("--n_steps", type=int, default=0, help="SDE steps for SC-SI sampling (0 = the model's n_sde_steps, i.e. 64)")
    ap.add_argument("--ref_cache", default="", help="reference cache path (default results/toys_M=-1/{toy.name}_ref.pt); "
                                                    "the cache ignores --n_ref, so a different n_ref needs its own file")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    toy = make_pd_toy(a.toy, a.p, a.N) if a.toy == "iso_diag" else make_toy(a.toy)
    tag = a.tag or TAGS[toy.name]
    run = f"{toy.name}_{tag}"
    best_k = json.load(open(f"results/toys_M=-1/{run}_train.json"))["best_k"]
    k_eval = best_k if a.k is None else a.k
    stem = f"{run}_k{k_eval}eval"                                    # diagnostics files
    os.makedirs("results/toys_M=-1", exist_ok=True)
    C_true, Ce = make_testset(toy, a.n_test)

    #Calculate statistics from true posterior
    ref_C, ref_diag = get_reference(toy, Ce, a.n_ref, a.ref_cache or f"results/toys_M=-1/{toy.name}_ref.pt")
    print(f"reference ({'exact IW' if toy.exact else 'HMC'}): {ref_C.shape}  diag={ {k: v for k, v in ref_diag.items() if k != 'rhat'} }")

    if "rhat" in ref_diag:
        print("  R-hat (max over test obs):", {k: round(v['max'], 3) for k, v in ref_diag['rhat'].items()})
    if a.ref_only:
        raise SystemExit
    # ---- learners ---- #
    #Take the best model
    model, kbest = load_scsi(toy, f"results/toys_M=-1/{run}_k{k_eval}.pt", a.threads)
    g = torch.Generator().manual_seed(99)
    # SC-SI draws: noise from a device generator (no per-step host RNG + copy) and TF32 matmuls in the drift net
    # (~4x faster on an A100 for w1024; W1 unchanged to 4 decimals). Eval only -- training keeps full fp32.
    torch.backends.cuda.matmul.allow_tf32 = True
    gs = torch.Generator(device=model.device).manual_seed(99)
    Cs_si = model.sample_posterior(Ce, a.J, gen=gs, chunk=a.chunk, n_steps=a.n_steps or None)
    torch.backends.cuda.matmul.allow_tf32 = False

    gg = torch.Generator().manual_seed(1000)                                             
    M_fit = toy.M if toy.M > 0 else a.m_inf                                              
    Ce_tr = wishart_channel(toy.sample_prior(M_fit, gg), toy.N, gg)
    
    #Fit an IW surrogate prior
    iw = fit_iw_cached(Ce_tr, toy.N, f"results/toys_M=-1/{toy.name}_iwfit_ref.pt")
    #Sample posterior
    Cs_iw = iw.sample_posterior(Ce, a.J, gen=g)
    print(f"SC-SI checkpoint k={kbest};  IW-ML fit nu0={iw.nu0:.2f}")
    #Calculate posterior Oracle, SCSI and WI-ML stats
    S = {"ref": stats_np(ref_C, Ce), "scsi": stats_np(Cs_si, Ce), "iwfit": stats_np(Cs_iw, Ce)}
    # Calculate prior p(C) and sample p(Ce) stats
    T = stats_of(C_true[:, None], Ce); SCM = stats_of(Ce[:, None], Ce)
    T = {s: v[:, 0].numpy() for s, v in T.items()}; SCM = {s: v[:, 0].numpy() for s, v in SCM.items()}
    
    R = a.n_ref // 2
    metrics = {}
    for s in STATS:
        #Value of stat 's' for oracle estimator
        ref = S["ref"][s]
        # W1 aggregated over test cases two ways: mean ("w1", "floor_w1") and median ("w1_med", "floor_w1_med");
        # the median is robust to the few near-degenerate cases whose tiny sd(ref) blows up the normalised W1
        fl = w1_norm(ref[:, :R], ref[:, R:2 * R]) if 2 * R <= ref.shape[1] else None
        r = {"floor_w1": float(fl.mean()) if fl is not None else np.nan,
             "floor_w1_med": nanmedian0(fl) if fl is not None else np.nan}
        for nm in ("scsi", "iwfit"):
            x = S[nm][s]
            u = pit_vs_ref(x, ref)
            w = w1_norm(x, ref)
            r[nm] = dict(w1=float(w.mean()), w1_med=nanmedian0(w), sd_ratio=float((x.std(1) / ref.std(1)).mean()),
                         bias_z=float(((x.mean(1) - ref.mean(1)) / ref.std(1)).mean()), pit_ks=ks_unif(u))
        r["scm_absz"] = float((np.abs(SCM[s] - ref.mean(1)) / ref.std(1)).mean())
        # SBC of the truth in each posterior
        for nm in ("ref", "scsi", "iwfit"):
            ranks = (S[nm][s] < T[s][:, None]).mean(1)
            r[f"sbc_ks_{nm}"] = ks_unif(ranks)
        metrics[s] = r
    
    #Dump evaluation metric file
    json.dump({"toy": toy.name, "d": toy.d, "N": toy.N, "M": toy.M, "kbest": kbest, "n_test": a.n_test, "n_ref": a.n_ref, "J": a.J,
               "ref_diag": ref_diag, "metrics": metrics}, open(f"results/toys_M=-1/{stem}_metrics.json", "w"), indent=1, default=float)
    
    #
    arrs = {f"{nm}__{s}": S[nm][s].astype(np.float32) for nm in S for s in STATS}
    arrs.update({f"truth__{s}": T[s] for s in STATS}); arrs.update({f"scm__{s}": SCM[s] for s in STATS})
    
    #Save summary stats
    np.savez(f"results/toys_M=-1/{stem}_stats.npz", **arrs)

    # {toy}_W1.json: oracle (sampling floor), IW-ML and the run's best_k checkpoint
    if k_eval == best_k:
        ref = S["ref"]
        W = {"toy": toy.name, "n_test": a.n_test, "n_ref": a.n_ref, "J": a.J, "N": toy.N, "d": toy.d}
        fl = w1_summary({s: v[:, :R] for s, v in ref.items()}, {s: v[:, R:2 * R] for s, v in ref.items()})
        iw_s, si_s = w1_summary(S["iwfit"], ref), w1_summary(S["scsi"], ref)
        for agg in AGGS:
            W[agg] = {"oracle": fl[agg], "iw_ml": iw_s[agg], "scsi": {"tag": tag, "k": k_eval, **si_s[agg]}}
        json.dump(W, open(f"results/toys_M=-1/{toy.name}_W1.json", "w"), indent=1)
        print(f"{toy.name}_W1.json: " + "  ".join(f"{agg}: oracle {W[agg]['oracle']['value']:.4f} iw_ml "
              f"{W[agg]['iw_ml']['value']:.4f} scsi(k={k_eval}) {W[agg]['scsi']['value']:.4f}" for agg in AGGS))
    print(f"{'stat':12s} {'floor':>6s} | SC-SI: W1 W1med sd-ratio bias  PITks | IW-fit: W1 W1med sd-ratio bias PITks | SCM|z|  | SBC-KS ref/scsi/iw")
    for s in STATS:
        r = metrics[s]
        print(f"{s:12s} {r['floor_w1']:6.3f} | {r['scsi']['w1']:6.3f} {r['scsi']['w1_med']:6.3f} {r['scsi']['sd_ratio']:6.2f} {r['scsi']['bias_z']:+6.2f} {r['scsi']['pit_ks']:6.3f} | "
              f"{r['iwfit']['w1']:6.3f} {r['iwfit']['w1_med']:6.3f} {r['iwfit']['sd_ratio']:6.2f} {r['iwfit']['bias_z']:+6.2f} {r['iwfit']['pit_ks']:6.3f} | {r['scm_absz']:5.2f}  | "
              f"{r['sbc_ks_ref']:.3f}/{r['sbc_ks_scsi']:.3f}/{r['sbc_ks_iwfit']:.3f}")
