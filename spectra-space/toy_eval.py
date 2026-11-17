"""Evaluate factor-space SC-SI (Cholesky-factor / ODE-flow-matching variant) against the reference posterior
of a toy model, next to a fitted inverse-Wishart baseline.

The reference-posterior CACHE (results/toys/{toy.name}_ref.pt, local to this directory) is reused across every
checkpoint/tag evaluated for the same toy, instead of recomputing an expensive HMC/exact reference each time:
the TRUE posterior doesn't depend on which learner is being evaluated, so every tagged run's results stay
directly, apples-to-apples comparable. (This directory used to share that cache with a sibling toys/ project
one level up -- no longer the case: everything this project needs now lives under factor-space/ itself.)"""
import argparse, json, math, os, time
import numpy as np, torch
from scipy import stats as sps
from scsi import SCSI, SCSIConfig, Chol, Spec, ResNetDrift, sample_covariance_from_factors, DT
from model import make_drift
from toys import make_toy, SpikedModelToy, IsoDiagToy  # this directory's own copies -- self-contained, no
from covutils import IWPrior                            # cross-directory scsi.py swap needed anymore
from run_paths import find_run_file                     # per-encoding results/toys/{chol,spectral}/ routing

STATS = ["logdet", "logcond", "top_share", "ldv_top", "ldv_bot", "corr01"]
LABELS = {"logdet": "log det C", "logcond": "log condition number", "top_share": "top-eigenvalue share",
          "ldv_top": "log dir. variance (top emp. eigvec)", "ldv_bot": "log dir. variance (bottom emp. eigvec)",
          "corr01": "correlation C01"}


def make_testset(toy, n=128, seed=777):
    """Same (C, Ce) as toys/toy_eval.py's make_testset given the same seed -- see module docstring."""
    g = torch.Generator().manual_seed(seed)
    Cho = Chol(toy.d)
    C = toy.sample_prior(n, g)
    Sigma = Cho.encode(C)
    Ce = sample_covariance_from_factors(Sigma, toy.N, g)
    return C, Ce


def stats_of(C, Ce):
    """C: (M,J,d,d), Ce: (M,d,d) -> dict of (M,J) tensors. Pure eigendecomposition diagnostics of a covariance
    matrix -- identical to toys/toy_eval.py's, no dependency on how C was produced (Sigma@Sigma.T or expm(log))."""
    w, V = torch.linalg.eigh(Ce)
    vt, vb = V[..., -1], V[..., 0]
    we = torch.linalg.eigvalsh(C)
    dv = lambda v: torch.einsum("mi,mjik,mk->mj", v, C, v)
    return {"logdet": torch.linalg.slogdet(C)[1], "logcond": torch.log(we[..., -1] / we[..., 0]),
            "top_share": we[..., -1] / we.sum(-1), "ldv_top": torch.log(dv(vt)), "ldv_bot": torch.log(dv(vb)),
            "corr01": C[..., 0, 1] / torch.sqrt(C[..., 0, 0] * C[..., 1, 1])}


def rhat(x):
    nk = x.shape[-1]
    m = x.mean(-1); W = x.var(-1, unbiased=True).mean(-1); B = nk * m.var(-1, unbiased=True)
    return torch.sqrt(((nk - 1) / nk * W + B / nk) / W)


WARMUP = {}  # see toys/toy_eval.py's comment: the generic-LAPACK bug is fixed (venv_paper); default warmup=1500 is fine


def get_reference(toy, Ce, n_ref, cache):
    """Identical to toys/toy_eval.py's get_reference -- called with cache pointing at the SHARED
    results/toys/{toy.name}_ref.pt (one directory up) so the expensive HMC/exact reference is computed (and
    cached) at most once across both the toys/ and factor_space/ campaigns."""
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
        idx = torch.randperm(Cs.shape[1], generator=g).to(Cs.device); Cs = Cs[:, idx]
    diag["sec"] = time.time() - t0
    torch.save({"Cs": Cs.float().cpu(), "diag": diag}, cache)
    return Cs.to(orig_device), diag


def make_pd_toy(kind, p, N, r=5):
    name = f"{kind}_d{p}_inf"
    if kind == "spiked":
        return SpikedModelToy(name, p, N, -1, r=r)
    return IsoDiagToy(name, p, N, -1)


def load_scsi(toy, path, threads=2):
    """Load a factor-space SCSI model from a checkpoint saved by this directory's toy_train.py. SIX
    normalisation tensors (mu_y/sd_y, mu_ce/sd_ce, velocity_loc/scale), not two -- see Drift's docstring in
    scsi.py (unlike toys/scsi.py, where I/Y-space and Ye/Ce-space are the same log-cov space, one pair suffices).

    Some d=8 toy checkpoints predate the "arch" field and model.Drift's ResBlock-era body -- their
    net.0/2/4/6/8.* state_dict keys are a plain flat Sequential, not the in_proj/blocks/out_proj layout
    model.ResNetDrift uses. The stored "arch" field can't disambiguate these two on its own: some older
    ResNetDrift-era checkpoints (e.g. the d16 "resnet10" tag) still have arch=="mlp" literally stored,
    from back when that string meant the ResBlock body, not today's flat one -- so "mlp" is ambiguous
    and the body style is detected directly from the state_dict's own keys instead, for both cases.
    transformer/transformer_eig/structured state_dicts don't share keys with either MLP body, so their
    stored "arch" field is unambiguous and kept as-is."""
    z = torch.load(path, map_location="cpu")
    state = z["state"]
    if "net.0.weight" in state:
        arch = "mlp"                                    # model.Drift: flat Sequential body
    elif "in_proj.weight" in state:
        arch = "resnet"                                  # any arch not matching mlp/transformer*/structured
    else:                                                 # -> make_drift's fallback, model.ResNetDrift
        arch = z.get("arch", "mlp")
    encoding = z.get("encoding", "chol")  # old checkpoints predate --encoding -- default "chol" matches their actual chart
    sampler = z.get("sampler", "ode")    # old checkpoints predate --sampler -- default "ode" matches their actual training
    cfg = SCSIConfig(N=toy.N, d=toy.d, threads=threads, hidden=z.get("hidden", 256), depth=z.get("depth", 4),
                      arch=arch, nhead=z.get("nhead", 8), encoding=encoding, sampler=sampler)
    m = SCSI(cfg)
    p = (Chol(toy.d) if encoding == "chol" else Spec(toy.d)).p
    m.ema_net = make_drift(cfg.arch, p, z["mu_y"], z["sd_y"], z["mu_ce"], z["sd_ce"],
                            z["velocity_loc"], z["velocity_scale"], hidden=cfg.hidden, depth=cfg.depth,
                            d=cfg.d, nhead=cfg.nhead, N=cfg.N).to(m.device)
    m.ema_net.load_state_dict(z["state"])
    m.ema_net.eval()            # loaded checkpoint is only ever sampled from, never trained further
    return m, z["k"]


def sigma_to_cov(Sigma: torch.Tensor) -> torch.Tensor:
    """sample_posterior returns Sigma (factor space), not C: this is Sigma@Sigma.T, a plain matrix product
    (NOT a Cholesky decomposition), so calling it as often as needed downstream is free."""
    Sigma = Sigma.to(DT)
    return Sigma @ Sigma.transpose(-1, -2)


def w1_norm(x, ref):                           # per-observation W1 / sd(ref)
    """NaN (not inf) when ref[m] has zero variance -- e.g. corr01 for a strictly-diagonal-covariance
    toy (diag), where every posterior draw has C[0,1]==0 identically, so there's no scale to normalize
    by. NaN lets callers nanmean/nanignore it instead of an inf silently poisoning every downstream
    average (seen: diag_d8_inf's whole W1-mean curve coming out as +inf at every single checkpoint)."""
    sd = ref.std(axis=-1)
    out = np.array([sps.wasserstein_distance(x[m], ref[m]) for m in range(x.shape[0])])
    return np.where(sd > 0, out / np.where(sd > 0, sd, 1.0), np.nan)


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
    ap.add_argument("--toy", required=True, help="TOYS registry name, or 'spiked'/'iso_diag' for a direct "
                                                   "(--p/--N/--r) construction bypassing the registry")
    ap.add_argument("--p", type=int, default=8, help="dimension (only used with --toy spiked/iso_diag)")
    ap.add_argument("--N", type=int, default=40, help="(only used with --toy spiked/iso_diag)")
    ap.add_argument("--r", type=int, default=5, help="rank, spiked only (only used with --toy spiked/iso_diag)")
    ap.add_argument("--n_test", type=int, default=128)
    ap.add_argument("--n_ref", type=int, default=2048)
    ap.add_argument("--J", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ref_only", action="store_true")
    ap.add_argument("--m_inf", type=int, default=5000, help="IW-fit ensemble size when toy.M == -1 (infinite-data mode)")
    ap.add_argument("--tag", default="", help="suffix identifying which factor_space/toy_train.py run to evaluate "
                                                "(matches that script's --tag); empty = untagged run")
    ap.add_argument("--agg", choices=["median", "mean"], default="mean",
                     help="aggregation over test tasks for the per-statistic W1 (and its sampling-noise "
                          "floor): mean (default -- matches this script's historical, pre-'--agg' "
                          "behavior exactly, unlike toy_W1_mean.py which has always defaulted to median) "
                          "or outlier-robust median. Does not affect sd_ratio/bias_z/pit_ks/SBC-KS, "
                          "which are already mean- or rank-based.")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    AGG = np.nanmean if a.agg == "mean" else np.nanmedian
    toy = make_pd_toy(a.toy, a.p, a.N, a.r) if a.toy in ("spiked", "iso_diag") else make_toy(a.toy)
    stem = toy.name + (f"_{a.tag}" if a.tag else "")
    os.makedirs("results/toys", exist_ok=True)
    C_true, Ce = make_testset(toy, a.n_test)

    #Calculate statistics from true posterior -- cache local to this directory (see module docstring)
    ref_C, ref_diag = get_reference(toy, Ce, a.n_ref, f"results/toys/{toy.name}_ref.pt")
    print(f"reference ({'exact IW' if toy.exact else 'HMC'}): {ref_C.shape}  diag={ {k: v for k, v in ref_diag.items() if k != 'rhat'} }")
    if "rhat" in ref_diag:
        print("  R-hat (max over test obs):", {k: round(v['max'], 3) for k, v in ref_diag['rhat'].items()})
    if a.ref_only:
        raise SystemExit

    # ---- learners ---- #
    model_path = find_run_file(f"{stem}_model.pt")
    run_subdir = os.path.dirname(model_path)   # results/toys/{chol,spectral} -- write this run's outputs alongside it
    model, kbest = load_scsi(toy, model_path, a.threads)
    g = torch.Generator().manual_seed(99)
    Cs_si = sigma_to_cov(model.sample_posterior_sigma(Ce, a.J, gen=g, chunk=4000))  # caps peak memory (entry-grid archs scale O(d^2)); _sigma variant is encoding-agnostic (chol or spectral)

    gg = torch.Generator().manual_seed(1000)
    M_fit = toy.M if toy.M > 0 else a.m_inf
    Cho = Chol(toy.d)
    Sigma_fit = Cho.encode(toy.sample_prior(M_fit, gg))
    Ce_tr = sample_covariance_from_factors(Sigma_fit, toy.N, gg)

    #Fit an IW surrogate prior
    iw = IWPrior(Ce_tr, toy.N, iters=800)
    Cs_iw = iw.sample_posterior(Ce, a.J, gen=g)
    print(f"factor-space SC-SI checkpoint k={kbest};  IW-ML fit nu0={iw.nu0:.2f}")
    #Calculate Oracle, SCSI and WI-ML stats
    S = {"ref": stats_of(ref_C, Ce), "scsi": stats_of(Cs_si, Ce), "iwfit": stats_of(Cs_iw, Ce)}
    S = {k: {s: v.numpy() for s, v in d_.items()} for k, d_ in S.items()}
    #
    T = stats_of(C_true[:, None], Ce); SCM = stats_of(Ce[:, None], Ce)
    T = {s: v[:, 0].numpy() for s, v in T.items()}; SCM = {s: v[:, 0].numpy() for s, v in SCM.items()}

    R = a.n_ref // 2
    metrics = {}
    for s in STATS:
        ref = S["ref"][s]
        r = {"floor_w1": float(AGG(w1_norm(ref[:, :R], ref[:, R:2 * R])) if 2 * R <= ref.shape[1] else np.nan)}
        for nm in ("scsi", "iwfit"):
            x = S[nm][s]
            u = pit_vs_ref(x, ref)
            r[nm] = dict(w1=float(AGG(w1_norm(x, ref))), sd_ratio=float((x.std(1) / ref.std(1)).mean()),
                         bias_z=float(((x.mean(1) - ref.mean(1)) / ref.std(1)).mean()), pit_ks=ks_unif(u))
        r["scm_absz"] = float((np.abs(SCM[s] - ref.mean(1)) / ref.std(1)).mean())
        for nm in ("ref", "scsi", "iwfit"):
            ranks = (S[nm][s] < T[s][:, None]).mean(1)
            r[f"sbc_ks_{nm}"] = ks_unif(ranks)
        metrics[s] = r

    #Dump file
    json.dump({"toy": toy.name, "d": toy.d, "N": toy.N, "M": toy.M, "kbest": kbest, "n_test": a.n_test, "n_ref": a.n_ref, "J": a.J,
               "agg": a.agg, "ref_diag": ref_diag, "metrics": metrics}, open(f"{run_subdir}/{stem}_metrics.json", "w"), indent=1, default=float)

    arrs = {f"{nm}__{s}": S[nm][s].astype(np.float32) for nm in S for s in STATS}
    arrs.update({f"truth__{s}": T[s] for s in STATS}); arrs.update({f"scm__{s}": SCM[s] for s in STATS})
    np.savez(f"{run_subdir}/{stem}_stats.npz", **arrs)
    print(f"[W1 aggregated over test tasks by: {a.agg}]")
    print(f"{'stat':12s} {'floor':>6s} | SC-SI: W1 sd-ratio bias  PITks | IW-fit: W1 sd-ratio bias PITks | SCM|z|  | SBC-KS ref/scsi/iw")
    for s in STATS:
        r = metrics[s]
        print(f"{s:12s} {r['floor_w1']:6.3f} | {r['scsi']['w1']:6.3f} {r['scsi']['sd_ratio']:6.2f} {r['scsi']['bias_z']:+6.2f} {r['scsi']['pit_ks']:6.3f} | "
              f"{r['iwfit']['w1']:6.3f} {r['iwfit']['sd_ratio']:6.2f} {r['iwfit']['bias_z']:+6.2f} {r['iwfit']['pit_ks']:6.3f} | {r['scm_absz']:5.2f}  | "
              f"{r['sbc_ks_ref']:.3f}/{r['sbc_ks_scsi']:.3f}/{r['sbc_ks_iwfit']:.3f}")
