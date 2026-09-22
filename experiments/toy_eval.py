"""Evaluate SC-SI (and a fitted inverse-Wishart baseline) against the reference posterior of a toy model.

Statistics of C (all computed from posterior draws): logdet, log condition number, top-eigenvalue share,
log directional variance along the top / bottom empirical eigenvector of Ce, correlation C01/sqrt(C00 C11).
Metrics per statistic: normalised Wasserstein-1 distance to the reference posterior, dispersion ratio, mean shift,
KS distance of the reference-PIT, and simulation-based-calibration (SBC) ranks of the true value.
"""
import argparse, json, math, os, time
import numpy as np, torch
from scipy import stats as sps

from scsi import SCSI, SCSIConfig, Drift, Sym, wishart_channel, DT
from covutils import IWPrior
from toys import make_toy

STATS = ["logdet", "logcond", "top_share", "ldv_top", "ldv_bot", "corr01"]
LABELS = {"logdet": "log det C", "logcond": "log condition number", "top_share": "top-eigenvalue share",
          "ldv_top": "log dir. variance (top emp. eigvec)", "ldv_bot": "log dir. variance (bottom emp. eigvec)",
          "corr01": "correlation C01"}


def make_testset(toy, n=128, seed=777):
    g = torch.Generator(device=toy.device).manual_seed(seed)
    C = toy.sample_prior(n, g)
    Ce = wishart_channel(C, toy.N, g)
    return C, Ce


def stats_of(C, Ce):
    """C: (M,J,d,d), Ce: (M,d,d) -> dict of (M,J) tensors."""
    w, V = torch.linalg.eigh(Ce)
    vt, vb = V[..., -1], V[..., 0]
    we = torch.linalg.eigvalsh(C)
    dv = lambda v: torch.einsum("mi,mjik,mk->mj", v, C, v)
    return {"logdet": torch.linalg.slogdet(C)[1], "logcond": torch.log(we[..., -1] / we[..., 0]),
            "top_share": we[..., -1] / we.sum(-1), "ldv_top": torch.log(dv(vt)), "ldv_bot": torch.log(dv(vb)),
            "corr01": C[..., 0, 1] / torch.sqrt(C[..., 0, 0] * C[..., 1, 1])}


def rhat(x):                                   # x: (M, nchain, nkeep)
    nk = x.shape[-1]
    m = x.mean(-1); W = x.var(-1, unbiased=True).mean(-1); B = nk * m.var(-1, unbiased=True)
    return torch.sqrt(((nk - 1) / nk * W + B / nk) / W)


def get_reference(toy, Ce, n_ref, cache):
    if os.path.exists(cache):
        z = torch.load(cache, map_location="cpu")
        return z["Cs"].to(DT).to(toy.device), z.get("diag", {})
    g = torch.Generator(device=toy.device).manual_seed(4242)
    t0 = time.time()
    diag = {}
    if toy.exact:
        Cs = toy.ref_posterior(Ce, n_ref, g)
    else:
        n_chains = 8
        Cs, diag = toy.hmc(Ce, n_chains=n_chains, n_keep=int(math.ceil(n_ref / n_chains)), thin=4, warmup=1500, gen=g, return_diag=True)
        S = stats_of(Cs, Ce)
        diag["rhat"] = {k: dict(max=float(rhat(v.reshape(v.shape[0], n_chains, -1)).max()), median=float(rhat(v.reshape(v.shape[0], n_chains, -1)).median()))
                        for k, v in S.items()}
        idx = torch.randperm(Cs.shape[1], device=toy.device, generator=g); Cs = Cs[:, idx]  # mix chains (RHS metrics already computed)
    diag["sec"] = time.time() - t0
    torch.save({"Cs": Cs.float().cpu(), "diag": diag}, cache)   # cached on CPU so it loads on any device later
    return Cs, diag


def load_scsi(toy, path, threads=2, device=None):
    device = device or toy.device
    z = torch.load(path, map_location="cpu")
    cfg = SCSIConfig(N=toy.N, d=toy.d, kappa=1.0, threads=threads, device=str(device))
    m = SCSI(cfg)
    m.ema_net = Drift(Sym(toy.d).p, z["mu"], z["sd"], cfg.hidden, cfg.depth).to(device)
    m.ema_net.load_state_dict(z["state"])
    return m, z["k"]


def w1_norm(x, ref):                           # per-observation W1 / sd(ref)
    return np.array([sps.wasserstein_distance(x[m], ref[m]) / ref[m].std() for m in range(x.shape[0])])


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
    ap.add_argument("--toy", required=True)
    ap.add_argument("--n_test", type=int, default=128)
    ap.add_argument("--n_ref", type=int, default=2048)
    ap.add_argument("--J", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ref_only", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    toy = make_toy(a.toy, device=dev)
    os.makedirs("results/toys", exist_ok=True)
    C_true, Ce = make_testset(toy, a.n_test)
    ref_C, ref_diag = get_reference(toy, Ce, a.n_ref, f"results/toys/{toy.name}_ref.pt")
    print(f"reference ({'exact IW' if toy.exact else 'HMC'}): {ref_C.shape}  diag={ {k: v for k, v in ref_diag.items() if k != 'rhat'} }")
    if "rhat" in ref_diag:
        print("  R-hat (max over test obs):", {k: round(v['max'], 3) for k, v in ref_diag['rhat'].items()})
    if a.ref_only:
        raise SystemExit
    # ---- learners ---- #
    model, kbest = load_scsi(toy, f"results/toys/{toy.name}_model.pt", a.threads, device=dev)
    g = torch.Generator(device=dev).manual_seed(99)
    Cs_si = model.sample_posterior(Ce, a.J, gen=g)
    gg = torch.Generator(device=dev).manual_seed(1000)                                   # same training ensemble as toy_train.py
    Ce_tr = wishart_channel(toy.sample_prior(toy.M, gg), toy.N, gg)
    iw = IWPrior(Ce_tr, toy.N, iters=800)
    Cs_iw = iw.sample_posterior(Ce, a.J, gen=g)
    print(f"SC-SI checkpoint k={kbest};  IW-ML fit nu0={iw.nu0:.2f}")
    S = {"ref": stats_of(ref_C, Ce), "scsi": stats_of(Cs_si, Ce), "iwfit": stats_of(Cs_iw, Ce)}
    T = stats_of(C_true[:, None], Ce); SCM = stats_of(Ce[:, None], Ce)
    S = {k: {s: v.cpu().numpy() for s, v in d_.items()} for k, d_ in S.items()}
    T = {s: v[:, 0].cpu().numpy() for s, v in T.items()}; SCM = {s: v[:, 0].cpu().numpy() for s, v in SCM.items()}
    R = a.n_ref // 2
    metrics = {}
    for s in STATS:
        ref = S["ref"][s]
        r = {"floor_w1": float(w1_norm(ref[:, :R], ref[:, R:2 * R]).mean() if 2 * R <= ref.shape[1] else np.nan)}
        for nm in ("scsi", "iwfit"):
            x = S[nm][s]
            u = pit_vs_ref(x, ref)
            r[nm] = dict(w1=float(w1_norm(x, ref).mean()), sd_ratio=float((x.std(1) / ref.std(1)).mean()),
                         bias_z=float(((x.mean(1) - ref.mean(1)) / ref.std(1)).mean()), pit_ks=ks_unif(u))
        r["scm_absz"] = float((np.abs(SCM[s] - ref.mean(1)) / ref.std(1)).mean())
        # SBC of the truth in each posterior
        for nm in ("ref", "scsi", "iwfit"):
            ranks = (S[nm][s] < T[s][:, None]).mean(1)
            r[f"sbc_ks_{nm}"] = ks_unif(ranks)
        metrics[s] = r
    json.dump({"toy": toy.name, "d": toy.d, "N": toy.N, "M": toy.M, "kbest": kbest, "n_test": a.n_test, "n_ref": a.n_ref, "J": a.J,
               "ref_diag": ref_diag, "metrics": metrics}, open(f"results/toys/{toy.name}_metrics.json", "w"), indent=1, default=float)
    arrs = {f"{nm}__{s}": S[nm][s].astype(np.float32) for nm in S for s in STATS}
    arrs.update({f"truth__{s}": T[s] for s in STATS}); arrs.update({f"scm__{s}": SCM[s] for s in STATS})
    np.savez(f"results/toys/{toy.name}_stats.npz", **arrs)
    print(f"{'stat':12s} {'floor':>6s} | SC-SI: W1 sd-ratio bias  PITks | IW-fit: W1 sd-ratio bias PITks | SCM|z|  | SBC-KS ref/scsi/iw")
    for s in STATS:
        r = metrics[s]
        print(f"{s:12s} {r['floor_w1']:6.3f} | {r['scsi']['w1']:6.3f} {r['scsi']['sd_ratio']:6.2f} {r['scsi']['bias_z']:+6.2f} {r['scsi']['pit_ks']:6.3f} | "
              f"{r['iwfit']['w1']:6.3f} {r['iwfit']['sd_ratio']:6.2f} {r['iwfit']['bias_z']:+6.2f} {r['iwfit']['pit_ks']:6.3f} | {r['scm_absz']:5.2f}  | "
              f"{r['sbc_ks_ref']:.3f}/{r['sbc_ks_scsi']:.3f}/{r['sbc_ks_iwfit']:.3f}")
