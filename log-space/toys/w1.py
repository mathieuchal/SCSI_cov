"""W1 distance of posterior samplers to the oracle posterior, on the test tasks of toy_data.make_testset (128 tasks).

Statistics of C, computed from posterior draws (STATS): logdet, log condition number, top-eigenvalue share, log
directional variance along the top / bottom empirical eigenvector of Ce, correlation C01/sqrt(C00 C11). Per statistic
and per task: W1(method, oracle) / sd(oracle); aggregated over the tasks by the mean and by the median; the value is
the average over the statistics (w1_summary).

The oracle reference draws (direct, or HMC with 8 chains, seed 4242) are cached in {toy}_ref.pt. One function per
method, each returning w1_summary's {agg: {"value", "detail": {stat: {"w1", "std"}}}}:
  w1_oracle_floor : the sampling floor, W1 between the two halves of the reference draws
  w1_iwml         : a fitted IW-ML prior (toy_data.iwml_w1), J draws per task, seed 99
  w1_scsi         : an SC-SI checkpoint, J draws per task, seed 99, TF32 during sampling
Used by toy_W1_oracle.py, toy_W1_scsi.py and toy_dt_study.py.
"""
import math, os, time
import numpy as np, torch
from scipy import stats as sps
from scsi import DT
from runs import load_scsi

STATS = ["logdet", "logcond", "top_share", "ldv_top", "ldv_bot", "corr01"]
LABELS = {"logdet": "log det C", "logcond": "log condition number", "top_share": "top-eigenvalue share",
          "ldv_top": "log dir. variance (top emp. eigvec)", "ldv_bot": "log dir. variance (bottom emp. eigvec)",
          "corr01": "correlation C01"}
AGGS = ("mean", "median")


def stats_of(C, Ce):
    """C: (M,J,d,d), Ce: (M,d,d) -> dict of (M,J) tensors."""
    w, V = torch.linalg.eigh(Ce)
    vt, vb = V[..., -1], V[..., 0]                                  # top and bottom empirical eigenvectors
    we = torch.linalg.eigvalsh(C)
    dv = lambda v: torch.einsum("mi,mjik,mk->mj", v, C, v)
    return {"logdet": torch.linalg.slogdet(C)[1], "logcond": torch.log(we[..., -1] / we[..., 0]),
            "top_share": we[..., -1] / we.sum(-1), "ldv_top": torch.log(dv(vt)), "ldv_bot": torch.log(dv(vb)),
            "corr01": C[..., 0, 1] / torch.sqrt(C[..., 0, 0] * C[..., 1, 1])}


def stats_np(C, Ce):
    """stats_of on the GPU when there is one (batched eigh: ~0.1 s vs ~2.5 s per method on the CPU), as numpy."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {s: v.cpu().numpy() for s, v in stats_of(C.to(dev), Ce.to(dev)).items()}


def rhat(x):
    nk = x.shape[-1]
    m = x.mean(-1); W = x.var(-1, unbiased=True).mean(-1); B = nk * m.var(-1, unbiased=True)
    return torch.sqrt(((nk - 1) / nk * W + B / nk) / W)


def get_reference(toy, Ce, n_ref, cache, warmup=1500):
    """oracle posterior draws (n_test, n_ref, d, d) given the test Ce and their sampler diagnostics, cached in cache
    (the cache ignores n_ref: another n_ref needs its own file)"""
    if os.path.exists(cache):
        z = torch.load(cache, map_location="cpu")
        return z["Cs"].to(DT).to(Ce.device), z.get("diag", {})
    orig_device = Ce.device
    Ce = Ce.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    g = torch.Generator().manual_seed(4242)
    t0 = time.time(); diag = {}
    if toy.exact:
        Cs = toy.ref_posterior(Ce, n_ref, g)
    else:
        n_chains = 8
        Cs, diag = toy.hmc(Ce, n_chains=n_chains, n_keep=int(math.ceil(n_ref / n_chains)), thin=4, warmup=warmup, gen=g,
                           return_diag=True)
        diag["warmup"] = warmup
        S = stats_of(Cs, Ce)
        diag["rhat"] = {k: dict(max=float(rhat(v.reshape(v.shape[0], n_chains, -1)).max()),
                                median=float(rhat(v.reshape(v.shape[0], n_chains, -1)).median())) for k, v in S.items()}
        Cs = Cs[:, torch.randperm(Cs.shape[1], generator=g).to(Cs.device)]       # mix the chains
    diag["sec"] = time.time() - t0
    torch.save({"Cs": Cs.float().cpu(), "diag": diag}, cache)
    return Cs.to(orig_device), diag


def w1_norm(x, ref):
    """per-task W1 / sd(ref)"""
    return np.array([sps.wasserstein_distance(x[m], ref[m]) / ref[m].std() for m in range(x.shape[0])])


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


def w1_oracle_floor(ref_stats):
    R = next(iter(ref_stats.values())).shape[1] // 2
    return w1_summary({s: v[:, :R] for s, v in ref_stats.items()}, {s: v[:, R:2 * R] for s, v in ref_stats.items()})


def w1_iwml(iw, Ce, J, ref_stats):
    return w1_summary(stats_np(iw.sample_posterior(Ce, J, gen=torch.Generator().manual_seed(99)), Ce), ref_stats)


def w1_scsi(toy, path, Ce, J, ref_stats, threads=4, chunk=131072, n_steps=None):
    """SC-SI checkpoint at path; n_steps = SDE steps (None: the model's default, Delta t = 1/64)"""
    m, _ = load_scsi(toy, path, threads)
    torch.backends.cuda.matmul.allow_tf32 = True           # sampling only (~4x faster, W1 unchanged to 4 decimals)
    Cs = m.sample_posterior(Ce, J, gen=torch.Generator(device=m.device).manual_seed(99), chunk=chunk, n_steps=n_steps)
    torch.backends.cuda.matmul.allow_tf32 = False
    del m
    s = w1_summary(stats_np(Cs, Ce), ref_stats)
    del Cs
    return s
