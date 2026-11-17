"""Predictive log-score of posterior samplers on validation pairs (Ce_cal, Ce_val) that share a latent C.

Score of a posterior sampler q:  mean over pairs of  log (1/J) sum_j p(Ce_val | C_j),  C_j ~ q(. | Ce_cal),
with p the Wishart channel up to C-independent terms (covutils.wishart_logscore, which takes the full covariance C).
The log-mean-exp of J draws is biased low, by an amount that depends on the sampler, so every method is scored at the
SAME J on the SAME pairs. For the conjugate IW toys the predictive density is also available in closed form.

One function per method, each returning an entry {mean, se, per_task, ess, sec[, ...]} (log_score_oracle returns
{"oracle": ...[, "oracle_exact": ...]}). Seeds: oracle 4321, IW-ML 4322, SC-SI 777, as in all the result files.
The validation pairs and the IW-ML fit are defined in toy_data.py. Used by toy_logscore_oracle.py,
toy_logscore_scsi.py and toy_dt_study.py.
"""
import math, time
import torch
from scsi import DT
from covutils import wishart_logscore, iw_marginal_logpdf, log_mvgamma
from toys import IWToy
from runs import load_scsi


def mc_logscore(Ce_val, N, Cs, device, chunk=50000):
    """Per pair: log (1/J) sum_j p(Ce_val|C_j) and the importance ESS of the weights p(Ce_val|C_j).
    Ce_val (M,d,d); Cs (M,J,d,d) on any device. Returns two (M,) CPU tensors."""
    M, J = Cs.shape[:2]
    ls, ess = torch.empty(M, dtype=DT), torch.empty(M, dtype=DT)
    for i in range(M):
        cv = Ce_val[i:i + 1].to(device, DT)
        s = torch.cat([wishart_logscore(cv, N, Cs[i:i + 1, j:j + chunk].to(device, DT))[0] for j in range(0, J, chunk)])
        lse = torch.logsumexp(s, 0)
        ls[i] = (lse - math.log(J)).cpu()
        ess[i] = torch.exp(2 * lse - torch.logsumexp(2 * s, 0)).cpu()
    return ls, ess


def iw_exact_logscore(Ce_val, Ce_cal, N, nu0, Psi0):
    """Closed-form log p(Ce_val | Ce_cal) under an IW(nu0, Psi0) prior, on wishart_logscore's scale (i.e. dropping
    the same C-independent terms (N-d-1)/2 log|S| - Nd/2 log 2 - log Gamma_d(N/2), S = N Ce_val)."""
    d = Ce_val.shape[-1]
    Ce_val, Ce_cal = Ce_val.to(DT), Ce_cal.to(DT)
    nu, Psi = nu0 + N, Psi0.to(Ce_cal.device, DT) + N * Ce_cal
    ld_S = torch.linalg.slogdet(N * Ce_val)[1]
    return (iw_marginal_logpdf(Ce_val, N, nu, Psi) - (N - d - 1) / 2 * ld_S + N * d / 2 * math.log(2.0)
            + log_mvgamma(N / 2, d).to(Ce_val.device)).cpu()


def summary(x):
    x = x.double()
    return {"mean": float(x.mean()), "se": float(x.std() / math.sqrt(len(x))), "per_task": [float(v) for v in x]}


def _entry(ls, ess, t0):
    return summary(ls) | {"ess": [float(v) for v in ess], "sec": time.time() - t0}


def _per_pair(Ce_cal, Ce_val, N, draw, device):
    """score pair by pair, draw(i) = (1, J, d, d) posterior draws given Ce_cal[i]"""
    per = [mc_logscore(Ce_val[i:i + 1], N, draw(i), device) for i in range(Ce_cal.shape[0])]
    return torch.cat([p[0] for p in per]), torch.cat([p[1] for p in per])


def log_score_oracle(toy, Ce_cal, Ce_val, J, device, hmc_chains=2000, warmup=1500):
    """{"oracle": ...[, "oracle_exact": ...]}: the true posterior by direct draws (IW conjugate, AR(1) grid) or HMC
    (hmc_chains short chains per pair), plus the closed-form predictive for the IW toys."""
    gg = torch.Generator().manual_seed(4321); t0 = time.time()
    if toy.exact:
        ls, ess = _per_pair(Ce_cal, Ce_val, toy.N, lambda i: toy.ref_posterior(Ce_cal[i:i + 1], J, gg), device)
        diag = {"sampler": "direct"}
    else:
        Cs, diag = toy.hmc(Ce_cal, n_chains=hmc_chains, n_keep=int(math.ceil(J / hmc_chains)), thin=4, warmup=warmup,
                           gen=gg, return_diag=True)
        if Cs.shape[1] > J:                                                    # all draws used otherwise: no shuffle copy
            Cs = Cs[:, torch.randperm(Cs.shape[1], generator=gg).to(Cs.device)][:, :J]
        diag.update(sampler="hmc", warmup=warmup)
        ls, ess = mc_logscore(Ce_val, toy.N, Cs, device); del Cs
    out = {"oracle": _entry(ls, ess, t0) | {"diag": diag}}
    if isinstance(toy, IWToy):
        out["oracle_exact"] = summary(iw_exact_logscore(Ce_val, Ce_cal, toy.N, toy.nu0, toy.Psi0))
    return out


def log_score_iwml(iw, Ce_cal, Ce_val, J, device):
    """IW-ML baseline: posterior draws of a fitted covutils.IWPrior (the caller chooses its fit ensemble)."""
    gi = torch.Generator().manual_seed(4322); t0 = time.time()
    ls, ess = _per_pair(Ce_cal, Ce_val, iw.N, lambda i: iw.sample_posterior(Ce_cal[i:i + 1], J, gen=gi), device)
    return _entry(ls, ess, t0)


def log_score_scsi(toy, path, Ce_cal, Ce_val, J, device, threads=4, chunk=131072, n_steps=None):
    """SC-SI checkpoint at path; n_steps = SDE steps (None: the model's default, Delta t = 1/64)."""
    t0 = time.time()
    m, _ = load_scsi(toy, path, threads)
    gm = torch.Generator(device=m.device).manual_seed(777)
    kw = {} if n_steps is None else {"n_steps": n_steps}
    with torch.no_grad():
        # draws in blocks of <= 200000: cusolver's batched eigh (Sym.decode) rejects ~2e6-matrix batches
        draw = lambda i: torch.cat([m.sample_posterior(Ce_cal[i:i + 1], min(200000, J - j), gen=gm, chunk=chunk, **kw)
                                    for j in range(0, J, 200000)], 1)
        ls, ess = _per_pair(Ce_cal, Ce_val, toy.N, draw, device)
    del m
    return _entry(ls, ess, t0)
