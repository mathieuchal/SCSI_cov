"""Validation of the SCSI implementation against exact inverse-Wishart posteriors (Appendix H).

The learner only sees M empirical covariances Ce (never the clean C nor the prior parameters).
Diagnostics follow Appendix H.3: posterior mean / precision-mean error, directional-variance PIT
(eq. 142), RI orientation second moments (eq. 143), Bartlett-Cholesky check (eq. 137).
"""
import argparse, json, math, time
import numpy as np
import torch
from scipy import stats

from scsi import SCSI, SCSIConfig, wishart_channel, DT


def sample_iw(nu, Psi, size, gen=None):
    """C ~ IW_d(nu, Psi) via Bartlett:  C = L_B (T T^T)^-1 L_B^T."""
    d = Psi.shape[-1]
    LB = torch.linalg.cholesky(Psi)
    T = torch.zeros(size, d, d, dtype=DT)
    for i in range(d):
        T[:, i, i] = torch.sqrt(torch.distributions.Chi2(torch.tensor(float(nu - i), dtype=DT)).sample((size,)))
        for j in range(i):
            T[:, i, j] = torch.randn(size, dtype=DT, generator=gen)
    # solve T D = L_B^T  -> D = T^-1 L_B^T ; C = D^T D
    D = torch.linalg.solve_triangular(T, LB.T.expand(size, d, d), upper=False)
    return D.transpose(-1, -2) @ D


def make_prior(d, kind, kappa0, seed=0):
    nu0 = d + 1 + kappa0
    if kind == "ri":
        C0 = torch.eye(d, dtype=DT)
    else:  # non-RI: fixed rotated geometric spectrum (condition number 16), mean eigenvalue 1
        g = torch.Generator().manual_seed(123)
        Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=DT, generator=g))
        ev = torch.tensor(np.geomspace(1.0, 16.0, d), dtype=DT)
        ev = ev / ev.mean()
        C0 = Q @ torch.diag(ev) @ Q.T
    return nu0, kappa0 * C0, C0


def evaluate(model: SCSI, nu0, Psi0, N, d, n_test=256, J=512, seed=999, kind="ri"):
    gen = torch.Generator().manual_seed(seed)
    C_true = sample_iw(nu0, Psi0, n_test, gen)
    Ce = wishart_channel(C_true, N, gen)
    n = nu0 + N
    B = Psi0 + N * Ce                       # (T,d,d)
    r = n - d - 1
    M_star = B / r
    Om_star = n * torch.linalg.inv(B)
    Cs = model.sample_posterior(Ce, J, gen=gen)          # (T,J,d,d)
    out = {}
    M_hat = Cs.mean(1)
    Om_hat = torch.linalg.inv(Cs).mean(1)
    out["e_M"] = float((torch.linalg.norm(M_hat - M_star, dim=(-1, -2)) / torch.linalg.norm(M_star, dim=(-1, -2))).mean())
    out["e_Omega"] = float((torch.linalg.norm(Om_hat - Om_star, dim=(-1, -2)) / torch.linalg.norm(Om_star, dim=(-1, -2))).mean())
    # trivial baselines for scale of e_M: sample covariance and the prior mean
    out["e_M_SCM"] = float((torch.linalg.norm(Ce - M_star, dim=(-1, -2)) / torch.linalg.norm(M_star, dim=(-1, -2))).mean())

    # directional variance PIT (eq. 142) over coordinate axes, empirical eigenvectors, and pair sums
    w, V = torch.linalg.eigh(Ce)                       # columns of V = empirical eigenvectors
    dirs = []
    for i in range(d):
        e = torch.zeros(d, dtype=DT); e[i] = 1
        dirs.append(e.expand(n_test, d))
    for i in range(d):
        dirs.append(V[:, :, i])
    for i in range(d - 1):
        dirs.append((V[:, :, i] + V[:, :, i + 1]) / math.sqrt(2))
    pits = []
    a = (r + 2) / 2
    for v in dirs:
        vB = torch.einsum("ti,tij,tj->t", v, B, v)
        Vs = torch.einsum("ti,tjik,tk->tj", v, Cs, v)              # (T,J)
        P = stats.invgamma.cdf(Vs.numpy(), a, scale=(vB / 2).numpy()[:, None])
        pits.append(P.reshape(-1))
    pits = np.concatenate(pits)
    out["pit_ks"] = float(stats.kstest(pits, "uniform").statistic)
    # coverage of central intervals
    for lvl in (0.5, 0.8, 0.95):
        lo, hi = (1 - lvl) / 2, 1 - (1 - lvl) / 2
        out[f"pit_cov{int(lvl*100)}"] = float(((pits > lo) & (pits < hi)).mean())

    # spectral summary: log det mean/var vs exact (eq. 144)
    ld = torch.linalg.slogdet(Cs)[1]                               # (T,J)
    ld_mean = (torch.linalg.slogdet(B)[1] - d * math.log(2)
               - sum(torch.special.digamma(torch.tensor((n + 1 - i) / 2)).item() for i in range(1, d + 1)))
    ld_var = sum(torch.special.polygamma(1, torch.tensor((n + 1 - i) / 2)).item() for i in range(1, d + 1))
    out["logdet_mean_err"] = float((ld.mean(1) - ld_mean).abs().mean())
    out["logdet_sd_ratio"] = float((ld.std(1) / math.sqrt(ld_var)).mean())

    if kind == "ri":
        # orientation-sensitive check (eq. 143): ratio of sampled to exact E[Ctilde_ij^2]
        Ct = V.transpose(-1, -2)[:, None] @ Cs @ V[:, None]         # (T,J,d,d)
        bi = Psi0[0, 0] + N * w                                    # (T,d)
        rat = []
        for i in range(d):
            for j in range(i + 1, d):
                ex = bi[:, i] * bi[:, j] / ((r + 1) * r * (r - 2))
                rat.append(((Ct[:, :, i, j] ** 2).mean(1) / ex).mean().item())
        out["offdiag_2nd_moment_ratio"] = float(np.mean(rat))
        # negative control: eigenvectors of Ce with spectrum from samples (no orientation) -> ratio 0
    # Bartlett-Cholesky check (eq. 137): W = L_B^T C^-1 L_B ~ Wishart(n, I)
    LB = torch.linalg.cholesky(B)
    Cinv = torch.linalg.inv(Cs)
    Wm = LB.transpose(-1, -2)[:, None] @ Cinv @ LB[:, None]
    Tm = torch.linalg.cholesky(Wm)
    ks_diag = []
    for i in range(d):
        x = (Tm[..., i, i] ** 2).reshape(-1).numpy()
        ks_diag.append(stats.kstest(x, stats.chi2(n - i).cdf).statistic)
    ks_off = []
    for i in range(d):
        for j in range(i):
            ks_off.append(stats.kstest(Tm[..., i, j].reshape(-1).numpy(), "norm").statistic)
    out["chol_diag_ks"] = float(np.mean(ks_diag))
    out["chol_off_ks"] = float(np.mean(ks_off))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--Nfac", type=float, default=2.0)
    ap.add_argument("--M", type=int, default=5000)
    ap.add_argument("--kind", default="ri", choices=["ri", "nonri"])
    ap.add_argument("--kappa0", type=float, default=None)
    ap.add_argument("--n_outer", type=int, default=6)
    ap.add_argument("--steps_first", type=int, default=4000)
    ap.add_argument("--steps_outer", type=int, default=2000)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--supervised", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--init", default="raw")
    ap.add_argument("--inflate", type=float, default=1.0)
    ap.add_argument("--n_recon", type=int, default=8)
    a = ap.parse_args()

    torch.set_num_threads(8)
    d, N = a.d, int(round(a.Nfac * a.d))
    kappa0 = a.kappa0 if a.kappa0 is not None else d + 8
    nu0, Psi0, C0 = make_prior(d, a.kind, kappa0)
    gen = torch.Generator().manual_seed(a.seed + 1)
    C_train = sample_iw(nu0, Psi0, a.M, gen)
    Ce_train = wishart_channel(C_train, N, gen)
    cfg = SCSIConfig(N=N, d=d, kappa=a.kappa, n_outer=a.n_outer, steps_first=a.steps_first,
                     steps_outer=a.steps_outer, seed=a.seed, init=a.init,
                     log_prior_inflate=a.inflate, n_recon=a.n_recon)
    model = SCSI(cfg)
    res_by_iter = {}

    def cb(m, k):
        if k in (0, 1, 3, 6, 10, 15, 20, 30, a.n_outer):
            r = evaluate(m, nu0, Psi0, N, d, n_test=64, J=128, kind=a.kind)
            res_by_iter[k] = r
            print(f"   iter {k}: " + " ".join(f"{kk}={v:.3f}" for kk, v in r.items()), flush=True)

    t0 = time.time()
    if a.supervised:
        model.fit_supervised(lambda B: sample_iw(nu0, Psi0, B), Ce_train)
    else:
        model.fit(Ce_train, callback=cb)
    print(f"fit time {time.time()-t0:.0f}s")
    final = evaluate(model, nu0, Psi0, N, d, n_test=256, J=512, kind=a.kind)
    print("FINAL", json.dumps(final, indent=1))
    name = f"results/iw_{a.kind}_d{d}_N{N}_M{a.M}{'_sup' if a.supervised else ''}{a.tag}.json"
    json.dump(dict(args=vars(a), final=final, by_iter=res_by_iter, hist=model.hist), open(name, "w"), indent=1)


if __name__ == "__main__":
    main()
