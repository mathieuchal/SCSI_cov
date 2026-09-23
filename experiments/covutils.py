"""Baselines and metrics shared by the real-data experiments (EEG, finance).

Everything works from (Ce, N) only: the empirical covariance and its (effective) degrees of freedom.
"""
import math
import numpy as np
import torch

from scsi import DT, wishart_channel, safe_eigh, safe_eigvalsh


# ----------------------------------------------------------------------------------------------- #
# inverse-Wishart machinery (Bartlett sampler, conjugate posterior, marginal likelihood)
# ----------------------------------------------------------------------------------------------- #
def sample_iw(nu, Psi, size, gen=None):
    """C ~ IW_d(nu, Psi) (paper's convention, eq. 132); Psi may be (d,d) or (size,d,d).

    Runs on Psi's device; gen, if given, must be a torch.Generator on that same device."""
    Psi = Psi.to(DT)
    d = Psi.shape[-1]
    dev = Psi.device
    LB = torch.linalg.cholesky(Psi)
    T = torch.zeros(size, d, d, dtype=DT, device=dev)
    for i in range(d):
        T[:, i, i] = torch.sqrt(torch._standard_gamma(torch.full((size,), (nu - i) / 2.0, dtype=DT, device=dev), generator=gen) * 2.0)
        for j in range(i):
            T[:, i, j] = torch.randn(size, dtype=DT, device=dev, generator=gen)
    LBt = LB.transpose(-1, -2)
    if LBt.dim() == 2:
        LBt = LBt.expand(size, d, d)
    D = torch.linalg.solve_triangular(T, LBt, upper=False)
    return D.transpose(-1, -2) @ D


def log_mvgamma(a, d):
    return d * (d - 1) / 4 * math.log(math.pi) + sum(torch.special.gammaln(torch.as_tensor(a + (1 - i) / 2, dtype=DT))
                                                     for i in range(1, d + 1))


def iw_marginal_logpdf(Ce, N, nu0, Psi0):
    """log p(S=N*Ce) under IW(nu0,Psi0) prior x Wishart channel (matrix-variate beta type II), up to the
    parameter-free Jacobian.  Ce: (M,d,d)."""
    d = Ce.shape[-1]
    S = N * Ce
    ld = lambda A: torch.linalg.slogdet(A)[1]
    return ((N - d - 1) / 2 * ld(S) + nu0 / 2 * ld(Psi0) - (N + nu0) / 2 * ld(Psi0 + S)
            + log_mvgamma((nu0 + N) / 2, d) - log_mvgamma(nu0 / 2, d) - log_mvgamma(N / 2, d))


class IWPrior:
    """Conjugate inverse-Wishart prior fitted to a noisy ensemble by maximising the observed marginal
    likelihood (the paper's parametric baseline, App. H.5).  Uses only (Ce_1..M, N)."""

    def __init__(self, Ce, N, iters=600, seed=0):
        d = Ce.shape[-1]
        self.N, self.d = N, d
        Ce = Ce.to(DT)
        dev = Ce.device
        m = Ce.mean(0)
        L0 = torch.linalg.cholesky(m * 10.0)
        tri = torch.tril_indices(d, d, device=dev)
        raw = L0[tri[0], tri[1]].clone().detach().requires_grad_(True)   # device-preserving (no numpy round-trip)
        a = torch.tensor(math.log(10.0), dtype=DT, device=dev, requires_grad=True)   # nu0 = d + 1 + exp(a) ... (kappa0)

        def build(raw, a):
            L = torch.zeros(d, d, dtype=DT, device=dev)
            L[tri[0], tri[1]] = raw
            L = L - torch.diag(torch.diag(L)) + torch.diag(torch.exp(torch.log(torch.diag(L).abs() + 1e-12)))
            return L @ L.T + 1e-9 * torch.eye(d, dtype=DT, device=dev), d + 1 + torch.exp(a)

        opt = torch.optim.Adam([raw, a], lr=0.05)
        for _ in range(iters):
            Psi0, nu0 = build(raw, a)
            loss = -iw_marginal_logpdf(Ce, N, nu0, Psi0).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            self.Psi0, self.nu0 = build(raw, a)
        self.nu0 = float(self.nu0)
        self.loglik = -float(loss.detach())

    def posterior_params(self, Ce):
        return self.nu0 + self.N, self.Psi0 + self.N * Ce

    def sample_posterior(self, Ce, n_draws, gen=None):
        """Ce: (M,d,d) -> (M,n_draws,d,d)."""
        M = Ce.shape[0]
        n, B = self.posterior_params(Ce.to(DT))
        Bre = B.repeat_interleave(n_draws, dim=0)
        return sample_iw(n, Bre, M * n_draws, gen).reshape(M, n_draws, self.d, self.d)


# ----------------------------------------------------------------------------------------------- #
# point estimators from (Ce, N) only
# ----------------------------------------------------------------------------------------------- #
def oas_shrink(Ce, N):
    """Oracle-approximating shrinkage (Chen et al. 2010): linear shrinkage of Ce toward mu*I using only
    (Ce, N) -- the Gaussian-model analogue of Ledoit-Wolf linear shrinkage.  Ce: (...,d,d)."""
    d = Ce.shape[-1]
    tr = torch.diagonal(Ce, dim1=-2, dim2=-1).sum(-1)
    tr2 = (Ce * Ce).sum((-1, -2))
    num = (1 - 2.0 / d) * tr2 + tr ** 2
    den = (N + 1 - 2.0 / d) * (tr2 - tr ** 2 / d)
    rho = (num / den).clamp(0, 1)
    mu = tr / d
    I = torch.eye(d, dtype=Ce.dtype, device=Ce.device)
    return (1 - rho)[..., None, None] * Ce + (rho * mu)[..., None, None] * I


def nls_shrink(Ce, N):
    """Ledoit-Wolf analytical nonlinear shrinkage (Ledoit & Wolf, Ann. Statist. 48(5), 2020), p < N case.

    Rotation-equivariant: keeps the eigenvectors of Ce and replaces each sample eigenvalue l_i by
        d_i = l_i / [ (pi c l_i f(l_i))^2 + (1 - c - pi c l_i Hf(l_i))^2 ],     c = d / N,
    where f is an Epanechnikov-kernel estimate of the sample spectral density (local bandwidth h*l_j,
    h = N^(-1/3)) and Hf its Hilbert transform (convention H g(x) = (1/pi) PV int g(t)/(t-x) dt).
    Uses only (Ce, N), like OAS; N is the (effective) Wishart degrees of freedom.  Ce: (...,d,d)."""
    d = Ce.shape[-1]
    c = d / float(N)
    if c >= 1.0:
        raise ValueError("analytical NLS implemented for N > d")
    lam, U = safe_eigh(Ce)                                       # ascending; chunked for cuSOLVER's batch limit
    lam = lam.clamp_min(1e-12 * lam[..., -1:])
    h = float(N) ** (-1.0 / 3.0)
    hj = h * lam.unsqueeze(-2)                                   # (...,1,d): bandwidth of eigenvalue j
    x = (lam.unsqueeze(-1) - lam.unsqueeze(-2)) / hj             # (...,d,d): [i,j] = (l_i - l_j) / (h l_j)
    s5 = math.sqrt(5.0)
    f = ((3.0 / (4.0 * s5)) * (1.0 - x ** 2 / 5.0).clamp_min(0.0) / hj).mean(-1)
    num, den = s5 - x, s5 + x
    logterm = torch.where((num.abs() > 0) & (den.abs() > 0), torch.log((num / den).abs()), torch.zeros_like(x))
    Hk = -3.0 * x / (10.0 * math.pi) + (3.0 / (4.0 * s5 * math.pi)) * (1.0 - x ** 2 / 5.0) * logterm
    Hf = (Hk / hj).mean(-1)
    dt = lam / ((math.pi * c * lam * f) ** 2 + (1.0 - c - math.pi * c * lam * Hf) ** 2)
    return (U * dt.unsqueeze(-2)) @ U.transpose(-1, -2)


# ----------------------------------------------------------------------------------------------- #
# predictive scoring
# ----------------------------------------------------------------------------------------------- #
def wishart_logscore(Cval, N, C):
    """log p(Cval | C) up to terms not depending on C:  -N/2 [ tr(C^-1 Cval) + log det C ].
    Cval: (M,d,d); C: (M,d,d) or (M,J,d,d) -> (M,) or (M,J)."""
    if C.dim() == 3:
        return -N / 2 * (torch.einsum("mij,mji->m", torch.linalg.inv(C), Cval) + torch.linalg.slogdet(C)[1])
    Ci = torch.linalg.inv(C)
    return -N / 2 * (torch.einsum("mjab,mba->mj", Ci, Cval) + torch.linalg.slogdet(C)[1])


def posterior_predictive_logscore(Cval, N, Cs):
    """log (1/J) sum_j p(Cval | C_j)  for posterior draws Cs (M,J,d,d)."""
    s = wishart_logscore(Cval, N, Cs)
    return torch.logsumexp(s, dim=1) - math.log(s.shape[1])


def directional_pit(vals, Cval, N, C_draws, dirs, gen=None, n_pred=1):
    """PIT of observed v^T Cval v under the predictive law (draw C, then Wishart(N) noise).
    C_draws: (M,J,d,d) (a plug-in estimate is J=1 repeated).  dirs: (K,d) fixed directions or (M,K,d).
    Returns pit (M,K), predictive samples of v^T Ce v as (M,K,J)."""
    M, J, d, _ = C_draws.shape
    Ce_pred = wishart_channel(C_draws.reshape(M * J, d, d), N, gen).reshape(M, J, d, d)
    if dirs.dim() == 2:
        dirs = dirs.unsqueeze(0).expand(M, -1, -1)
    obs = torch.einsum("mkd,mde,mke->mk", dirs, Cval, dirs)
    pred = torch.einsum("mkd,mjde,mke->mkj", dirs, Ce_pred, dirs)
    pit = (pred < obs.unsqueeze(-1)).to(DT).mean(-1) + 0.5 * (pred == obs.unsqueeze(-1)).to(DT).mean(-1)
    return pit, pred, obs


def functional_pit(Cval, N, C_draws, stat_fn, gen=None):
    """PIT of an observed scalar functional stat_fn(Cval) under the posterior predictive law (draw C from the
    posterior, then push through the known Wishart(N) channel) -- the eigenvalue-functional analogue of
    directional_pit (which is for a fixed direction's quadratic form).  Ground-truth free: uses only Cval, N,
    and posterior draws, exactly like directional_pit / posterior_predictive_logscore.

    C_draws: (M,J,d,d) posterior draws conditioned on a companion observation (e.g. Ce_cal) sharing Cval's
    latent covariance.  stat_fn: (...,d,d) -> (...), a functional of a covariance's own eigenvalues/entries
    (e.g. logdet, log condition number, top-eigenvalue share, corr01) -- must NOT depend on an external
    reference direction (use directional_pit for that).  Returns pit (M,), obs (M,), pred (M,J)."""
    M, J, d, _ = C_draws.shape
    Ce_pred = wishart_channel(C_draws.reshape(M * J, d, d), N, gen).reshape(M, J, d, d)
    obs = stat_fn(Cval)
    pred = stat_fn(Ce_pred.reshape(M * J, d, d)).reshape(M, J)
    pit = (pred < obs.unsqueeze(-1)).to(DT).mean(-1) + 0.5 * (pred == obs.unsqueeze(-1)).to(DT).mean(-1)
    return pit, obs, pred


def coverage_from_pit(pit, levels=(0.5, 0.8, 0.95)):
    out = {}
    for lv in levels:
        lo, hi = (1 - lv) / 2, 1 - (1 - lv) / 2
        out[lv] = float(((pit > lo) & (pit < hi)).to(DT).mean())
    return out


# ----------------------------------------------------------------------------------------------- #
# Riemannian geometry
# ----------------------------------------------------------------------------------------------- #
def airm_distance(A, B):
    """Affine-invariant Riemannian distance sqrt(sum log^2 eig(A^-1 B)); batched."""
    L = torch.linalg.cholesky(A)
    Li = torch.linalg.inv(L)
    M = Li @ B @ Li.transpose(-1, -2)
    w = safe_eigvalsh((M + M.transpose(-1, -2)) / 2)   # A,B can carry a large (M,J,...) batch of posterior draws
    return torch.sqrt((torch.log(w.clamp_min(1e-300)) ** 2).sum(-1))


def tangent_features(C, Cref_isqrt, svec):
    """Tangent-space (log-Euclidean at Cref) features svec(log(Cref^-1/2 C Cref^-1/2))."""
    from scsi import Sym
    M = Cref_isqrt @ C @ Cref_isqrt
    M = (M + M.transpose(-1, -2)) / 2
    return svec.svec(Sym.logm(M))


def inv_sqrt(C):
    w, V = torch.linalg.eigh(C)
    return (V * (w ** -0.5).unsqueeze(-2)) @ V.transpose(-1, -2)


def bootstrap_ci(x, groups, n_boot=1000, seed=0, agg=np.mean):
    """Group-level bootstrap (resample groups = subjects) of a per-case statistic x."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x); groups = np.asarray(groups)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    stats_ = []
    for _ in range(n_boot):
        gs = rng.choice(ug, len(ug), replace=True)
        stats_.append(agg(np.concatenate([x[idx[g]] for g in gs])))
    return float(agg(x)), float(np.percentile(stats_, 2.5)), float(np.percentile(stats_, 97.5))
