"""Baselines and metrics shared by the real-data experiments (EEG, finance).

Everything works from (Ce, N) only: the empirical covariance and its (effective) degrees of freedom.
"""
import math
import numpy as np
import torch

from scsi import DT


# ----------------------------------------------------------------------------------------------- #
# Inverse-Wishart machinery (Bartlett sampler, conjugate posterior, marginal likelihood)
# ----------------------------------------------------------------------------------------------- #
def sample_iw(nu, Psi, size, gen=None):
    """C ~ IW_d(nu, Psi); Psi may be (d,d) or (size,d,d).
    Runs on Psi's device: gen is always a CPU generator, randomness is drawn on CPU then moved (same pattern
    as scsi.wishart_channel), so this works transparently whether Psi is CPU or CUDA."""
    Psi = Psi.to(DT)
    device = Psi.device
    d = Psi.shape[-1]
    LB = torch.linalg.cholesky(Psi)
    T = torch.zeros(size, d, d, dtype=DT, device=device)
    for i in range(d):
        T[:, i, i] = torch.sqrt(torch._standard_gamma(torch.full((size,), (nu - i) / 2.0, dtype=DT), generator=gen).to(device) * 2.0)
        for j in range(i):
            T[:, i, j] = torch.randn(size, dtype=DT, generator=gen).to(device)
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


def iw_exact_predictive_logscore(Ce_val, Ce_cal, N, nu0, Psi0):
    """Exact (zero Monte-Carlo) posterior-predictive log-score log p(Ce_val|Ce_cal) for the conjugate
    IW(nu0,Psi0) prior x Wishart(N,C) channel, in the SAME units/convention as wishart_logscore-based MC
    scoring (posterior_predictive_logscore), so directly comparable to any other method's MC-estimated score
    on the same (Ce_val,N).  Ce_val, Ce_cal: (M,d,d) -> (M,).

    Why this exists: toy.ref_posterior's own Monte-Carlo score (log-mean-exp of J posterior draws) is a
    BIASED estimator of this same quantity -- unbiased-in-the-mean draws C_j fed through the concave log
    give E[log-mean-exp] < log(true mean) by Jensen's inequality, and at d=16 this bias is enormous (~20+
    nats at J up to 2e5) because p(Ce_val|C) has an extremely heavy right tail under the exact IW posterior
    (effective sample size of the log-mean-exp sum is ~2-17 out of J=2048 draws). This closed form sidesteps
    sampling entirely, so it has no such bias -- used to discover and fix the spurious "SC-SI/IW-ML beat the
    oracle" readings on iw_ri_d16_inf/iw_nonri_d16_inf, which were really just measuring the oracle's own MC
    bias (much larger than SC-SI's/IW-ML's, since their implicit predictive distributions are smoother).

    Derivation: plugging the TRUE posterior params (nu0+N, Psi0+N*Ce_cal) into iw_marginal_logpdf (which
    computes the fully-normalized log p(S) for S=N*Ce under an IW(.)-prior x Wishart(N,C)-channel model)
    gives, by standard conjugacy, exactly log p(S_val|Ce_cal) where S_val=N*Ce_val. Converting this to the
    same convention wishart_logscore uses (which omits the (Ce_val,N,d)-dependent, C-independent normalizing
    term K0 of the Wishart channel's own log-density) requires subtracting that same K0 -- verified to
    reproduce the large-J Monte-Carlo estimate of toy.ref_posterior's score to within MC noise at small d,
    and to match torch.distributions.Wishart's log_prob to 1e-5 at the conditional-density level."""
    d = Ce_val.shape[-1]
    nu_post = nu0 + N
    Psi_post = Psi0.to(Ce_cal.device) + N * Ce_cal.to(DT)
    exact_margin = iw_marginal_logpdf(Ce_val, N, nu_post, Psi_post)
    S_val = N * Ce_val.to(DT)
    ld_S = torch.linalg.slogdet(S_val)[1]
    K0 = (N - d - 1) / 2 * ld_S - N * d / 2 * math.log(2.0) - float(log_mvgamma(N / 2, d))
    return exact_margin - K0


class IWPrior:
    """Conjugate inverse-Wishart prior fitted to a noisy ensemble by maximising the observed marginal
    likelihood.  Uses only (Ce_1..M, N)."""

    def __init__(self, Ce, N, iters=600, seed=0):
        d = Ce.shape[-1]
        self.N, self.d = N, d
        Ce = Ce.to(DT)
        device = Ce.device                                             # fit (and later sample) on Ce's own device
        m = Ce.mean(0)
        L0 = torch.linalg.cholesky(m * 10.0)
        raw = torch.tensor(L0.cpu().numpy()[np.tril_indices(d)], dtype=DT, device=device).clone().requires_grad_(True)
        a = torch.tensor(math.log(10.0), dtype=DT, device=device, requires_grad=True)   # nu0 = d+1+exp(a) (kappa0)
        tri = torch.tril_indices(d, d, device=device)

        def build(raw, a):
            L = torch.zeros(d, d, dtype=DT, device=device)
            L[tri[0], tri[1]] = raw
            L = L - torch.diag(torch.diag(L)) + torch.diag(torch.exp(torch.log(torch.diag(L).abs() + 1e-12)))
            return L @ L.T + 1e-9 * torch.eye(d, dtype=DT, device=device), d + 1 + torch.exp(a)

        opt = torch.optim.Adam([raw, a], lr=0.05)
        for _ in range(iters):
            Psi0, nu0 = build(raw, a)
            loss = -iw_marginal_logpdf(Ce, N, nu0, Psi0).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            self.Psi0, self.nu0 = build(raw, a)
        self.nu0 = float(self.nu0)
        self.loglik = -float(loss)

    def posterior_params(self, Ce):
        """Posterior parameters"""
        return self.nu0 + self.N, self.Psi0 + self.N * Ce.to(self.Psi0.device)

    def sample_posterior(self, Ce, n_draws, gen=None):
        """Ce: (M,d,d) -> (M,n_draws,d,d)."""
        M = Ce.shape[0]
        n, B = self.posterior_params(Ce.to(DT))
        Bre = B.repeat_interleave(n_draws, dim=0)
        return sample_iw(n, Bre, M * n_draws, gen).reshape(M, n_draws, self.d, self.d)

    def exact_logscore(self, Ce_val, Ce_cal):
        """log p(Ce_val|Ce_cal), exact (zero Monte-Carlo) -- same closed form as IWToy.exact_logscore,
        just using this instance's ML-fitted (nu0,Psi0) instead of a toy's true prior params. IWPrior.
        sample_posterior-based MC scoring suffers the exact same severe low-ESS bias as toy.ref_posterior
        at d=16 (both are IW posteriors), so this should be preferred wherever available."""
        return iw_exact_predictive_logscore(Ce_val, Ce_cal, self.N, self.nu0, self.Psi0)


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
    I = torch.eye(d, dtype=Ce.dtype)
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
    lam, U = torch.linalg.eigh(Ce)                               # ascending
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
def wishart_logscore(Cval, N, Sigma):
    """Returns Whishart log density p(Cval|C) with C=Sigma @ Sigma^T. C^-1 Cval is computed via two triangular solves instead of torch.linalg.inv for
    better numerical stability.
    Args:
        Cval (M, d, d). Validation Covariance matrices.
        N. Number of samples to form a sample covariance matrix.
        Sigma. Cholesky factorization of the population covariance matrix.
    """
    ld = 2.0 * torch.log(torch.diagonal(Sigma, dim1=-2, dim2=-1).abs()).sum(-1)
    Cval_b = Cval.unsqueeze(-3) if Sigma.dim() == 4 else Cval            # broadcast (M,d,d) -> (M,1,d,d) vs (M,J,d,d)
    Y = torch.linalg.solve_triangular(Sigma, Cval_b.expand_as(Sigma) if Sigma.dim() == 4 else Cval_b, upper=False)
    Z = torch.linalg.solve_triangular(Sigma.transpose(-1, -2), Y, upper=True)
    tr = torch.diagonal(Z, dim1=-2, dim2=-1).sum(-1)
    return -N / 2 * (tr + ld)


def posterior_predictive_logscore(Cval, N, Cs):
    """log (1/J) sum_j p(Cval | C_j)  for posterior draws Cs (M,J,d,d)."""
    s = wishart_logscore(Cval, N, Cs)
    return torch.logsumexp(s, dim=1) - math.log(s.shape[1])


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
    w = torch.linalg.eigvalsh((M + M.transpose(-1, -2)) / 2)
    return torch.sqrt((torch.log(w.clamp_min(1e-300)) ** 2).sum(-1))


def inv_sqrt(C):
    """Inverse square-root of C"""
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
