"""Controlled toy models with a *reference* posterior P(C | Ce).

  IWToy     : inverse-Wishart prior (RI or non-RI scale).  Conjugate => exact posterior IW(nu0+N, Psi0+N Ce).
  FactorToy : factor + idiosyncratic covariance  C = B B^T + diag(psi)  (K = 2 factors: a positive-loading
              "market" factor and a mixed-sign second factor).  No conjugacy: the reference posterior is a
              batched Hamiltonian Monte Carlo sampler on (B, log psi), validated by simulation-based calibration.
  WishartToy, IsoDiagToy, AR1Toy, HeteroskedasticToy : our own (non-conjugate) priors,
              each with its sampling law defined directly in its own sample_prior (previously separate
              functions in a sample.py this file no longer depends on). None of these four is conjugate to
              the Gaussian/Wishart observation channel, so none has a closed-form posterior -- p(C | Ce) is
              targeted by MCMC in all of them except AR1Toy.

              AR1Toy is the exception to "HMC": its prior C(rho)_ij = rho^|i-j| has a single scalar parameter
              (rho), so instead of MCMC its reference posterior is computed by 1D quadrature -- grid-evaluate
              the (unnormalized) Wishart-channel log-posterior over rho, normalize, sample by inverse-CDF.
              Exact up to the grid resolution, no warmup/acceptance-rate/R-hat concerns.

              Design note -- the U (Haar-orthogonal) component: IsoDiagToy draws
              an orthonormal U (p x r, U^T U = I_r), i.e. a point on the Stiefel manifold, not a Euclidean
              vector. Plain (Euclidean) HMC on an unconstrained embedding of U has no simple, correct
              Jacobian (the embedding is generally not a bijection onto the manifold), so it would silently
              target the WRONG posterior. Instead we use a Gibbs-style hybrid, alternating:
                - proper leapfrog HMC on the magnitude parameters (the spikes), in an unconstrained
                  (logit) reparametrization of their truncated-normal priors, and
                - a Givens-rotation Metropolis step on U (U' = G(i,j,theta) @ U): these proposals stay on the
                  Stiefel manifold by construction and are symmetric with respect to the Haar measure there
                  (same mechanism as orientation_mcmc in covariance_posterior_em_pd_spectral.ipynb,
                  generalized from O(p) to p x r), so the acceptance ratio needs only the likelihood term,
                  no prior term for U and no Hastings correction.
              WishartToy's prior (C ~ Wishart(nu0, I), no orthogonal factor at all) and HeteroskedasticToy's
              (C = F F^T + diag(d), F unconstrained) need none of this Givens machinery -- plain HMC on an
              unconstrained parametrization is exact for both.

The learner (SC-SI) only ever sees noisy empirical covariances from these priors.
"""
import math
import numpy as np
import torch

from scsi import DT, wishart_channel
from covutils import sample_iw


class Toy:
    name: str
    d: int #Dimension of the problem
    N: int #Amount of data to manufacture Ce
    M: int #Total number of covariance matrix for training; -1 means infinite (fresh (C, Ce) ensemble redrawn
           #every outer/E-step instead of a fixed recorded set -- see SCSI.fit's `infinite_batch`).
    exact: bool

    def sample_prior(self, n, gen=None):
        raise NotImplementedError

    def ref_posterior(self, Ce, n_draws, gen=None):
        raise NotImplementedError


# --------------------------------------------------------------------------------------------- #
class IWToy(Toy):
    exact = True

    def __init__(self, name, d, N, M, kind, kappa0):
        self.name, self.d, self.N, self.M, self.kind, self.kappa0 = name, d, N, M, kind, kappa0
        self.nu0 = d + 1 + kappa0
        if kind == "ri":
            C0 = torch.eye(d, dtype=DT)
        else:  # non-RI: fixed rotated geometric spectrum (condition number 16), mean eigenvalue 1
            g = torch.Generator().manual_seed(123)
            Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=DT, generator=g))
            ev = torch.tensor(np.geomspace(1.0, 16.0, d), dtype=DT)
            C0 = Q @ torch.diag(ev / ev.mean()) @ Q.T
        self.C0, self.Psi0 = C0, kappa0 * C0

    def sample_prior(self, n, gen=None):
        """ Sample Inverse Wishart prior"""
        return sample_iw(self.nu0, self.Psi0, n, gen)

    def ref_posterior(self, Ce, n_draws, gen=None):
        M = Ce.shape[0]
        B = (self.Psi0.to(Ce.device) + self.N * Ce.to(DT)).repeat_interleave(n_draws, 0)
        return sample_iw(self.nu0 + self.N, B, M * n_draws, gen).reshape(M, n_draws, self.d, self.d)


# --------------------------------------------------------------------------------------------- #
class FactorToy(Toy):
    exact = False

    def __init__(self, name, d, N, M, m1=1.0, s1=0.3, s2=0.6, lpsi_mean=math.log(0.4), lpsi_sd=0.4):
        self.name, self.d, self.N, self.M, self.K = name, d, N, M, 2
        self.m1, self.s1, self.s2, self.lm, self.ls = m1, s1, s2, lpsi_mean, lpsi_sd
        self.P = d * self.K + d

    # parameters u = [B (d x K, row-major), log psi (d)]
    def unpack(self, u):
        B = u[..., : self.d * self.K].reshape(*u.shape[:-1], self.d, self.K)
        return B, u[..., self.d * self.K:]

    def C_from(self, u):
        B, lp = self.unpack(u)
        return B @ B.transpose(-1, -2) + torch.diag_embed(torch.exp(lp))

    def sample_params(self, n, gen=None, device=None):
        b1 = self.m1 + self.s1 * torch.randn(n, self.d, dtype=DT, generator=gen)
        b2 = self.s2 * torch.randn(n, self.d, dtype=DT, generator=gen)
        lp = self.lm + self.ls * torch.randn(n, self.d, dtype=DT, generator=gen)
        out = torch.cat([torch.stack([b1, b2], -1).reshape(n, -1), lp], -1)
        return out.to(device) if device is not None else out

    def sample_prior(self, n, gen=None):
        return self.C_from(self.sample_params(n, gen))

    def potential(self, u, Ce):
        """U(u) = -log p(u | Ce) up to a constant, per chain.  Wishart likelihood + Gaussian / log-normal priors."""
        B, lp = self.unpack(u)
        C = B @ B.transpose(-1, -2) + torch.diag_embed(torch.exp(lp))
        # cholesky_ex (not cholesky): a diverging leapfrog step can occasionally send lp very negative
        # (exp(lp) underflows), leaving C numerically singular/indefinite for that one chain. cholesky_ex
        # reports failure per batch element via `info` instead of raising, so we swap just that element's
        # C for a placeholder (identity) before the real factorization -- keeps every actual computed value
        # finite (no NaN/inf feeding other chains' gradients through autograd) -- then force ITS potential to
        # +inf below, which the existing isfinite guard on logacc already treats as an automatic HMC reject.
        # upper=True + transpose, not the default lower=True: this system's generic (non-MKL/OpenBLAS) LAPACK
        # also has a DPOTRF('L') bug that spuriously fails on genuinely PD matrices at certain sizes/conditioning
        # (diagnosed earlier this session for scsi.py's _fit_init_prior; upper=True is unaffected).
        _, info = torch.linalg.cholesky_ex(C, upper=True)
        bad = info != 0
        eye = torch.eye(self.d, dtype=C.dtype, device=C.device).expand_as(C)
        C_safe = torch.where(bad[..., None, None], eye, C)
        L = torch.linalg.cholesky(C_safe, upper=True).transpose(-1, -2)
        logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        tr = torch.diagonal(torch.cholesky_solve(Ce, L), dim1=-2, dim2=-1).sum(-1)
        nll = 0.5 * self.N * (tr + logdet)
        pri = 0.5 * (((B[..., 0] - self.m1) / self.s1) ** 2).sum(-1) + 0.5 * ((B[..., 1] / self.s2) ** 2).sum(-1) \
            + 0.5 * (((lp - self.lm) / self.ls) ** 2).sum(-1)
        out = nll + pri
        return torch.where(bad, torch.full_like(out, float("inf")), out)

    def _U_and_grad(self, u, Ce):
        """ Compute potential energy and its gradient"""
        with torch.enable_grad():
            u = u.detach().requires_grad_(True)
            U = self.potential(u, Ce)
            g, = torch.autograd.grad(U.sum(), u)
        return U.detach(), g

    def hmc(self, Ce, n_chains=8, n_keep=256, thin=4, warmup=1500, L=20, target=0.75, gen=None, return_diag=False):
        """Batched HMC: (n_obs x n_chains) independent chains started from prior draws, step size adapted per chain
        during warm-up (stochastic approximation on the acceptance rate) then frozen."""
        Ce = Ce.to(DT)
        device = Ce.device
        M = Ce.shape[0]
        Cer = Ce.repeat_interleave(n_chains, 0)
        n = M * n_chains
        u = self.sample_params(n, gen, device=device)
        eps = torch.full((n,), 0.05, dtype=DT, device=device)
        U, g = self._U_and_grad(u, Cer)
        keep, acc_hist = [], []
        for it in range(warmup + n_keep * thin):
            p = torch.randn(u.shape, dtype=DT, generator=gen).to(device)
            H0 = U + 0.5 * (p ** 2).sum(-1)
            e = eps[:, None] * (1.0 + 0.1 * (2 * torch.rand(n, 1, dtype=DT, generator=gen).to(device) - 1))   # step jitter
            uu, pp, gg = u, p - 0.5 * e * g, g
            # Leapfrog scheme
            for l in range(L):
                #Update position
                uu = uu + e * pp
                Uu, gg = self._U_and_grad(uu, Cer)
                if l < L - 1:
                    #Update momentum
                    pp = pp - e * gg
            pp = pp - 0.5 * e * gg
            #Energy difference
            H1 = Uu + 0.5 * (pp ** 2).sum(-1)
            logacc = H0 - H1
            logacc = torch.where(torch.isfinite(logacc), logacc, torch.full_like(logacc, -1e30))
            acc = torch.log(torch.rand(n, dtype=DT, generator=gen).to(device)) < logacc
            u = torch.where(acc[:, None], uu, u)
            U = torch.where(acc, Uu, U)
            g = torch.where(acc[:, None], gg, g)
            pacc = torch.exp(torch.clamp(logacc, max=0.0))
            if it < warmup:
                eps = (eps * torch.exp(0.05 * (pacc - target))).clamp(1e-3, 0.5)
            else:
                acc_hist.append(float(acc.double().mean()))
                if (it - warmup) % thin == thin - 1:
                    keep.append(self.C_from(u))
        Cs = torch.stack(keep, 1)                                         # (n, n_keep, d, d)
        Cs = Cs.reshape(M, n_chains * n_keep, self.d, self.d)
        if return_diag:
            return Cs, dict(acc=float(np.mean(acc_hist)), eps_median=float(eps.median()), n_chains=n_chains, n_keep=n_keep)
        return Cs

    ## Sample posterior with HMC
    def ref_posterior(self, Ce, n_draws, gen=None, warmup=1500):
        n_chains = 8
        Cs = self.hmc(Ce, n_chains=n_chains, n_keep=int(math.ceil(n_draws / n_chains)), warmup=warmup, gen=gen)
        # shuffle so that a prefix is a mixture of chains
        idx = torch.randperm(Cs.shape[1], generator=gen).to(Cs.device)
        return Cs[:, idx][:, :n_draws]


# --------------------------------------------------------------------------- #
# shared helpers for the WishartToy / IsoDiagToy / HeteroskedasticToy sampling + HMC below
# --------------------------------------------------------------------------- #

def cholesky_qr(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """(..., p, r) with p >= r -> (..., p, r) with orthonormal columns."""
    G64 = G.to(torch.float64)
    L = torch.linalg.cholesky(G64.transpose(-1, -2) @ G64 + eps * torch.eye(G64.shape[-1], device=G64.device, dtype=G64.dtype))  # GᵀG = L Lᵀ
    Q = torch.linalg.solve_triangular(L, G64.transpose(-1, -2), upper=False)  # L⁻¹ Gᵀ = Qᵀ
    return Q.transpose(-1, -2).to(G.dtype)


def _Phi(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def unconstrain_bounded(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """(a,b) -> R, logit transform."""
    z = ((x - a) / (b - a)).clamp(1e-6, 1 - 1e-6)
    return torch.log(z) - torch.log(1 - z)


def constrain_bounded(v: torch.Tensor, a: float, b: float):
    """R -> (a,b), sigmoid transform. Returns (x, sigmoid(v)) -- the latter reused for the Jacobian."""
    s = torch.sigmoid(v)
    return a + (b - a) * s, s


def trunc_normal(shape, mean, std, a, b, gen=None):
    """Truncated N(mean, std^2) on [a, b] by inverse CDF, drawing from `gen` (torch.nn.init.trunc_normal_ ignores
    any generator and uses the global RNG -- which made the iso_diag test set non-reproducible across processes)."""
    sq2 = math.sqrt(2.0)
    lo = 0.5 * (1 + math.erf((a - mean) / (std * sq2)))
    hi = 0.5 * (1 + math.erf((b - mean) / (std * sq2)))
    u = lo + (hi - lo) * torch.rand(shape, dtype=DT, generator=gen)
    x = mean + std * sq2 * torch.erfinv(2 * u - 1)
    return x.clamp(a, b)


def trunc_normal_logprob(v: torch.Tensor, mean: float, std: float, a: float, b: float) -> torch.Tensor:
    """log density of x = constrain_bounded(v,a,b) under TruncNormal(mean,std,[a,b]), with
    respect to the UNCONSTRAINED v (i.e. includes the |dx/dv| Jacobian)."""
    x, s = constrain_bounded(v, a, b)
    log_phi = -0.5 * ((x - mean) / std) ** 2 - math.log(std) - 0.5 * math.log(2 * math.pi)
    Z = (_Phi(torch.tensor((b - mean) / std, dtype=v.dtype, device=v.device))
         - _Phi(torch.tensor((a - mean) / std, dtype=v.dtype, device=v.device)))
    log_jac = torch.log((b - a) * s * (1 - s) + 1e-300)
    return log_phi - torch.log(Z) + log_jac


def unif_logit_logprob(v: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """log density of x = constrain_bounded(v,a,b) under Uniform(a,b), wrt the unconstrained v.
    For a uniform target the (b-a) normalization cancels exactly against the logit Jacobian's own
    (b-a) factor, leaving just log(s) + log(1-s) (s = sigmoid(v)) -- simpler than trunc_normal_logprob."""
    s = torch.sigmoid(v)
    return torch.log(s + 1e-300) + torch.log(1 - s + 1e-300)


def wishart_nll(C: torch.Tensor, N: int, Ce: torch.Tensor) -> torch.Tensor:
    """-log p(Ce | C) up to a C-independent constant, standard Wishart-channel likelihood."""
    # cholesky_ex (not cholesky): a diverging leapfrog step can occasionally leave C numerically
    # singular/indefinite for one batch element. cholesky_ex reports failure per element via `info`
    # instead of raising, so we swap just that element's C for a placeholder (identity) before the
    # real factorization -- keeps every computed value finite (no NaN/inf leaking into other
    # elements' gradients through autograd) -- then return +inf for it: callers' HMC loops already
    # treat a non-finite potential as an automatic Metropolis reject (same fix as FactorToy.potential,
    # generalized here since WishartToy/IsoDiagToy/HeteroskedasticToy all share this).
    d = C.shape[-1]
    _, info = torch.linalg.cholesky_ex(C, upper=True)
    bad = info != 0
    eye = torch.eye(d, dtype=C.dtype, device=C.device).expand_as(C)
    C_safe = torch.where(bad[..., None, None], eye, C)
    L = torch.linalg.cholesky(C_safe, upper=True).transpose(-1, -2)
    logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    tr = torch.diagonal(torch.cholesky_solve(Ce, L), dim1=-2, dim2=-1).sum(-1)
    out = 0.5 * N * (tr + logdet)
    return torch.where(bad, torch.full_like(out, float("inf")), out)


def givens_step(U: torch.Tensor, energy_fn, step_size: float, gen=None):
    """One batched Metropolis step on the Stiefel manifold: U' = G(i,j,theta) @ U, (i,j)
    random coordinates, theta ~ N(0,step_size^2). Left-multiplication by a random
    orthogonal matrix preserves Haar measure on the Stiefel manifold, so this proposal is
    symmetric -- plain Metropolis ratio min(1, exp(E0-E1)), no Hastings term, no U-prior
    term needed (uniform/Haar). U: (B,p,r). Returns (U_new, accepted (B,)).
    """
    B, p, r = U.shape
    device, dtype = U.device, U.dtype
    i, j = torch.randperm(p, generator=gen)[:2].tolist()
    # gen is always a CPU generator in this codebase: draw on CPU then move to U's device (same pattern
    # as scsi.wishart_channel) -- torch.randn/rand error if device and generator disagree.
    theta = torch.randn(B, dtype=dtype, generator=gen).to(device) * step_size
    c, s = torch.cos(theta), torch.sin(theta)
    G = torch.eye(p, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    G[:, i, i] = c; G[:, i, j] = -s
    G[:, j, i] = s; G[:, j, j] = c
    U_prop = G @ U
    E0 = energy_fn(U)
    E1 = energy_fn(U_prop)
    log_accept = torch.clamp(E0 - E1, max=0.0)
    accept = torch.log(torch.rand(B, dtype=dtype, generator=gen).to(device)) < log_accept
    U_new = torch.where(accept.view(B, 1, 1), U_prop, U)
    return U_new, accept


# --------------------------------------------------------------------------- #
# WishartToy: C ~ Wishart(nu0, I) -- closed-form Bartlett/Cholesky prior density,
# plain (non-hybrid) HMC on an unconstrained Cholesky-factor parametrization.
# --------------------------------------------------------------------------- #

class WishartToy:
    exact = False
    name = "wishart"

    def __init__(self, name, p, N, M, nu0=None):
        self.name, self.p, self.N, self.M = name, p, N, M
        self.d = p                                                    # alias: toy_train.py and the evaluation scripts expect `.d`
        self.nu0 = nu0 if nu0 is not None else p + 10
        tri = torch.tril_indices(p, p, offset=-1)
        self.tri_i, self.tri_j = tri[0], tri[1]
        self.d_off = self.tri_i.shape[0]
        self.dim = p + self.d_off

    def sample_prior(self, n, gen=None):
        """C ~ Wishart(nu0, I_p): G (p, nu0) standard normal, C = G G^T."""
        G = torch.randn(n, self.p, int(self.nu0), dtype=DT, generator=gen)
        return G @ G.transpose(-1, -2)

    def unpack(self, u):
        p = self.p
        L = torch.diag_embed(torch.exp(u[..., :p]))
        L[..., self.tri_i, self.tri_j] = u[..., p:]
        return L

    def C_from(self, u):
        L = self.unpack(u)
        return L @ L.transpose(-1, -2)

    def sample_params(self, n, gen=None, device=None):
        u_diag = torch.empty(n, self.p, dtype=DT)
        for i in range(self.p):
            df = float(self.nu0 - i)
            x = torch.distributions.Chi2(torch.tensor(df, dtype=DT)).sample((n,))
            u_diag[:, i] = 0.5 * torch.log(x)
        u_off = torch.randn(n, self.d_off, dtype=DT, generator=gen)
        out = torch.cat([u_diag, u_off], dim=-1)
        return out.to(device) if device is not None else out

    def log_prior(self, u):
        p = self.p
        lp = torch.zeros(u.shape[:-1], dtype=DT, device=u.device)
        for i in range(p):
            df = float(self.nu0 - i)
            chi2 = torch.distributions.Chi2(torch.tensor(df, dtype=DT))
            udi = u[..., i]
            lp = lp + chi2.log_prob(torch.exp(2 * udi)) + math.log(2.0) + 2 * udi
        u_off = u[..., p:]
        lp = lp + (-0.5 * u_off ** 2 - 0.5 * math.log(2 * math.pi)).sum(-1)
        return lp

    def potential(self, u, Ce):
        return wishart_nll(self.C_from(u), self.N, Ce) - self.log_prior(u)

    def _U_and_grad(self, u, Ce):
        with torch.enable_grad():
            u = u.detach().requires_grad_(True)
            pot = self.potential(u, Ce)
            g, = torch.autograd.grad(pot.sum(), u)
        return pot.detach(), g

    def hmc(self, Ce, n_chains=8, n_keep=256, thin=4, warmup=1500, L=20, target=0.75, gen=None, return_diag=False):
        Ce = Ce.to(DT)
        device = Ce.device
        M = Ce.shape[0]
        Cer = Ce.repeat_interleave(n_chains, 0)
        n = M * n_chains
        u = self.sample_params(n, gen, device=device)
        eps = torch.full((n,), 0.05, dtype=DT, device=device)
        pot, g = self._U_and_grad(u, Cer)
        keep, acc_hist = [], []
        for it in range(warmup + n_keep * thin):
            p_mom = torch.randn(u.shape, dtype=DT, generator=gen).to(device)
            H0 = pot + 0.5 * (p_mom ** 2).sum(-1)
            e = eps[:, None] * (1.0 + 0.1 * (2 * torch.rand(n, 1, dtype=DT, generator=gen).to(device) - 1))
            uu, pp, gg = u, p_mom - 0.5 * e * g, g
            for l in range(L):
                uu = uu + e * pp
                pot_u, gg = self._U_and_grad(uu, Cer)
                if l < L - 1:
                    pp = pp - e * gg
            pp = pp - 0.5 * e * gg
            H1 = pot_u + 0.5 * (pp ** 2).sum(-1)
            logacc = torch.clamp(H0 - H1, max=0.0)
            logacc = torch.where(torch.isfinite(logacc), logacc, torch.full_like(logacc, -1e30))
            accept = torch.log(torch.rand(n, dtype=DT, generator=gen).to(device)) < logacc
            u = torch.where(accept[:, None], uu, u)
            pot = torch.where(accept, pot_u, pot)
            g = torch.where(accept[:, None], gg, g)
            pacc = torch.exp(torch.clamp(logacc, max=0.0))
            if it < warmup:
                eps = (eps * torch.exp(0.05 * (pacc - target))).clamp(1e-4, 0.5)
            else:
                acc_hist.append(float(accept.double().mean()))
                if (it - warmup) % thin == thin - 1:
                    keep.append(self.C_from(u))
        Cs = torch.stack(keep, 1).reshape(M, n_chains * n_keep, self.p, self.p)
        if return_diag:
            return Cs, dict(acc=float(sum(acc_hist) / len(acc_hist)), eps_median=float(eps.median()),
                             n_chains=n_chains, n_keep=n_keep)
        return Cs

    def ref_posterior(self, Ce, n_draws, gen=None, warmup=1500):
        n_chains = 8
        Cs = self.hmc(Ce, n_chains=n_chains, n_keep=int(math.ceil(n_draws / n_chains)), warmup=warmup, gen=gen)
        idx = torch.randperm(Cs.shape[1], generator=gen).to(Cs.device)
        return Cs[:, idx][:, :n_draws]


# --------------------------------------------------------------------------- #
# AR1Toy: C(rho)_ij = rho^|i-j|, rho ~ Uniform(0.3, 1.0) (sample_ar1's law, fixed range,
# not configurable). A single scalar parameter -> exact 1D-quadrature reference posterior,
# no MCMC needed.
# --------------------------------------------------------------------------- #

class AR1Toy:
    # not conjugate, but w1.get_reference dispatches on `exact` to decide ref_posterior() vs
    # hmc(): AR1Toy only has the former (direct quadrature sampler, no MCMC/R-hat needed), so this
    # must be True for that dispatch to call the right method.
    exact = True

    def __init__(self, name, p, N, M, rho_min=0.3, rho_max=1.0, n_grid=4001):
        self.name, self.p, self.N, self.M = name, p, N, M
        self.d = p                                                    # alias: toy_train.py and the evaluation scripts expect `.d`
        self.rho_min, self.rho_max, self.n_grid = rho_min, rho_max, n_grid
        self._idx_diff = None                                         # lazily built (p,p) |i-j| matrix, cached per device/dtype

    def _diff(self, device, dtype):
        if self._idx_diff is None or self._idx_diff.device != device or self._idx_diff.dtype != dtype:
            idx = torch.arange(self.p, device=device, dtype=dtype)
            self._idx_diff = torch.abs(idx[:, None] - idx[None, :])
        return self._idx_diff

    def C_from_rho(self, rho: torch.Tensor) -> torch.Tensor:
        """rho: (...,) -> C: (...,p,p)."""
        diff = self._diff(rho.device, rho.dtype)
        return rho[..., None, None] ** diff

    def sample_prior(self, n, gen=None):
        rho = torch.rand(n, dtype=DT, generator=gen) * (self.rho_max - self.rho_min) + self.rho_min
        return self.C_from_rho(rho)

    def ref_posterior(self, Ce, n_draws, gen=None):
        """Exact (up to grid resolution): p(rho | Ce) ~ Uniform(rho) x Wishart-channel(Ce | C(rho)).
        Grid-evaluate the log-posterior once (shared across all M test observations via a single
        batched Cholesky of the grid), normalize per observation, sample by inverse-CDF."""
        Ce = Ce.to(DT)
        M = Ce.shape[0]
        device = Ce.device
        # exclude the exact rho=1.0 endpoint: there C(rho) degenerates to the all-ones rank-1 matrix
        # (singular, not PD) -- consistent with sample_ar1's own torch.rand()-based draw, which is
        # likewise exclusive of 1.0.
        grid = torch.linspace(self.rho_min, self.rho_max, self.n_grid + 1, dtype=DT, device=device)[:-1]
        Cg = self.C_from_rho(grid)                                    # (n_grid, p, p)
        Lg = torch.linalg.cholesky(Cg, upper=True).transpose(-1, -2)  # (n_grid, p, p) -- upper=True: see scsi.py's DPOTRF('L') note
        logdet_g = 2 * torch.log(torch.diagonal(Lg, dim1=-2, dim2=-1)).sum(-1)   # (n_grid,)
        Lg_b = Lg.unsqueeze(0).expand(M, -1, -1, -1).reshape(-1, self.p, self.p)
        Ce_b = Ce.unsqueeze(1).expand(-1, self.n_grid, -1, -1).reshape(-1, self.p, self.p)
        tr = torch.diagonal(torch.cholesky_solve(Ce_b, Lg_b), dim1=-2, dim2=-1).sum(-1).reshape(M, self.n_grid)
        nll = 0.5 * self.N * (tr + logdet_g[None, :])                 # (M, n_grid) -- Wishart-channel NLL, uniform prior is a constant
        logp = -nll
        logp = logp - logp.max(dim=1, keepdim=True).values
        w = torch.exp(logp)
        cdf = torch.cumsum(w, dim=1)
        cdf = cdf / cdf[:, -1:]
        u = torch.rand(M, n_draws, dtype=DT, generator=gen).to(device)  # gen is CPU; draw on CPU then move (see wishart_channel)
        gi = torch.searchsorted(cdf.contiguous(), u.contiguous()).clamp(max=self.n_grid - 1)
        rho_draws = grid[gi]                                          # (M, n_draws)
        return self.C_from_rho(rho_draws)                             # (M, n_draws, p, p)


# --------------------------------------------------------------------------- #
# HeteroskedasticToy: C = F F^T + diag(d) -- F (p x r) unconstrained Gaussian, columns
# independently rescaled by a loading-strength ~ Uniform(scale_min,scale_max); d ~
# Uniform(d_min,d_max) idiosyncratic. No orthogonality constraint on F (unlike IsoDiag's
# U), so plain HMC on a single flat unconstrained parameter vector is exact -- same structure as
# WishartToy's HMC, just a different potential/C_from (no Givens/Stiefel step needed).
# --------------------------------------------------------------------------- #

class HeteroskedasticToy:
    exact = False

    def __init__(self, name, p, N, M, r=5, scale_min=1.0, scale_max=4.0, d_min=0.5, d_max=5.0):
        self.name, self.p, self.N, self.M, self.r = name, p, N, M, r
        self.d = p                                                    # alias: toy_train.py and the evaluation scripts expect `.d`
        self.scale_min, self.scale_max = scale_min, scale_max
        self.d_min, self.d_max = d_min, d_max

    def sample_prior(self, n, gen=None):
        """M = F F^T + diag(d): F (p,r) ~ N(0,1) per column rescaled by a loading-strength
        scale ~ Uniform(scale_min,scale_max); d ~ Uniform(d_min,d_max) idiosyncratic.
        Args:
            n (int): number of covariance matrix to draw.
            gen: Pseudo-random generator.
        """
        scale = self.scale_min + (self.scale_max - self.scale_min) * torch.rand(n, 1, self.r, dtype=DT, generator=gen)
        F = torch.randn(n, self.p, self.r, dtype=DT, generator=gen) * scale
        d = self.d_min + (self.d_max - self.d_min) * torch.rand(n, self.p, dtype=DT, generator=gen)
        return F @ F.transpose(-1, -2) + torch.diag_embed(d)

    def unpack(self, u):
        p, r = self.p, self.r
        Z = u[..., :p * r].reshape(*u.shape[:-1], p, r)
        v_scale = u[..., p * r:p * r + r]
        v_d = u[..., p * r + r:]
        return Z, v_scale, v_d

    def C_from(self, u):
        Z, v_scale, v_d = self.unpack(u)
        scale, _ = constrain_bounded(v_scale, self.scale_min, self.scale_max)
        d_diag, _ = constrain_bounded(v_d, self.d_min, self.d_max)
        F = Z * scale.unsqueeze(-2)
        return F @ F.transpose(-1, -2) + torch.diag_embed(d_diag)

    def sample_params(self, n, gen=None, device=None):
        Z = torch.randn(n, self.p * self.r, dtype=DT, generator=gen)
        v_scale = torch.rand(n, self.r, dtype=DT, generator=gen) * 4 - 2     # spread init in logit space
        v_d = torch.rand(n, self.p, dtype=DT, generator=gen) * 4 - 2
        out = torch.cat([Z, v_scale, v_d], dim=-1)
        return out.to(device) if device is not None else out

    def log_prior(self, u):
        Z, v_scale, v_d = self.unpack(u)
        lp = (-0.5 * (Z ** 2).sum((-1, -2)) - 0.5 * self.p * self.r * math.log(2 * math.pi)
              + unif_logit_logprob(v_scale, self.scale_min, self.scale_max).sum(-1)
              + unif_logit_logprob(v_d, self.d_min, self.d_max).sum(-1))
        return lp

    def potential(self, u, Ce):
        return wishart_nll(self.C_from(u), self.N, Ce) - self.log_prior(u)

    def _U_and_grad(self, u, Ce):
        with torch.enable_grad():
            u = u.detach().requires_grad_(True)
            pot = self.potential(u, Ce)
            g, = torch.autograd.grad(pot.sum(), u)
        return pot.detach(), g

    def hmc(self, Ce, n_chains=8, n_keep=256, thin=4, warmup=1500, L=20, target=0.75, gen=None, return_diag=False):
        Ce = Ce.to(DT)
        device = Ce.device
        M = Ce.shape[0]
        Cer = Ce.repeat_interleave(n_chains, 0)
        n = M * n_chains
        u = self.sample_params(n, gen, device=device)
        eps = torch.full((n,), 0.02, dtype=DT, device=device)
        pot, g = self._U_and_grad(u, Cer)
        keep, acc_hist = [], []
        for it in range(warmup + n_keep * thin):
            p_mom = torch.randn(u.shape, dtype=DT, generator=gen).to(device)
            H0 = pot + 0.5 * (p_mom ** 2).sum(-1)
            e = eps[:, None] * (1.0 + 0.1 * (2 * torch.rand(n, 1, dtype=DT, generator=gen).to(device) - 1))
            uu, pp, gg = u, p_mom - 0.5 * e * g, g
            for l in range(L):
                uu = uu + e * pp
                pot_u, gg = self._U_and_grad(uu, Cer)
                if l < L - 1:
                    pp = pp - e * gg
            pp = pp - 0.5 * e * gg
            H1 = pot_u + 0.5 * (pp ** 2).sum(-1)
            logacc = torch.clamp(H0 - H1, max=0.0)
            logacc = torch.where(torch.isfinite(logacc), logacc, torch.full_like(logacc, -1e30))
            accept = torch.log(torch.rand(n, dtype=DT, generator=gen).to(device)) < logacc
            u = torch.where(accept[:, None], uu, u)
            pot = torch.where(accept, pot_u, pot)
            g = torch.where(accept[:, None], gg, g)
            pacc = torch.exp(torch.clamp(logacc, max=0.0))
            if it < warmup:
                eps = (eps * torch.exp(0.05 * (pacc - target))).clamp(1e-4, 0.5)
            else:
                acc_hist.append(float(accept.double().mean()))
                if (it - warmup) % thin == thin - 1:
                    keep.append(self.C_from(u))
        Cs = torch.stack(keep, 1).reshape(M, n_chains * n_keep, self.p, self.p)
        if return_diag:
            return Cs, dict(acc=float(sum(acc_hist) / len(acc_hist)), eps_median=float(eps.median()),
                             n_chains=n_chains, n_keep=n_keep)
        return Cs

    def ref_posterior(self, Ce, n_draws, gen=None, warmup=1500):
        n_chains = 8
        Cs = self.hmc(Ce, n_chains=n_chains, n_keep=int(math.ceil(n_draws / n_chains)), warmup=warmup, gen=gen)
        idx = torch.randperm(Cs.shape[1], generator=gen).to(Cs.device)
        return Cs[:, idx][:, :n_draws]


# --------------------------------------------------------------------------------------------- #
class DiagToy:
    """C = diag(exp(u)), u ~ N(lmean, lstd^2) i.i.d. per component -- the simplest non-trivial toy: no loading
    matrix, no orthogonal group, flat HMC over p unconstrained log-variances only. Ce = wishart_channel(C, N, .)
    still couples all p entries through off-diagonal Wishart noise, so the posterior isn't simply a product of
    p independent 1-D posteriors and a real (if minimal) HMC sampler is still needed. Added as a control to
    isolate whether the N=40-vs-N=28 HMC-difficulty gap seen in FactorToy/HeteroskedasticToy is driven by their
    correlation-fitting geometry (loading matrices/ridges) or is a more generic property of higher N."""
    exact = False

    def __init__(self, name, p, N, M, lmean=0.0, lstd=0.5):
        self.name, self.p, self.N, self.M = name, p, N, M
        self.d = p                                                    # alias: toy_train.py and the evaluation scripts expect `.d`
        self.lmean, self.lstd = lmean, lstd

    def sample_prior(self, n, gen=None):
        u = self.lmean + self.lstd * torch.randn(n, self.p, dtype=DT, generator=gen)
        return torch.diag_embed(torch.exp(u))

    def sample_params(self, n, gen=None, device=None):
        u = self.lmean + self.lstd * torch.randn(n, self.p, dtype=DT, generator=gen)
        return u.to(device) if device is not None else u

    def C_from(self, u):
        return torch.diag_embed(torch.exp(u))

    def potential(self, u, Ce):
        nll = wishart_nll(self.C_from(u), self.N, Ce)
        pri = 0.5 * (((u - self.lmean) / self.lstd) ** 2).sum(-1)
        return nll + pri

    def _U_and_grad(self, u, Ce):
        with torch.enable_grad():
            u = u.detach().requires_grad_(True)
            pot = self.potential(u, Ce)
            g, = torch.autograd.grad(pot.sum(), u)
        return pot.detach(), g

    def hmc(self, Ce, n_chains=8, n_keep=256, thin=4, warmup=1500, L=20, target=0.75, gen=None, return_diag=False):
        Ce = Ce.to(DT)
        device = Ce.device
        M = Ce.shape[0]
        Cer = Ce.repeat_interleave(n_chains, 0)
        n = M * n_chains
        u = self.sample_params(n, gen, device=device)
        eps = torch.full((n,), 0.05, dtype=DT, device=device)
        U, g = self._U_and_grad(u, Cer)
        keep, acc_hist = [], []
        for it in range(warmup + n_keep * thin):
            p = torch.randn(u.shape, dtype=DT, generator=gen).to(device)
            H0 = U + 0.5 * (p ** 2).sum(-1)
            e = eps[:, None] * (1.0 + 0.1 * (2 * torch.rand(n, 1, dtype=DT, generator=gen).to(device) - 1))
            uu, pp, gg = u, p - 0.5 * e * g, g
            for l in range(L):
                uu = uu + e * pp
                Uu, gg = self._U_and_grad(uu, Cer)
                if l < L - 1:
                    pp = pp - e * gg
            pp = pp - 0.5 * e * gg
            H1 = Uu + 0.5 * (pp ** 2).sum(-1)
            logacc = H0 - H1
            logacc = torch.where(torch.isfinite(logacc), logacc, torch.full_like(logacc, -1e30))
            acc = torch.log(torch.rand(n, dtype=DT, generator=gen).to(device)) < logacc
            u = torch.where(acc[:, None], uu, u)
            U = torch.where(acc, Uu, U)
            g = torch.where(acc[:, None], gg, g)
            pacc = torch.exp(torch.clamp(logacc, max=0.0))
            if it < warmup:
                eps = (eps * torch.exp(0.05 * (pacc - target))).clamp(1e-3, 0.5)
            else:
                acc_hist.append(float(acc.double().mean()))
                if (it - warmup) % thin == thin - 1:
                    keep.append(self.C_from(u))
        Cs = torch.stack(keep, 1).reshape(M, n_chains * n_keep, self.p, self.p)
        if return_diag:
            return Cs, dict(acc=float(np.mean(acc_hist)), eps_median=float(eps.median()), n_chains=n_chains, n_keep=n_keep)
        return Cs

    def ref_posterior(self, Ce, n_draws, gen=None, warmup=1500):
        n_chains = 8
        Cs = self.hmc(Ce, n_chains=n_chains, n_keep=int(math.ceil(n_draws / n_chains)), warmup=warmup, gen=gen)
        idx = torch.randperm(Cs.shape[1], generator=gen).to(Cs.device)
        return Cs[:, idx][:, :n_draws]


#### Declare experiments
TOYS = {
    #"iw_ri_d4":    lambda: IWToy("iw_ri_d4", 4, 8, 5000, "ri", 12.0),
    "factor_d8":    lambda: FactorToy("factor_d8", 8, 28, 3000),  # old finite-data config (N=28, M=3000)
    "iw_ri_d8":     lambda: IWToy("iw_ri_d8", 8, 28, 3000, "ri", 16.0),
    "iw_nonri_d8":  lambda: IWToy("iw_nonri_d8", 8, 28, 3000, "nonri", 16.0),
    "iso_diag_d8":  lambda: IsoDiagToy("iso_diag_d8", 8, 28, 3000),
    "ar1_d8":       lambda: AR1Toy("ar1_d8", 8, 28, 3000),
    "hetero_d8":    lambda: HeteroskedasticToy("hetero_d8", 8, 28, 3000, r=3,
                                                 scale_min=0.4, scale_max=0.8, d_min=0.3, d_max=1.0),
    "diag_d8":      lambda: DiagToy("diag_d8", 8, 28, 3000),
    "iw_ri_d8_inf":    lambda: IWToy("iw_ri_d8_inf", 8, 40, -1, "ri", 16.0),
    "iw_ri_d16_inf":   lambda: IWToy("iw_ri_d16_inf", 16, 80, -1, "ri", 32.0),  # scaling test: p=136 coord-space (>>32 LAPACK-bug threshold)
    "iw_nonri_d8_inf": lambda: IWToy("iw_nonri_d8_inf", 8, 40, -1, "nonri", 16.0),
    "iw_nonri_d16_inf": lambda: IWToy("iw_nonri_d16_inf", 16, 80, -1, "nonri", 32.0),  # same N/d, kappa0/d ratios as iw_ri_d16_inf
    "factor_d8_inf":   lambda: FactorToy("factor_d8_inf", 8, 40, -1),
    "factor_d16_inf":  lambda: FactorToy("factor_d16_inf", 16, 80, -1),
    "iso_diag_d8_inf": lambda: IsoDiagToy("iso_diag_d8_inf", 8, 40, -1),
    "iso_diag_d16_inf": lambda: IsoDiagToy("iso_diag_d16_inf", 16, 80, -1),
    "ar1_d8_inf":      lambda: AR1Toy("ar1_d8_inf", 8, 40, -1),
    "ar1_d16_inf":     lambda: AR1Toy("ar1_d16_inf", 16, 80, -1, rho_min=0.3, rho_max=0.9),
    "hetero_d8_inf":   lambda: HeteroskedasticToy("hetero_d8_inf", 8, 40, -1, r=3,
                                                    scale_min=0.4, scale_max=0.8, d_min=0.3, d_max=1.0),
    "hetero_d16_inf":  lambda: HeteroskedasticToy("hetero_d16_inf", 16, 80, -1, r=3,
                                                    scale_min=0.4, scale_max=0.8, d_min=0.3, d_max=1.0),
    "iw_ri_d20_inf":    lambda: IWToy("iw_ri_d20_inf", 20, 100, -1, "ri", 40.0),      # same d/N=0.2 ratio, kappa0/d=2.0 as iw_ri_d16_inf
    "iw_nonri_d20_inf": lambda: IWToy("iw_nonri_d20_inf", 20, 100, -1, "nonri", 40.0),
    "iw_ri_d30_inf":    lambda: IWToy("iw_ri_d30_inf", 30, 150, -1, "ri", 60.0),      # same d/N=0.2 ratio, kappa0/d=2.0 as iw_ri_d20_inf
    "iw_nonri_d30_inf": lambda: IWToy("iw_nonri_d30_inf", 30, 150, -1, "nonri", 60.0),
    "hetero_d20_inf":  lambda: HeteroskedasticToy("hetero_d20_inf", 20, 100, -1, r=3,  # same d/N=0.2 ratio as hetero_d16_inf
                                                    scale_min=0.4, scale_max=0.8, d_min=0.3, d_max=1.0),
    "hetero_d20_inf_wide":  lambda: HeteroskedasticToy("hetero_d20_inf_wide", 20, 100, -1, r=3,  # same as hetero_d20_inf but
                                                    scale_min=0.4, scale_max=0.8, d_min=0.1, d_max=3.0),  # wider d_min/d_max: test
                                                    # whether spacing out the idiosyncratic-variance order statistics (the smallest
                                                    # eigenvalue is the min of p draws from U(d_min,d_max), which concentrates near
                                                    # d_min as p grows) eases learning of the bottom-eigenvector directional variance
    "diag_d8_inf":     lambda: DiagToy("diag_d8_inf", 8, 40, -1),
    "diag_d16_inf":    lambda: DiagToy("diag_d16_inf", 16, 80, -1),
    # finite-data counterparts of the d=8 production toys: same priors and N=40, one fixed training ensemble of M=5000 Ce
    "iw_ri_d8_M5000":    lambda: IWToy("iw_ri_d8_M5000", 8, 40, 5000, "ri", 16.0),
    "iw_nonri_d8_M5000": lambda: IWToy("iw_nonri_d8_M5000", 8, 40, 5000, "nonri", 16.0),
    "factor_d8_M5000":   lambda: FactorToy("factor_d8_M5000", 8, 40, 5000),
    "iso_diag_d8_M5000": lambda: IsoDiagToy("iso_diag_d8_M5000", 8, 40, 5000),
    "ar1_d8_M5000":      lambda: AR1Toy("ar1_d8_M5000", 8, 40, 5000),
    "hetero_d8_M5000":   lambda: HeteroskedasticToy("hetero_d8_M5000", 8, 40, 5000, r=3,
                                                      scale_min=0.4, scale_max=0.8, d_min=0.3, d_max=1.0),
    "diag_d8_M5000":     lambda: DiagToy("diag_d8_M5000", 8, 40, 5000),
}


def make_toy(name):
    return TOYS[name]()



# --------------------------------------------------------------------------- #
# IsoDiagToy: C = U diag(spikes) U^T, r = p ALWAYS (full rank, no diagonal term).
# --------------------------------------------------------------------------- #

class IsoDiagToy:
    exact = False

    def __init__(self, name, p, N, M, spike_mean=3.0, spike_std=1.0):
        if spike_mean < spike_std:
            raise ValueError("need spike_mean >= spike_std")
        self.name, self.p, self.N, self.M = name, p, N, M
        self.d = p                                                    # alias: toy_train.py and the evaluation scripts expect `.d`
        self.r = p   # always full rank -- no diagonal floor to support r<p
        self.spike_mean, self.spike_std = spike_mean, spike_std
        self.spike_a, self.spike_b = spike_mean - spike_std, spike_mean + spike_std

    def sample_prior(self, n, gen=None):
        """M = U diag(spikes) U^T: U (p,p) Haar-orthonormal (full rank), spikes genuinely truncated
        normal."""
        U = cholesky_qr(torch.randn(n, self.p, self.p, dtype=DT, generator=gen))
        spikes = trunc_normal((n, self.p), self.spike_mean, self.spike_std, self.spike_a, self.spike_b, gen)
        return self.C_from(U, spikes)

    def C_from(self, U, spikes):
        return (U * spikes.unsqueeze(-2)) @ U.transpose(-1, -2)

    def sample_params(self, n, gen=None, device=None):
        G = torch.randn(n, self.p, self.p, dtype=DT, generator=gen)
        U = cholesky_qr(G)
        spikes = trunc_normal((n, self.p), self.spike_mean, self.spike_std, self.spike_a, self.spike_b, gen)
        v_spikes = unconstrain_bounded(spikes, self.spike_a, self.spike_b)
        if device is not None:
            U, v_spikes = U.to(device), v_spikes.to(device)
        return U, v_spikes

    def _mag_potential(self, v_spikes, U, Ce):
        spikes, _ = constrain_bounded(v_spikes, self.spike_a, self.spike_b)
        C = self.C_from(U, spikes)
        nll = wishart_nll(C, self.N, Ce)
        lp = trunc_normal_logprob(v_spikes, self.spike_mean, self.spike_std, self.spike_a, self.spike_b).sum(-1)
        return nll - lp

    def _mag_U_and_grad(self, v_spikes, U, Ce):
        with torch.enable_grad():
            v_spikes = v_spikes.detach().requires_grad_(True)
            pot = self._mag_potential(v_spikes, U, Ce)
            g, = torch.autograd.grad(pot.sum(), v_spikes)
        return pot.detach(), g

    def hmc(self, Ce, n_chains=8, n_keep=256, thin=4, warmup=1500, L=10, hmc_target=0.75,
            givens_per_sweep=6, givens_step0=0.3, gen=None, return_diag=False):
        Ce = Ce.to(DT)
        device = Ce.device
        M = Ce.shape[0]
        Cer = Ce.repeat_interleave(n_chains, 0)
        n = M * n_chains
        U, v_spikes = self.sample_params(n, gen, device=device)
        eps = torch.full((n,), 0.05, dtype=DT, device=device)
        g_step = torch.full((n,), givens_step0, dtype=DT, device=device)
        pot, g = self._mag_U_and_grad(v_spikes, U, Cer)
        keep, acc_hist_hmc, acc_hist_g = [], [], []
        for it in range(warmup + n_keep * thin):
            def energy_fn(Uprop):
                spikes, _ = constrain_bounded(v_spikes, self.spike_a, self.spike_b)
                return wishart_nll(self.C_from(Uprop, spikes), self.N, Cer)
            for _ in range(givens_per_sweep):
                U, acc_g = givens_step(U, energy_fn, float(g_step.mean()), gen=gen)
                acc_hist_g.append(float(acc_g.double().mean()))
            pot, g = self._mag_U_and_grad(v_spikes, U, Cer)

            p_mom = torch.randn(v_spikes.shape, dtype=DT, generator=gen).to(device)
            H0 = pot + 0.5 * (p_mom ** 2).sum(-1)
            e = eps[:, None] * (1.0 + 0.1 * (2 * torch.rand(n, 1, dtype=DT, generator=gen).to(device) - 1))
            vv, pp, gg = v_spikes, p_mom - 0.5 * e * g, g
            for l in range(L):
                vv = vv + e * pp
                pot_v, gg = self._mag_U_and_grad(vv, U, Cer)
                if l < L - 1:
                    pp = pp - e * gg
            pp = pp - 0.5 * e * gg
            H1 = pot_v + 0.5 * (pp ** 2).sum(-1)
            logacc = torch.clamp(H0 - H1, max=0.0)
            logacc = torch.where(torch.isfinite(logacc), logacc, torch.full_like(logacc, -1e30))
            accept = torch.log(torch.rand(n, dtype=DT, generator=gen).to(device)) < logacc
            v_spikes = torch.where(accept[:, None], vv, v_spikes)
            pot = torch.where(accept, pot_v, pot)
            g = torch.where(accept[:, None], gg, g)
            pacc = torch.exp(torch.clamp(logacc, max=0.0))
            acc_hist_hmc.append(float(accept.double().mean()))
            if it < warmup:
                eps = (eps * torch.exp(0.05 * (pacc - hmc_target))).clamp(1e-4, 0.5)
            else:
                if (it - warmup) % thin == thin - 1:
                    spikes, _ = constrain_bounded(v_spikes, self.spike_a, self.spike_b)
                    keep.append(self.C_from(U, spikes))
        Cs = torch.stack(keep, 1).reshape(M, n_chains * n_keep, self.p, self.p)
        if return_diag:
            return Cs, dict(acc_hmc=float(sum(acc_hist_hmc) / len(acc_hist_hmc)),
                             acc_givens=float(sum(acc_hist_g) / len(acc_hist_g)),
                             eps_median=float(eps.median()), n_chains=n_chains, n_keep=n_keep)
        return Cs

    def ref_posterior(self, Ce, n_draws, gen=None, warmup=1500):
        n_chains = 8
        Cs = self.hmc(Ce, n_chains=n_chains, n_keep=int(math.ceil(n_draws / n_chains)), warmup=warmup, gen=gen)
        idx = torch.randperm(Cs.shape[1], generator=gen).to(Cs.device)
        return Cs[:, idx][:, :n_draws]
