"""Controlled toy models with a *reference* posterior P(C | Ce).

  IWToy     : inverse-Wishart prior (RI or non-RI scale).  Conjugate => exact posterior IW(nu0+N, Psi0+N Ce).
  FactorToy : factor + idiosyncratic covariance  C = B B^T + diag(psi)  (K = 2 factors: a positive-loading
              "market" factor and a mixed-sign second factor).  No conjugacy: the reference posterior is a
              batched Hamiltonian Monte Carlo sampler on (B, log psi), validated by simulation-based calibration.

The learner (SC-SI) only ever sees noisy empirical covariances from these priors.
"""
import math
import numpy as np
import torch

from scsi import DT, wishart_channel
from covutils import sample_iw


class Toy:
    name: str
    d: int
    N: int
    M: int
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
        return sample_iw(self.nu0, self.Psi0, n, gen)

    def ref_posterior(self, Ce, n_draws, gen=None):
        M = Ce.shape[0]
        B = (self.Psi0 + self.N * Ce.to(DT)).repeat_interleave(n_draws, 0)
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

    def sample_params(self, n, gen=None):
        b1 = self.m1 + self.s1 * torch.randn(n, self.d, dtype=DT, generator=gen)
        b2 = self.s2 * torch.randn(n, self.d, dtype=DT, generator=gen)
        lp = self.lm + self.ls * torch.randn(n, self.d, dtype=DT, generator=gen)
        return torch.cat([torch.stack([b1, b2], -1).reshape(n, -1), lp], -1)

    def sample_prior(self, n, gen=None):
        return self.C_from(self.sample_params(n, gen))

    def potential(self, u, Ce):
        """U(u) = -log p(u | Ce) up to a constant, per chain.  Wishart likelihood + Gaussian / log-normal priors."""
        B, lp = self.unpack(u)
        C = B @ B.transpose(-1, -2) + torch.diag_embed(torch.exp(lp))
        L = torch.linalg.cholesky(C)
        logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        tr = torch.diagonal(torch.cholesky_solve(Ce, L), dim1=-2, dim2=-1).sum(-1)
        nll = 0.5 * self.N * (tr + logdet)
        pri = 0.5 * (((B[..., 0] - self.m1) / self.s1) ** 2).sum(-1) + 0.5 * ((B[..., 1] / self.s2) ** 2).sum(-1) \
            + 0.5 * (((lp - self.lm) / self.ls) ** 2).sum(-1)
        return nll + pri

    def _U_and_grad(self, u, Ce):
        with torch.enable_grad():
            u = u.detach().requires_grad_(True)
            U = self.potential(u, Ce)
            g, = torch.autograd.grad(U.sum(), u)
        return U.detach(), g

    def hmc(self, Ce, n_chains=8, n_keep=256, thin=4, warmup=1500, L=20, target=0.75, gen=None, return_diag=False):
        """Batched HMC: (n_obs x n_chains) independent chains started from prior draws, step size adapted per chain
        during warm-up (stochastic approximation on the acceptance rate) then frozen."""
        Ce = Ce.to(DT)
        M = Ce.shape[0]
        Cer = Ce.repeat_interleave(n_chains, 0)
        n = M * n_chains
        u = self.sample_params(n, gen)
        eps = torch.full((n,), 0.05, dtype=DT)
        U, g = self._U_and_grad(u, Cer)
        keep, acc_hist = [], []
        for it in range(warmup + n_keep * thin):
            p = torch.randn(u.shape, dtype=DT, generator=gen)
            H0 = U + 0.5 * (p ** 2).sum(-1)
            e = eps[:, None] * (1.0 + 0.1 * (2 * torch.rand(n, 1, dtype=DT, generator=gen) - 1))   # step jitter
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
            acc = torch.log(torch.rand(n, dtype=DT, generator=gen)) < logacc
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

    def ref_posterior(self, Ce, n_draws, gen=None):
        n_chains = 8
        Cs = self.hmc(Ce, n_chains=n_chains, n_keep=int(math.ceil(n_draws / n_chains)), gen=gen)
        # shuffle so that a prefix is a mixture of chains
        idx = torch.randperm(Cs.shape[1], generator=gen)
        return Cs[:, idx][:, :n_draws]


TOYS = {
    "iw_ri_d4":    lambda: IWToy("iw_ri_d4", 4, 8, 5000, "ri", 12.0),
    "iw_ri_d8":    lambda: IWToy("iw_ri_d8", 8, 40, 2000, "ri", 16.0),
    "iw_nonri_d8": lambda: IWToy("iw_nonri_d8", 8, 28, 2000, "nonri", 16.0),
    "factor_d8":   lambda: FactorToy("factor_d8", 8, 28, 3000),
}


def make_toy(name):
    return TOYS[name]()
