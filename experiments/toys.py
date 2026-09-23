"""Controlled toy models with a *reference* posterior P(C | Ce).

  IWToy     : inverse-Wishart prior (RI or non-RI scale).  Conjugate => exact posterior IW(nu0+N, Psi0+N Ce).
  FactorToy : factor + idiosyncratic covariance  C = B B^T + diag(psi)  (K = 2 factors: a positive-loading
              "market" factor and a mixed-sign second factor).  No conjugacy: the reference posterior is a
              batched Hamiltonian Monte Carlo sampler on (B, log psi), validated by simulation-based calibration.

The learner (SC-SI) only ever sees noisy empirical covariances from these priors.

Device: every toy takes a `device` argument (default 'cpu'); all its own tensors (C0, Psi0, and for
FactorToy the HMC state) live there, and `sample_prior`/`ref_posterior` move whatever Ce they're given
onto that same device. The one fixed piece of randomness that must match across devices for
reproducibility -- the non-RI toy's shared eigenbasis Q -- is drawn on CPU and moved, so the same Q comes
out regardless of `device`.
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

    def __init__(self, name, d, N, M, kind, kappa0, device="cpu"):
        self.name, self.d, self.N, self.M, self.kind, self.kappa0 = name, d, N, M, kind, kappa0
        self.device = torch.device(device)
        self.nu0 = d + 1 + kappa0
        if kind == "ri":
            C0 = torch.eye(d, dtype=DT)
        else:  # non-RI: fixed rotated geometric spectrum (condition number 16), mean eigenvalue 1
            g = torch.Generator().manual_seed(123)   # CPU: keeps the same shared Q regardless of device
            Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=DT, generator=g))
            ev = torch.tensor(np.geomspace(1.0, 16.0, d), dtype=DT)
            ev = ev / ev.mean()
            C0 = Q @ torch.diag(ev) @ Q.T
        self.C0 = C0.to(self.device)
        self.Psi0 = (kappa0 * C0).to(self.device)

    def sample_prior(self, n, gen=None):
        return sample_iw(self.nu0, self.Psi0, n, gen)

    def ref_posterior(self, Ce, n_draws, gen=None):
        Ce = Ce.to(DT).to(self.device)
        M = Ce.shape[0]
        B = (self.Psi0 + self.N * Ce).repeat_interleave(n_draws, 0)
        return sample_iw(self.nu0 + self.N, B, M * n_draws, gen).reshape(M, n_draws, self.d, self.d)


# --------------------------------------------------------------------------------------------- #
class FactorToy(Toy):
    exact = False

    def __init__(self, name, d, N, M, m1=1.0, s1=0.3, s2=0.6, lpsi_mean=math.log(0.4), lpsi_sd=0.4, device="cpu"):
        self.name, self.d, self.N, self.M, self.K = name, d, N, M, 2
        self.device = torch.device(device)
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
        b1 = self.m1 + self.s1 * torch.randn(n, self.d, dtype=DT, device=self.device, generator=gen)
        b2 = self.s2 * torch.randn(n, self.d, dtype=DT, device=self.device, generator=gen)
        lp = self.lm + self.ls * torch.randn(n, self.d, dtype=DT, device=self.device, generator=gen)
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
        during warm-up (stochastic approximation on the acceptance rate) then frozen. Runs on self.device;
        gen, if given, must be a torch.Generator on that device."""
        Ce = Ce.to(DT).to(self.device)
        M = Ce.shape[0]
        Cer = Ce.repeat_interleave(n_chains, 0)
        n = M * n_chains
        u = self.sample_params(n, gen)
        eps = torch.full((n,), 0.05, dtype=DT, device=self.device)
        U, g = self._U_and_grad(u, Cer)
        keep, acc_hist = [], []
        for it in range(warmup + n_keep * thin):
            p = torch.randn(u.shape, dtype=DT, device=self.device, generator=gen)
            H0 = U + 0.5 * (p ** 2).sum(-1)
            e = eps[:, None] * (1.0 + 0.1 * (2 * torch.rand(n, 1, dtype=DT, device=self.device, generator=gen) - 1))   # step jitter
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
            acc = torch.log(torch.rand(n, dtype=DT, device=self.device, generator=gen)) < logacc
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
        idx = torch.randperm(Cs.shape[1], device=self.device, generator=gen)
        return Cs[:, idx][:, :n_draws]


TOYS = {
    "iw_ri_d4":    lambda device="cpu": IWToy("iw_ri_d4", 4, 8, 5000, "ri", 12.0, device=device),
    "iw_ri_d8":    lambda device="cpu": IWToy("iw_ri_d8", 8, 40, 2000, "ri", 16.0, device=device),
    "iw_ri_d20":   lambda device="cpu": IWToy("iw_ri_d20", 20, 100, 2000, "ri", 28.0, device=device),
    "iw_ri_d20_M8000": lambda device="cpu": IWToy("iw_ri_d20_M8000", 20, 100, 8000, "ri", 28.0, device=device),
    "iw_nonri_d8": lambda device="cpu": IWToy("iw_nonri_d8", 8, 28, 2000, "nonri", 16.0, device=device),
    "factor_d8":   lambda device="cpu": FactorToy("factor_d8", 8, 28, 3000, device=device),
}


def make_toy(name, device="cpu"):
    return TOYS[name](device=device)
