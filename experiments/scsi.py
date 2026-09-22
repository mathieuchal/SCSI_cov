"""Self-consistent stochastic interpolants for covariance posterior learning.

Implements Algorithm 1 of "Self-Consistent Stochastic Interpolants for Covariance
Prior and Posterior Learning" (general, matrix-logarithm version):

  * Wishart channel simulator           C'_e = (1/N) C^{1/2} Z Z^T C^{1/2}          (eq. 2)
  * log coordinates                     Y = svec(log C)                              (Sec. 3.1)
  * conditional Brownian interpolant    I_t = (1-t) Y'_e + t Y + sigma (1-t) W_t     (eq. 5)
                                        R_t = Y - Y'_e - sigma W_t
  * Follmer sampler                     dY = [(1+t) b(Y,t,Ye) - Y + Ye] dt
                                             + sigma sqrt(1-t^2) dB                  (eq. 7)
  * self-consistent EM loop             reconstruct -> re-corrupt -> refit           (Alg. 1)

Everything is batched torch; the SDE lives in raw svec(log) coordinates, the network
only sees standardised copies of them.

Device: SCSIConfig.device selects 'cpu' or 'cuda' (or a specific 'cuda:N'). Every tensor the
class allocates internally follows that device, and `fit`/`fit_supervised` move whatever the
caller hands them onto it. A `gen`/`self.gen` torch.Generator must live on the same device as
the tensors it's used with (torch itself enforces this) -- SCSI's own generator is created on
`cfg.device`; a generator you pass in explicitly (e.g. to `sample_posterior`) must match too.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)
DT = torch.float64  # dtype for matrix functions / simulation


def safe_eigh(C: torch.Tensor, chunk: int = 8192):
    """torch.linalg.eigh, chunked over the batch.

    cuSOLVER's batched symmetric eigensolver has an internal batch-size ceiling well below what plain
    CPU LAPACK (called in a loop under the hood) tolerates -- e.g. a batch of 200,000 8x8 matrices
    (SCSIConfig.pool_init) raises CUSOLVER_STATUS_INVALID_VALUE from cusolverDnXsyevBatched on an A100,
    while 8192 succeeds. Chunking is applied on every device (harmless on CPU) so the code path is the
    same regardless of where it runs. Handles arbitrary leading batch dims, as torch.linalg.eigh does."""
    lead = C.shape[:-2]
    n = int(torch.tensor(lead).prod()) if lead else 1
    if n <= chunk:
        return torch.linalg.eigh(C)
    flat = C.reshape(n, *C.shape[-2:])
    ws, Vs = [], []
    for s in range(0, n, chunk):
        w, V = torch.linalg.eigh(flat[s:s + chunk])
        ws.append(w); Vs.append(V)
    d = C.shape[-1]
    return torch.cat(ws).reshape(*lead, d), torch.cat(Vs).reshape(*lead, d, d)


def safe_eigvalsh(C: torch.Tensor, chunk: int = 8192):
    """torch.linalg.eigvalsh, chunked the same way as safe_eigh (eigenvalues only, cheaper when the
    eigenvectors aren't needed -- e.g. scoring a (M,J,d,d) block of posterior draws)."""
    lead = C.shape[:-2]
    n = int(torch.tensor(lead).prod()) if lead else 1
    if n <= chunk:
        return torch.linalg.eigvalsh(C)
    flat = C.reshape(n, *C.shape[-2:])
    ws = [torch.linalg.eigvalsh(flat[s:s + chunk]) for s in range(0, n, chunk)]
    return torch.cat(ws).reshape(*lead, C.shape[-1])


# --------------------------------------------------------------------------- #
# matrix utilities
# --------------------------------------------------------------------------- #
class Sym:
    """svec / logm helpers for a fixed dimension d."""

    def __init__(self, d: int, device="cpu"):
        self.d = d
        self.p = d * (d + 1) // 2
        self.device = torch.device(device)
        iu = torch.triu_indices(d, d, device=self.device)
        self.i0, self.i1 = iu[0], iu[1]
        self.scale = torch.where(self.i0 == self.i1, 1.0, math.sqrt(2.0)).to(DT)

    def svec(self, A: torch.Tensor) -> torch.Tensor:
        return A[..., self.i0, self.i1] * self.scale.to(A.dtype)

    def svec_inv(self, y: torch.Tensor) -> torch.Tensor:
        y = y / self.scale.to(y.dtype)
        A = torch.zeros(*y.shape[:-1], self.d, self.d, dtype=y.dtype, device=y.device)
        A[..., self.i0, self.i1] = y
        A[..., self.i1, self.i0] = y
        return A

    @staticmethod
    def logm(C: torch.Tensor) -> torch.Tensor:
        w, V = safe_eigh(C)
        return (V * torch.log(w.clamp_min(1e-300)).unsqueeze(-2)) @ V.transpose(-1, -2)

    @staticmethod
    def expm(S: torch.Tensor) -> torch.Tensor:
        w, V = safe_eigh(S)
        return (V * torch.exp(w).unsqueeze(-2)) @ V.transpose(-1, -2)

    def encode(self, C: torch.Tensor, eps0: float = 0.0) -> torch.Tensor:
        """Phi(C) = svec(log(C + eps0 I)); eps0 > 0 only needed when N < d."""
        C = C.to(DT)
        if eps0 > 0:
            C = C + eps0 * torch.eye(self.d, dtype=DT, device=C.device)
        return self.svec(self.logm(C))

    def decode(self, Y: torch.Tensor) -> torch.Tensor:
        return self.expm(self.svec_inv(Y.to(DT)))


def wishart_channel(C: torch.Tensor, N: int, gen: torch.Generator | None = None) -> torch.Tensor:
    """Ce = (1/N) L Z Z^T L^T with C = L L^T  ~  K_N(. | C)   (same law as eq. 2).

    gen, if given, must be a torch.Generator on the same device as C."""
    d = C.shape[-1]
    C = C.to(DT)
    L = torch.linalg.cholesky(C)
    Z = torch.randn(*C.shape[:-2], d, N, dtype=DT, device=C.device, generator=gen)
    X = L @ Z
    return X @ X.transpose(-1, -2) / N


def empirical_cov(X: torch.Tensor) -> torch.Tensor:
    """X: (..., d, N) raw samples -> (1/N) X X^T (zero-mean convention of eq. 1)."""
    X = X.to(DT)
    return X @ X.transpose(-1, -2) / X.shape[-1]


# --------------------------------------------------------------------------- #
# drift network
# --------------------------------------------------------------------------- #
class Drift(nn.Module):
    def __init__(self, p: int, mu: torch.Tensor, sd: torch.Tensor, hidden=256, depth=4, nfreq=8):
        super().__init__()
        self.p = p
        self.register_buffer("mu", mu.float())
        self.register_buffer("sd", sd.float())
        self.register_buffer("freq", (2.0 ** torch.arange(nfreq)) * math.pi)
        din = 2 * p + 2 * nfreq + 1
        layers = [nn.Linear(din, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, p)]
        self.net = nn.Sequential(*layers)
        # zero-init last layer: start from b = 0 (scaled)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.out_scale = float(sd.mean())

    def forward(self, t: torch.Tensor, I: torch.Tensor, Ye: torch.Tensor) -> torch.Tensor:
        # t: (B,1)
        zi = (I - self.mu) / self.sd
        ze = (Ye - self.mu) / self.sd
        tf = t * self.freq
        h = torch.cat([zi, ze, t, torch.sin(tf), torch.cos(tf)], dim=-1)
        return t * (I - Ye) + self.net(h) * self.out_scale  # exact limit b_1(y,Ye) = y - Ye


# --------------------------------------------------------------------------- #
# main learner
# --------------------------------------------------------------------------- #
@dataclass
class SCSIConfig:
    N: int
    d: int
    kappa: float = 1.0          # sigma = kappa * sqrt(2 / N)
    hidden: int = 256
    depth: int = 4
    n_outer: int = 6            # self-consistent iterations (K)
    steps_first: int = 4000     # gradient steps for the initial fit
    steps_outer: int = 2000     # gradient steps per subsequent refit
    batch: int = 1024
    lr: float = 1e-3
    lr_end: float = 1e-4
    n_recon: int = 8            # reconstructions per recorded observation per outer iteration
    n_sde_steps: int = 64
    eps0: float = 0.0           # >0 iff N < d
    seed: int = 0
    weight_decay: float = 0.0
    ema: float = 0.999
    log_prior_inflate: float = 1.0  # scale on covariance of the log-Gaussian initial prior
    pool_init: int = 200_000    # size of the pool drawn from the initial prior / supervised prior
    threads: int = 4
    init: str = "raw"          # 'raw' | 'deconv'
    device: str = "cpu"         # 'cpu' | 'cuda' | 'cuda:N'
    verbose: bool = True


class SCSI:
    """Learn P(C | Ce) from an ensemble of empirical covariances only."""

    def __init__(self, cfg: SCSIConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.S = Sym(cfg.d, device=self.device)
        self.sigma = cfg.kappa * math.sqrt(2.0 / cfg.N)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)
        torch.set_num_threads(cfg.threads)
        self.net: Drift | None = None
        self.ema_net: Drift | None = None
        self.hist: list[dict] = []

    # ---- initial sampleable prior: Gaussian in log-coordinates, fitted to the observed Ye ---- #
    def _fit_init_prior(self, Ye: torch.Tensor):
        """Sampleable pi_0: Gaussian in log-coordinates.

        init='raw'    : mean/cov of the observed Ye (over-dispersed by the sampling noise).
        init='deconv' : moment-matching deconvolution using only the known forward channel: the mean is
                        corrected for the bias of log(Ce) and the covariance has the simulated channel-noise
                        covariance (at a typical C) subtracted, then inflated by `log_prior_inflate` and floored.
        Neither uses clean covariances; both are declared, data-informed initialisations."""
        cfg, S = self.cfg, self.S
        m = Ye.mean(0)
        Xc = Ye - m
        cov = Xc.T @ Xc / max(len(Ye) - 1, 1)
        if cfg.init == "deconv":
            for _ in range(3):
                Cty = S.decode(m).expand(20000, cfg.d, cfg.d)
                Yn = S.encode(wishart_channel(Cty, cfg.N, self.gen), cfg.eps0)
                m = Ye.mean(0) - (Yn.mean(0) - m)
            Yn = Yn - Yn.mean(0)
            noise_cov = Yn.T @ Yn / (len(Yn) - 1)
            w, V = torch.linalg.eigh(cov - noise_cov)
            w_floor = 0.05 * torch.linalg.eigvalsh(cov)[-1] * 0 + 0.05 * w.abs().max()
            cov = (V * w.clamp_min(w_floor)) @ V.T
        cov = cov * cfg.log_prior_inflate + 1e-6 * torch.eye(Ye.shape[1], dtype=DT, device=self.device) * cov.diag().mean()
        self._init_m, self._init_L = m, torch.linalg.cholesky(cov)

    def _sample_init_prior(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.S.p, dtype=DT, device=self.device, generator=self.gen)
        return self._init_m + z @ self._init_L.T

    def _pool_sampler(self, pool):
        Yp, Cp = pool
        n = Yp.shape[0]

        def f(B):
            idx = torch.randint(0, n, (B,), device=self.device, generator=self.gen)
            return Yp[idx], Cp[idx]
        return f

    # ---- one supervised regression fit on pairs generated on the fly from a target bank ---- #
    def _fit(self, target_sampler, steps: int, lr0: float, lr1: float):
        cfg, S = self.cfg, self.S
        net = self.net
        opt = torch.optim.AdamW(net.parameters(), lr=lr0, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr1)
        ema = self.ema_net
        t0 = time.time()
        run = 0.0
        for it in range(steps):
            Y, C = target_sampler(cfg.batch)                                 # clean targets (B,p), (B,d,d)
            Ce = wishart_channel(C, cfg.N, self.gen)
            Yp = S.encode(Ce, cfg.eps0)                                      # Y'_e
            t = torch.rand(cfg.batch, 1, dtype=DT, device=self.device, generator=self.gen)
            z = torch.randn(cfg.batch, S.p, dtype=DT, device=self.device, generator=self.gen)
            W = torch.sqrt(t) * z
            I = (1 - t) * Yp + t * Y + self.sigma * (1 - t) * W
            R = Y - Yp - self.sigma * W
            pred = net(t.float(), I.float(), Yp.float())
            loss = ((pred - R.float()) ** 2).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(cfg.ema).add_(pn.detach(), alpha=1 - cfg.ema)
            run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
        return run, time.time() - t0

    # ---- Follmer SDE sampler (eq. 7) ---- #
    @torch.no_grad()
    def sample_posterior(self, Ce: torch.Tensor, n_draws: int = 1, net: Drift | None = None,
                         n_steps: int | None = None, chunk: int = 20000, gen=None) -> torch.Tensor:
        """Ce: (M,d,d) observations -> C draws (M,n_draws,d,d).

        Ce is moved to self.device automatically; a `gen` passed in must already live there."""
        cfg, S = self.cfg, self.S
        net = net or self.ema_net
        gen = gen or self.gen
        Ce = Ce.to(self.device)
        J = n_steps or cfg.n_sde_steps
        Ye = S.encode(Ce, cfg.eps0)                                          # (M,p)
        M = Ye.shape[0]
        Ye_rep = Ye.repeat_interleave(n_draws, dim=0)                        # (M*n,p)
        out = []
        for s in range(0, Ye_rep.shape[0], chunk):
            ye = Ye_rep[s:s + chunk]
            y = ye.clone()
            dt = 1.0 / J
            for j in range(J):
                t = j * dt
                tt = torch.full((y.shape[0], 1), t, device=self.device)
                b = net(tt, y.float(), ye.float()).to(DT)
                drift = (1 + t) * b - y + ye
                g = self.sigma * math.sqrt(max(1 - t * t, 0.0))
                y = y + drift * dt + g * math.sqrt(dt) * torch.randn(y.shape, dtype=DT, device=self.device, generator=gen)
            out.append(y)
        Y1 = torch.cat(out)
        C = S.decode(Y1)
        return C.reshape(M, n_draws, cfg.d, cfg.d)

    # ---- Algorithm 1 ---- #
    def fit(self, Ce_obs: torch.Tensor, callback=None):
        """Ce_obs: (M,d,d) recorded empirical covariances; moved to self.device automatically."""
        cfg, S = self.cfg, self.S
        Ce_obs = Ce_obs.to(DT).to(self.device)
        M = Ce_obs.shape[0]
        Ye = S.encode(Ce_obs, cfg.eps0)
        self._fit_init_prior(Ye)
        # normalisation stats for the network (fixed for the whole run)
        mu = Ye.mean(0)
        sd = Ye.std(0).clamp_min(1e-3)
        self.net = Drift(S.p, mu, sd, cfg.hidden, cfg.depth).to(self.device)
        self.ema_net = Drift(S.p, mu, sd, cfg.hidden, cfg.depth).to(self.device)
        self.ema_net.load_state_dict(self.net.state_dict())
        for q in self.ema_net.parameters():
            q.requires_grad_(False)

        # step 1: initialise with the sampleable prior pi_0
        Y0 = self._sample_init_prior(cfg.pool_init)
        pool0 = (Y0, S.decode(Y0))
        loss, dt = self._fit(self._pool_sampler(pool0), cfg.steps_first, cfg.lr, cfg.lr_end)
        self._log(0, loss, dt, note="init prior")
        if callback:
            callback(self, 0)

        # steps 2-3: reconstruct at recorded observations, re-corrupt, refit
        for k in range(1, cfg.n_outer + 1):
            t0 = time.time()
            C_rec = self.sample_posterior(Ce_obs, cfg.n_recon)               # (M,R,d,d)
            C_bank = C_rec.reshape(-1, cfg.d, cfg.d)
            bank = (S.encode(C_bank), C_bank)                                # clean targets (Y, C)
            t_rec = time.time() - t0
            loss, dt = self._fit(self._pool_sampler(bank), cfg.steps_outer, cfg.lr * 0.5, cfg.lr_end)
            self._log(k, loss, dt + t_rec, note=f"bank={bank[0].shape[0]}")
            if callback:
                callback(self, k)
        return self

    def _log(self, k, loss, dt, note=""):
        self.hist.append(dict(k=k, loss=loss, sec=dt))
        if self.cfg.verbose:
            print(f"  [SCSI] outer {k}: loss={loss:.4f}  ({dt:.0f}s)  {note}", flush=True)

    # ---- supervised control: train on true (C, Ce) pairs, for diagnostics only ---- #
    def fit_supervised(self, C_true_sampler, Ce_ref: torch.Tensor, steps: int | None = None):
        cfg, S = self.cfg, self.S
        Ce_ref = Ce_ref.to(DT).to(self.device)
        Ye = S.encode(Ce_ref, cfg.eps0)
        mu, sd = Ye.mean(0), Ye.std(0).clamp_min(1e-3)
        self.net = Drift(S.p, mu, sd, cfg.hidden, cfg.depth).to(self.device)
        self.ema_net = Drift(S.p, mu, sd, cfg.hidden, cfg.depth).to(self.device)
        self.ema_net.load_state_dict(self.net.state_dict())
        for q in self.ema_net.parameters():
            q.requires_grad_(False)
        Cp = C_true_sampler(cfg.pool_init).to(self.device)
        loss, dt = self._fit(self._pool_sampler((S.encode(Cp), Cp)), steps or cfg.steps_first, cfg.lr, cfg.lr_end)
        self._log(0, loss, dt, note="supervised")
        return self
