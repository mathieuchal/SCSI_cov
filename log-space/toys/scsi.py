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
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from model import Drift, make_drift

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

    def __init__(self, d: int, device=None):
        #Data dimension
        self.d = d
        #
        self.p = d * (d + 1) // 2
        device = torch.device(device) if device is not None else torch.device("cpu")
        self.device = device
        iu = torch.triu_indices(d, d, device=device)
        self.i0, self.i1 = iu[0], iu[1]
        self.scale = torch.where(self.i0 == self.i1, 1.0, math.sqrt(2.0)).to(DT).to(device)

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
        """Matrix logarithm"""
        w, V = torch.linalg.eigh(C)
        return (V * torch.log(w.clamp_min(1e-300)).unsqueeze(-2)) @ V.transpose(-1, -2)

    @staticmethod
    def expm(S: torch.Tensor) -> torch.Tensor:
        """Matrix exponential"""
        w, V = torch.linalg.eigh(S)
        return (V * torch.exp(w).unsqueeze(-2)) @ V.transpose(-1, -2)

    def encode(self, C: torch.Tensor, eps0: float = 0.0) -> torch.Tensor:
        """Phi(C) = svec(log(C + eps0 I)); eps0 > 0 only needed when N < d."""
        C = C.to(DT)
        if eps0 > 0:
            C = C + eps0 * torch.eye(self.d, dtype=DT, device=C.device)
        return self.svec(self.logm(C))

    def decode(self, Y: torch.Tensor) -> torch.Tensor:
        return self.expm(self.svec_inv(Y.to(DT)))


def _safe_cholesky(C: torch.Tensor) -> torch.Tensor:
    """Cholesky with a numerical safety net: C is mathematically guaranteed PD here (it always comes from
    Sym.expm, a sum of positive eigenvalue outer products), but the SDE sampler's unclamped Brownian steps
    can occasionally push a reconstructed coordinate to an extreme value; expm's eigh-based reconstruction
    then suffers floating-point cancellation across widely-spread eigenvalues, and the result can come out
    numerically (not mathematically) non-PD. Symmetrizing first (cheap) fixes the common roundoff case;
    the eigh-based floor-and-rebuild fallback (expensive, only hit on actual Cholesky failure) handles the
    rare case where that isn't enough -- it doesn't change the model, it just repairs a construction that
    was always supposed to be exactly PD."""
    C = 0.5 * (C + C.transpose(-1, -2))
    try:
        return torch.linalg.cholesky(C)
    except torch._C._LinAlgError:
        w, V = torch.linalg.eigh(C)
        w = w.clamp_min(1e-9 * w.abs().amax(-1, keepdim=True).clamp_min(1e-9))
        C = (V * w.unsqueeze(-2)) @ V.transpose(-1, -2)
        C = 0.5 * (C + C.transpose(-1, -2))
        return torch.linalg.cholesky(C)


def randn_on(shape, gen: torch.Generator | None, device, dtype=DT) -> torch.Tensor:
    """Gaussian draw on `device`. If `gen` lives on that same device type the draw happens there directly (no host
    RNG, no host->device copy -- the hot path inside SCSI's loops, which use the device generator self.dgen);
    otherwise (a CPU generator with a CUDA target, e.g. toy_train's data stream) draw on the CPU and move."""
    device = torch.device(device)
    if gen is None or gen.device.type == device.type:
        return torch.randn(*shape, dtype=dtype, generator=gen, device=device)
    return torch.randn(*shape, dtype=dtype, generator=gen).to(device)


def wishart_channel(C: torch.Tensor, N: int, gen: torch.Generator | None = None) -> torch.Tensor:
    """Ce = (1/N) L Z Z^T L^T with C = L L^T  ~  K_N(. | C)."""
    d = C.shape[-1]
    C = C.to(DT)
    L = _safe_cholesky(C)
    Z = randn_on((*C.shape[:-2], d, N), gen, C.device)
    X = L @ Z
    return X @ X.transpose(-1, -2) / N

def empirical_cov(X: torch.Tensor) -> torch.Tensor:
    """X: (..., d, N) raw samples -> (1/N) X X^T (zero-mean convention of eq. 1)."""
    X = X.to(DT)
    return X @ X.transpose(-1, -2) / X.shape[-1]



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
    activation: str = "silu"    # hidden-layer nonlinearity of the 'mlp' Drift (silu | relu | gelu | tanh)
    arch: str = "mlp"           # 'mlp' (Drift, default) | 'transformer' (TransformerDrift) | 'structured' (StructuredFactorDrift)
    nhead: int = 8               # attention heads, only used by arch in {'transformer', 'structured'}
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
    lr_outer_floor_ratio: float = 0.02  # fresh run: k=n_outer's lr ceiling, as a fraction of k=1's (lr*0.5)
    resume_lr: float = 5e-5     # resumed run: constant lr ceiling over the remaining outer iterations
    verbose: bool = True
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class SCSI:
    """Learn P(C | Ce) from an ensemble of empirical covariances only."""

    def __init__(self, cfg: SCSIConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.S = Sym(cfg.d, device=self.device)
        self.sigma = cfg.kappa * math.sqrt(2.0 / cfg.N)
        self.gen = torch.Generator().manual_seed(cfg.seed)  # CPU generator: one-off draws (init prior, its pool)
        # device generator for the hot loops (channel noise, t/z in _fit, bank indices, SDE noise): drawing these on
        # the CPU and copying them to the GPU at every step was ~75% of the wall time
        self.dgen = torch.Generator(device=self.device).manual_seed(cfg.seed + 1)
        torch.manual_seed(cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)
        torch.set_num_threads(cfg.threads)
        self.net: Drift | None = None
        self.ema_net: Drift | None = None
        self.hist: list[dict] = []
        # outer lr-ceiling schedule of this fit() call, set in fit() -- see _outer_lr0; saved in the checkpoints.
        self.lr_outer: dict | None = None

    def _outer_lr0(self, k: int) -> float:
        """Ceiling lr of outer iteration k's own (inner) CosineAnnealingLR refit, from self.lr_outer:
          resumed run: {"schedule": "const", "k_start": start_k+1, "k_end": n_outer, "lr": resume_lr} (warm restart);
          fresh run:   {"schedule": "cosine", "k_start": 1, "k_end": n_outer, "lr_max": lr*0.5,
                        "lr_min": lr*0.5*lr_outer_floor_ratio}, a cosine from lr_max (k=k_start) to lr_min (k=k_end)."""
        o = self.lr_outer
        if o["schedule"] == "const":
            return o["lr"]
        start_k, end_k, lr_max, lr_min = o["k_start"], o["k_end"], o["lr_max"], o["lr_min"]
        if end_k <= start_k:
            return lr_min
        frac = min(max((k - start_k) / (end_k - start_k), 0.0), 1.0)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * frac))

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
        cov = cov * cfg.log_prior_inflate + 1e-6 * torch.eye(Ye.shape[1], dtype=DT, device=Ye.device) * cov.diag().mean()
        self._init_m, self._init_L = m, torch.linalg.cholesky(cov, upper=True).transpose(-1, -2)

    def _sample_init_prior(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.S.p, dtype=DT, generator=self.gen).to(self.device)
        return self._init_m + z @ self._init_L.T

    def _pool_sampler(self, pool):
        Yp, Cp = pool
        n = Yp.shape[0]

        def f(B):
            idx = torch.randint(0, n, (B,), generator=self.dgen, device=self.device).to(Yp.device)
            return Yp[idx], Cp[idx]
        return f

    # ---- one supervised regression fit on pairs generated on the fly from a target bank ---- #
    def _fit(self, target_sampler, steps: int, lr0: float, lr1: float):
        cfg, S = self.cfg, self.S
        net = self.net
        net.train()                                                          # training mode (dropout on, e.g. TransformerDrift)
        opt = torch.optim.AdamW(net.parameters(), lr=lr0, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr1)
        ema = self.ema_net
        ema.eval()                                                           # EMA copy is only ever used for inference
        t0 = time.time()
        run = 0.0
        n_skip = 0
        for it in range(steps):
            Y, C = target_sampler(cfg.batch)                                 # clean targets (B,p), (B,d,d)
            Ce = wishart_channel(C, cfg.N, self.dgen)
            Yp = S.encode(Ce, cfg.eps0)                                      # Y'_e
            t = torch.rand(cfg.batch, 1, dtype=DT, generator=self.dgen, device=self.device)
            z = torch.randn(cfg.batch, S.p, dtype=DT, generator=self.dgen, device=self.device)
            W = torch.sqrt(t) * z
            I = (1 - t) * Yp + t * Y + self.sigma * (1 - t) * W
            R = Y - Yp - self.sigma * W
            pred = net(t.float(), I.float(), Yp.float())
            loss = ((pred - R.float()) ** 2).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            if not (torch.isfinite(loss) and torch.isfinite(gnorm)):     # rare non-finite batch: skip it instead of
                n_skip += 1                                              # letting one NaN poison the weights for good
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
            opt.step()
            sched.step()
            #Update EMA net
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(cfg.ema).add_(pn.detach(), alpha=1 - cfg.ema)
            run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
        if n_skip:
            print(f"  [SCSI] warning: skipped {n_skip}/{steps} non-finite steps", flush=True)
        return run, time.time() - t0

    # ---- Follmer SDE sampler (eq. 7) ---- #
    @torch.no_grad()
    def sample_posterior(self, Ce: torch.Tensor, n_draws: int = 1, net: Drift | None = None,
                         n_steps: int | None = None, chunk: int = 20000, gen=None) -> torch.Tensor:
        """Ce: (M,d,d) observations -> C draws (M,n_draws,d,d)."""
        cfg, S = self.cfg, self.S
        net = net or self.ema_net
        net.eval()                                                           # inference: no dropout in the SDE drift
        gen = gen or self.dgen
        J = n_steps or cfg.n_sde_steps
        Ce = Ce.to(self.device)
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
                y = y + drift * dt + g * math.sqrt(dt) * randn_on(y.shape, gen, self.device)
            out.append(y)
        Y1 = torch.cat(out)
        C = S.decode(Y1)
        return C.reshape(M, n_draws, cfg.d, cfg.d).cpu()

    # ---- Algorithm 1 ---- #
    def fit(self, Ce_obs, callback=None, infinite_batch: int | None = None, resume: dict | None = None):
        """Ce_obs: either
             - a (M,d,d) tensor of recorded empirical covariances (finite-data mode, as before): the SAME
               ensemble is reconditioned on at every outer iteration, or
             - a callable B -> (B,d,d) tensor (infinite-data mode): a brand-new (C, Ce) ensemble of size
               `infinite_batch` is drawn fresh at the E-step of *every* outer iteration (including the very
               first, used to fit the initial prior), instead of reusing one fixed observed set.
           `infinite_batch` is required (and only used) when Ce_obs is a callable.

           resume: optional dict to continue a previous run instead of starting from the sampleable-prior
           init -- {"net_state", "ema_state", "mu", "sd", "gen_state" (optional, this SCSI's own self.gen,
           CPU ByteTensor), "dgen_state" (optional), "start_k"} as saved by toy_train.py's checkpoint callback. The outer lr ceiling
           is then held at cfg.resume_lr. Skips _fit_init_prior
           and the step-1 fit entirely; the outer loop resumes at start_k+1. Network architecture (hidden/
           depth) must already match cfg (the caller is responsible for constructing SCSI(cfg) accordingly)."""
        cfg, S = self.cfg, self.S
        infinite = callable(Ce_obs)
        if resume is not None:
            if not infinite:
                Ce_cur = Ce_obs.to(DT).to(self.device)                       # finite-data mode: still need this for the loop below
            #Normalizations statistics
            mu, sd = resume["mu"].to(self.device), resume["sd"].to(self.device)
            #Load module
            self.net = make_drift(cfg.arch, S.p, mu, sd, hidden=cfg.hidden, depth=cfg.depth, d=S.d, nhead=cfg.nhead, N=cfg.N, activation=cfg.activation).to(self.device)
            self.net.load_state_dict(resume["net_state"])
            self.ema_net = make_drift(cfg.arch, S.p, mu, sd, hidden=cfg.hidden, depth=cfg.depth, d=S.d, nhead=cfg.nhead, N=cfg.N, activation=cfg.activation).to(self.device)
            self.ema_net.load_state_dict(resume["ema_state"])
            self.ema_net.eval()
            #Stop-grad operator
            for q in self.ema_net.parameters():
                q.requires_grad_(False)
            if resume.get("gen_state") is not None:
                self.gen.set_state(resume["gen_state"])                      # CPU ByteTensor; self.gen is always CPU
            if resume.get("dgen_state") is not None:
                self.dgen.set_state(resume["dgen_state"].cpu())             # device generator (absent in pre-fix checkpoints)
            start_k = resume["start_k"]
            self.lr_outer = {"schedule": "const", "k_start": start_k + 1, "k_end": cfg.n_outer, "lr": cfg.resume_lr}
        else:
            self.lr_outer = {"schedule": "cosine", "k_start": 1, "k_end": cfg.n_outer,
                             "lr_max": cfg.lr * 0.5, "lr_min": cfg.lr * 0.5 * cfg.lr_outer_floor_ratio}
            if infinite:
                assert infinite_batch is not None, "infinite_batch must be given when Ce_obs is a sampler (infinite-data mode)"
                Ce_cur = Ce_obs(infinite_batch).to(DT).to(self.device)
            else:
                Ce_cur = Ce_obs.to(DT).to(self.device)
            #Encode observation matrices
            Ye = S.encode(Ce_cur, cfg.eps0)
            #Fit an initial prior
            self._fit_init_prior(Ye)
            # normalisation stats for the network (fixed for the whole run)
            mu = Ye.mean(0)
            sd = Ye.std(0).clamp_min(1e-3)
            #Define neural network
            self.net = make_drift(cfg.arch, S.p, mu, sd, hidden=cfg.hidden, depth=cfg.depth, d=S.d, nhead=cfg.nhead, N=cfg.N, activation=cfg.activation).to(self.device)
            self.ema_net = make_drift(cfg.arch, S.p, mu, sd, hidden=cfg.hidden, depth=cfg.depth, d=S.d, nhead=cfg.nhead, N=cfg.N, activation=cfg.activation).to(self.device)
            self.ema_net.load_state_dict(self.net.state_dict())
            self.ema_net.eval()
            for q in self.ema_net.parameters():
                q.requires_grad_(False)

            # step 1: initialise with the sampleable prior pi_0
            Y0 = self._sample_init_prior(cfg.pool_init)
            pool0 = (Y0, S.decode(Y0))
            loss, dt = self._fit(self._pool_sampler(pool0), cfg.steps_first, cfg.lr, cfg.lr_end)
            self._log(0, loss, dt, note="init prior")
            if callback:
                callback(self, 0)
            start_k = 0

        # steps 2-3: reconstruct at recorded observations, re-corrupt, refit
        for k in range(start_k + 1, cfg.n_outer + 1):
            t0 = time.time()
            
            #Renew the observed matrices
            if infinite:
                Ce_cur = Ce_obs(infinite_batch).to(DT).to(self.device)       # brand-new (C, Ce) ensemble this E-step
            #Sample clean posterior matrices
            C_rec = self.sample_posterior(Ce_cur, cfg.n_recon).to(self.device)   # (M,R,d,d)
            C_bank = C_rec.reshape(-1, cfg.d, cfg.d)
            bank = (S.encode(C_bank), C_bank)                                # clean targets (Y, C)
            t_rec = time.time() - t0
            lr0_k = self._outer_lr0(k)
            lr1_k = lr0_k * (cfg.lr_end / cfg.lr)          # keep the same inner ceiling:floor ratio as cfg.lr/cfg.lr_end
            loss, dt = self._fit(self._pool_sampler(bank), cfg.steps_outer, lr0_k, lr1_k)

            #Log data
            self._log(k, loss, dt + t_rec, note=f"bank={bank[0].shape[0]} lr0={lr0_k:.1e}" + (" [fresh]" if infinite else ""))
            if callback:
                callback(self, k)
        return self

    def _log(self, k, loss, dt, note=""):
        """Log training metrics"""
        self.hist.append(dict(k=k, loss=loss, sec=dt))
        if self.cfg.verbose:
            print(f"  [SCSI] outer {k}: loss={loss:.4f}  ({dt:.0f}s)  {note}", flush=True)

    # ---- supervised control: train on true (C, Ce) pairs, for diagnostics only ---- #
    def fit_supervised(self, C_true_sampler, Ce_ref: torch.Tensor, steps: int | None = None):
        cfg, S = self.cfg, self.S
        Ye = S.encode(Ce_ref.to(DT).to(self.device), cfg.eps0)
        mu, sd = Ye.mean(0), Ye.std(0).clamp_min(1e-3)
        self.net = make_drift(cfg.arch, S.p, mu, sd, hidden=cfg.hidden, depth=cfg.depth, d=S.d, nhead=cfg.nhead, N=cfg.N, activation=cfg.activation).to(self.device)
        self.ema_net = make_drift(cfg.arch, S.p, mu, sd, hidden=cfg.hidden, depth=cfg.depth, d=S.d, nhead=cfg.nhead, N=cfg.N, activation=cfg.activation).to(self.device)
        self.ema_net.load_state_dict(self.net.state_dict())
        self.ema_net.eval()
        for q in self.ema_net.parameters():
            q.requires_grad_(False)
        Cp = C_true_sampler(cfg.pool_init).to(self.device)
        loss, dt = self._fit(self._pool_sampler((S.encode(Cp), Cp)), steps or cfg.steps_first, cfg.lr, cfg.lr_end)
        self._log(0, loss, dt, note="supervised")
        return self
