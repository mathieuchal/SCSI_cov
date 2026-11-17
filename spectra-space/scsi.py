"""Self-consistent stochastic interpolants for covariance posterior learning -- factor-space variant.

Same self-consistent EM scheme as toys/scsi.py (Algorithm 1 of the paper: reconstruct -> re-corrupt ->
refit), but the network learns directly in the space of the Cholesky factor Sigma (C = Sigma Sigma^T)
instead of the matrix-logarithm coordinates svec(log C) -- the parametrization used by the notebook
factor_space_nonri_self_consistent_em.ipynb. Three things change relative to toys/scsi.py:

  1. encoding: Phi(C) = flatten(cholesky(C))   instead of   Phi(C) = svec(logm(C))       (class `Chol` vs `Sym`)
  2. sampler:  a deterministic ODE (flow matching, Euler/Ralston)   instead of   the stochastic Follmer SDE
  3. the Cholesky decomposition (an actual matrix decomposition: a linear solve, can be ill-conditioned) is
     called AT MOST ONCE in the whole training workflow -- right where a fresh prior sample C is first turned
     into a factor (toy_train.py's Ce_tr closure, or here in fit_supervised). Every downstream step stays in
     factor/matrix space and needs no further decomposition:
       - Sigma -> Ce: `sample_covariance_from_factors` (Sigma @ Z, not cholesky(C) @ Z -- Sigma is already known)
       - Ce's own representation (the network's conditioning input Ye): `Chol.flatten(Ce)`, i.e. Ce's own
         lower-triangular entries taken directly -- Ce is *never* decomposed, it stays a plain matrix throughout.
       - Sigma -> C (only where C is genuinely needed downstream, e.g. for stats_of()): `Chol.decode`, i.e.
         `Sigma @ Sigma.T` -- a plain matrix product, not a decomposition, so this one is free to call as often
         as needed. In particular sample_posterior's own output stays in Sigma form (no decode call at all): the
         self-consistent loop's "reconstruct" step reuses the ODE's own factor output directly as next round's
         target, with *zero* additional Cholesky calls.

Point (2) is not just "drop the noise term" from the SDE: the Follmer SDE's *initial condition* is the
conditioning point itself (I(0) = Y'_e), so the accumulated Brownian motion during integration is the SDE's
only source of posterior diversity for a fixed Ce. A deterministic ODE has no such term, so integrating it
from I(0) = Y'_e would collapse every draw for a given Ce onto the *same* point -- the model could not
represent posterior uncertainty at all. Instead (matching the notebook): the interpolant runs between an
independent reference draw `base ~ N(0, ref_sd^2 I)` (unconditional, fixed, replayable at sampling time) and
the clean target Y, with the conditioning Ye passed to the network as a *separate* input (exactly the role it
already plays via `Drift.forward(t, I, Ye)`, which is why that class barely needs to change). The randomness
behind posterior diversity is now the random choice of `base`, not accumulated SDE noise.

  conditional ODE interpolant   I_t = (1-t) base + t Y                    base ~ N(0, ref_sd^2 I), Y indep.
  straight-line velocity        R = Y - base                              (constant in t; dI/dt = R exactly)
  sampler (Euler / Ralston)     dY = net(t, Y, Ye) dt,  Y(0) = base,  Ye = Chol.flatten(Ce) (Ce, not Sigma)

Everything else -- the self-consistent EM outer loop, the data-informed sampleable initial prior pi_0, the
EMA network, checkpoint/resume contract -- is unchanged from toys/scsi.py; see that file for the reference
narrative and Algorithm 1. NOTE: `_fit_init_prior` still calls Chol.encode/decode internally and has not yet
been revisited for the one-Cholesky-only design (left as is for now, by explicit request).

  4. output calibration: `Drift` holds THREE normalisation pairs (mu_y/sd_y, mu_ce/sd_ce, velocity_loc/scale),
     not toys/scsi.py's one (mu/sd) -- matching the notebook's ConditionalFactorDrift, which calibrates
     state_loc/scale, cond_loc/scale AND velocity_loc/scale from a real batch, additively AND multiplicatively,
     per coordinate. A single global out_scale (as an earlier version of this file used) under-specifies R=Y-base:
     base is centred but Y generally isn't, so R has a nonzero per-coordinate mean the network would otherwise
     have to learn from scratch instead of starting from the right place (zero-init last layer => prediction
     = velocity_loc exactly at k=0). cfg.refresh_every controls how often all three pairs get recalibrated
     (0 = only at construction, matching Y0/Y_p's current distribution) -- see Drift.refresh_norm.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from tqdm import tqdm

from model import ResNetDrift, make_drift
from utils_spectra import orientation_mcmc

torch.set_default_dtype(torch.float32)
DT = torch.float64  # dtype for matrix functions / simulation


# --------------------------------------------------------------------------- #
# matrix utilities
# --------------------------------------------------------------------------- #
class Chol:
    """Cholesky-factor flatten/unflatten helpers for a fixed dimension d.

    Plays the same structural role as toys/scsi.py's `Sym` (svec/logm), but encodes a SPD matrix via its
    (lower-triangular, unconstrained-sign-diagonal) factor Sigma instead of its matrix logarithm -- the
    representation the factor-space notebook learns in directly.

    decode() is many-to-one on Sigma: for lower-triangular Sigma, the orthogonal transforms Q preserving both
    C = Sigma Sigma^T and lower-triangularity are exactly the diagonal sign flips Q = diag(+-1,...,+-1), i.e.
    Sigma -> Sigma Q flips entire COLUMNS (not rows) -- (Sigma Q)_ij = Sigma_ij * Q_jj, which flips the sign of
    diagonal entry (j,j) along with every entry below it in that same column, leaving C unchanged. encode()
    always returns the canonical (positive-diagonal) torch.linalg.cholesky factor; canonicalize() applies the
    same column-sign-flip to an ARBITRARY lower-triangular Sigma (e.g. the network's own ODE output) without
    any decomposition, for the same reason -- see canonicalize()'s docstring."""

    def __init__(self, d: int, device=None):
        self.d = d
        self.p = d * (d + 1) // 2
        device = torch.device(device) if device is not None else torch.device("cpu")
        self.device = device
        il = torch.tril_indices(d, d, device=device)
        self.i0, self.i1 = il[0], il[1]
        self.SVEC_WEIGHTS=torch.where(self.i0==self.i1, 1, math.sqrt(2.0)).to(device).to(DT)

    def canonicalize(self, Sigma: torch.Tensor) -> torch.Tensor:
        diag = torch.diagonal(Sigma, dim1=-2, dim2=-1)
        sign = torch.where(diag < 0, -1.0, 1.0).to(Sigma.dtype)
        return Sigma * sign.unsqueeze(-2)

    def flatten(self, Sigma: torch.Tensor, sym: bool=False) -> torch.Tensor:
        const = self.SVEC_WEIGHTS if sym else 1
        return Sigma[..., self.i0, self.i1] * const

    def unflatten(self, y: torch.Tensor, sym: bool=False) -> torch.Tensor:
        const=self.SVEC_WEIGHTS if sym else 1
        Sigma = torch.zeros(*y.shape[:-1], self.d, self.d, dtype=y.dtype, device=y.device)
        Sigma[..., self.i0, self.i1] = y/const
        return Sigma

    def encode(self, C: torch.Tensor, eps0: float = 0.0) -> torch.Tensor:
        """Phi(C) = flatten(cholesky(C + eps0 I)); eps0 > 0 only needed when N < d."""
        C = C.to(DT)
        if eps0 > 0:
            C = C + eps0 * torch.eye(self.d, dtype=DT, device=C.device)
        return torch.linalg.cholesky(C)

    def decode(self, Y: torch.Tensor) -> torch.Tensor:
        Sigma = self.unflatten(Y.to(DT))
        return Sigma @ Sigma.transpose(-1, -2)

    @property
    def sigma_shape(self):
        """Shape of one `Chol`-space "factor" sample (as returned by sample_posterior before the leading
        (M, n_draws) dims) -- (d, d) here, (d,) for Spec. Lets SCSI.fit/sample_posterior reshape generically
        without hardcoding which encoding is in use."""
        return (self.d, self.d)

    def sample_Ce(self, Sigma: torch.Tensor, N: int, gen: torch.Generator | None = None) -> torch.Tensor:
        """Degradation operator dispatch: Sigma (a Cholesky factor) -> noisy Ce. See sample_Ce on `Spec`
        for the rotationally-invariant analogue (eigenvalues -> noisy Ce's eigenvalues, no (d,d) factor)."""
        return sample_covariance_from_factors(Sigma, N, gen)


class Spec:
    """Spectral (eigenvalue-only) encoding for a fixed dimension d -- rotation-invariant-prior (RI)
    analogue of `Chol`, used by SCSIConfig(encoding="spectral"). Valid ONLY when the prior law of C is
    rotationally invariant (law(C) == law(R C R^T) for every orthogonal R): under that assumption the
    Bayes-optimal (Frobenius) posterior-mean estimator of C given Ce always has the form
    V_Ce @ diag(f(eigenvalues of Ce)) @ V_Ce^T -- it keeps Ce's OWN eigenvectors and only reshapes its
    eigenvalues via some (possibly data-dependent) function of Ce's full spectrum (classical RIE result,
    Bouchaud & Potters). So instead of learning the self-consistent posterior over the full
    p=d(d+1)/2-dim chart of C (what `Chol` does), it suffices to learn the conditional law of C's
    eigenvalues given Ce's eigenvalues -- a d-dim problem regardless of how large d(d+1)/2 gets. If the
    prior is NOT rotationally invariant, use `Chol` instead -- see spectra-space/covsi_pd_em_spectral.py
    for the original (non-self-consistent-flow-matching) version of this same idea.

    Plays the same structural role as `Chol` (encode/flatten/unflatten/canonicalize/sample_Ce), with ONE
    necessary interface difference: decode (vector -> matrix) needs an explicit reference eigenbasis
    V_ref (always Ce's own, by the RIE result above) since a bare eigenvalue vector alone doesn't
    determine a matrix -- Chol's Cholesky factor never has this issue, so this is NOT called from
    SCSI.fit's generic inner loop at all (which never needs an actual (d,d) matrix mid-training -- see
    SCSI.sample_posterior_sigma for where an actual matrix IS reconstructed, at scoring/evaluation time)."""

    def __init__(self, d: int, device=None):
        self.d = d
        self.p = d                              # NOT d(d+1)/2 -- this is the whole point
        device = torch.device(device) if device is not None else torch.device("cpu")
        self.device = device

    @property
    def sigma_shape(self):
        return (self.d,)

    def canonicalize(self, lam: torch.Tensor) -> torch.Tensor:
        """No sign/permutation ambiguity for real eigenvalues (unlike Chol's Cholesky-sign ambiguity) --
        kept only for interface parity with Chol so SCSI.fit's loop can call S.canonicalize(...)
        uniformly regardless of parametrization. Clamps away from 0 for numerical safety only."""
        return lam.clamp_min(1e-10)

    def flatten(self, M_or_lam: torch.Tensor, sym: bool = False) -> torch.Tensor:
        """Two call conventions, both used by SCSI.fit's generic inner loop:
          - (..., d, d) matrix (e.g. a real Ce, never pre-decomposed by the caller): eigvalsh it (one
            cheap d-dim decomposition EVERY call -- unlike Chol.flatten, which assumes an
            ALREADY-decomposed factor and never decomposes) and return its ascending log-eigenvalues.
          - (..., d) vector already holding real eigenvalues (e.g. this class's own canonicalize/
            unflatten output): just take its log. `sym` is accepted for call-site parity with
            Chol.flatten (which uses it for svec weighting) but unused here -- there is no analogous
            weighting for a plain eigenvalue vector."""
        if M_or_lam.dim() >= 2 and M_or_lam.shape[-1] == self.d and M_or_lam.shape[-2] == self.d:
            lam = torch.linalg.eigvalsh(M_or_lam.to(DT))
        else:
            lam = M_or_lam.to(DT)
        return torch.log(lam.clamp_min(1e-10))

    def unflatten(self, Y: torch.Tensor) -> torch.Tensor:
        """Y (log-eigenvalues) -> lam (real eigenvalues, ascending)."""
        return torch.exp(Y.to(DT))

    def encode(self, C: torch.Tensor, eps0: float = 0.0) -> torch.Tensor:
        """Phi(C) = eigvalsh(C + eps0 I), ascending, REAL (not log) -- the one allowed decomposition of a
        fresh prior sample C, mirroring Chol.encode. (Chol.encode returns a Cholesky FACTOR; here the
        "factor" is simply the eigenvalue vector itself, since no eigenvector information is ever
        modeled or learned.)"""
        C = C.to(DT)
        if eps0 > 0:
            C = C + eps0 * torch.eye(self.d, dtype=DT, device=C.device)
        return torch.linalg.eigvalsh(C)

    def decode(self, Y: torch.Tensor, V_ref: torch.Tensor) -> torch.Tensor:
        """(..., d) log-eigenvalues + (..., d, d) orthonormal reference eigenvectors (in the SAME
        ascending order as Y -- in practice always Ce's own torch.linalg.eigh output) -> (..., d, d)
        SPD. Unlike Chol.decode, a bare eigenvalue vector alone cannot determine a matrix, so V_ref is a
        required second argument -- see this class's docstring for why using Ce's own V_ref is valid."""
        lam = torch.exp(Y.to(DT))
        V_ref = V_ref.to(DT)
        return (V_ref * lam.unsqueeze(-2)).matmul(V_ref.transpose(-1, -2))

    def sample_Ce(self, lam: torch.Tensor, N: int, gen: torch.Generator | None = None) -> torch.Tensor:
        """Degradation operator dispatch: lam (eigenvalues) -> noisy Ce's OWN eigenvalues (not a (d,d)
        matrix -- see sample_eigs_from_lambda)."""
        return sample_eigs_from_lambda(lam, N, gen)


def sample_covariance_from_factors(Sigma: torch.Tensor, N: int, gen: torch.Generator | None = None) -> torch.Tensor:
    """Ce = (1/N) Sigma Z Z^T Sigma^T, given an ALREADY-COMPUTED factor Sigma (C = Sigma Sigma^T) -- the torch
    equivalent of the notebook's sample_covariance_from_factors. Same math as wishart_channel, but skips its
    internal Cholesky call since Sigma is already available: this is what lets Cholesky happen only once in
    the whole training workflow (at the point a prior sample C is first turned into a factor, via Chol.encode),
    never again downstream -- every subsequent noisy draw goes Sigma -> Ce directly through this function."""
    d = Sigma.shape[-1]
    Sigma = Sigma.to(DT)
    Z = torch.randn(*Sigma.shape[:-2], d, N, dtype=DT, generator=gen).to(Sigma.device)
    X = Sigma @ Z
    return X @ X.transpose(-1, -2) / N


def sample_eigs_from_lambda(lam: torch.Tensor, N: int, gen: torch.Generator | None = None) -> torch.Tensor:
    """lam: (..., d) REAL eigenvalues of a (notional) diagonal C -> (..., d) ascending eigenvalues of a
    fresh noisy Ce = (1/N) X X^T, X ~ iid N(0, diag(lam)) columns. `Spec`'s analogue of
    sample_covariance_from_factors -- valid as a stand-in for the TRUE (possibly non-diagonal,
    same-eigenvalue) channel's eigenvalue-marginal by rotational invariance of the Gaussian/Wishart
    sampling model: Ce's eigenvalue distribution given C depends only on C's eigenvalues, never on its
    eigenvectors, so simulating as if C were exactly diag(lam) gives the exactly right law for Ce's
    spectrum. This is what lets SCSI.fit's self-consistent loop never materialize a (d,d) matrix for
    the spectral encoding, except for Ce_cur (the genuinely-observed/simulated data)."""
    d = lam.shape[-1]
    lam = lam.to(DT).clamp_min(1e-10)
    Z = torch.randn(*lam.shape[:-1], N, d, dtype=DT, generator=gen).to(lam.device) * lam.sqrt().unsqueeze(-2)
    Ce = torch.einsum("...nj,...ni->...ij", Z, Z) / N
    return torch.linalg.eigvalsh(Ce)


def empirical_cov(X: torch.Tensor) -> torch.Tensor:
    """X: (..., d, N) raw samples -> (1/N) X X^T (zero-mean convention)."""
    X = X.to(DT)
    return X @ X.transpose(-1, -2) / X.shape[-1]


def follmer_interpolate(Ye: torch.Tensor, Y: torch.Tensor, t: torch.Tensor, epsilon: float):
    """Verbatim from spectra-space/covsi_pd_em.py (shape-generic, works for any (...,p)): a Follmer
    BRIDGE between Ye (conditioning anchor, at t=0) and Y (clean target, at t=1), with Brownian noise
    epsilon*(1-t)*sqrt(t)*Z added to the mean path. `target` is the drift a reverse-time SDE needs to
    track this specific noisy bridge (not just Y-Ye, the noise-free straight-line case) -- see
    SCSI._integrate_follmer's sampling-time counterpart below."""
    Z = torch.randn_like(Y)
    sqrt_t = t.sqrt()
    state = (1 - t) * Ye + t * Y + epsilon * (1.0 - t) * sqrt_t * Z
    target = Y - Ye - epsilon * sqrt_t * Z
    return state, target


# --------------------------------------------------------------------------- #
# main learner
# --------------------------------------------------------------------------- #
@dataclass
class SCSIConfig:
    N: int
    d: int
    hidden: int = 256
    depth: int = 4
    arch: str = "mlp"           # "mlp" (ResBlock MLP) | "transformer" (row-tokenized self-attention)
    nhead: int = 8               # transformer only
    encoding: str = "chol"       # "chol" (Chol: d(d+1)/2-dim Cholesky-factor chart, any prior) |
                                  # "spectral" (Spec: d-dim log-eigenvalue chart, RI priors only)
    n_outer: int = 6            # self-consistent iterations (K)
    steps_first: int = 4000     # gradient steps for the initial fit
    steps_outer: int = 2000     # gradient steps per subsequent refit
    batch: int = 1024
    lr: float = 1e-3
    lr_end: float = 1e-4
    n_recon: int = 8            # reconstructions per recorded observation per outer iteration
    n_ode_steps: int = 50       # Euler/Ralston steps for sample_posterior (replaces toys/scsi.py's n_sde_steps)
    ode_solver: str = "euler"   # "euler" | "ralston" (2-stage, 2nd order)
    ref_sd: float | None = None  # per-coordinate std of the base/reference draw in Y-space; None -> 1.10/sqrt(d)
                                  # (matches the notebook's REFERENCE_FACTOR_SD convention)
    sampler: str = "ode"        # "ode" (deterministic flow-matching ODE, default -- straight-line
                                  # interpolant from an INDEPENDENT base) | "follmer" (stochastic Follmer
                                  # bridge SDE, ANCHORED at Ye, real noise injected during sampling --
                                  # matches spectra-space/covsi_pd_em_spectral.py's integrate_sde/
                                  # follmer_interpolate. Experimental: added to test whether the "ode"
                                  # sampler's lack of injected noise is why the spectral encoding's
                                  # generated eigenvalue spread collapses toward the conditional mean
                                  # (confirmed empirically: ruling out network capacity, training
                                  # duration, and ODE step count as explanations).
    follmer_epsilon: float = 1.0   # "follmer" only: bridge noise scale (same name as the reference script)
    follmer_T: int = 50            # "follmer" only: number of SDE steps (same role as n_ode_steps)
    eps0: float = 0.0           # >0 iff N < d
    seed: int = 0
    weight_decay: float = 0.0
    ema: float = 0.999
    log_prior_inflate: float = 1.0  # scale on covariance of the Gaussian initial prior (in Y-space)
    pool_init: int = 200_000    # size of the pool drawn from the initial prior / supervised prior
    threads: int = 4
    init: str = "raw"           # 'raw' | 'deconv'
    lr_outer_schedule: str = "cosine" 
    lr_outer_floor_ratio: float = 0.02
    refresh_every: int = 0
    persist_optimizer: bool = False 
    verbose: bool = True
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class SCSI:
    """Learn P(C | Ce) from an ensemble of empirical covariances only -- factor-space / ODE variant."""

    def __init__(self, cfg: SCSIConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if cfg.encoding not in ("chol", "spectral"):
            raise ValueError("cfg.encoding must be 'chol' or 'spectral'")
        self.S = Chol(cfg.d, device=self.device) if cfg.encoding == "chol" else Spec(cfg.d, device=self.device)
        self.ref_sd = cfg.ref_sd if cfg.ref_sd is not None else 1.10 / math.sqrt(cfg.d)
        if cfg.ode_solver not in ("euler", "ralston"):
            raise ValueError("ode_solver must be 'euler' or 'ralston'")
        if cfg.sampler not in ("ode", "follmer"):
            raise ValueError("cfg.sampler must be 'ode' or 'follmer'")
        self.gen = torch.Generator().manual_seed(cfg.seed)  # CPU generator; moved to self.device at each use site
        self.opt = None  # lazily created AdamW, reused across outer iterations iff cfg.persist_optimizer
        torch.manual_seed(cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)
        torch.set_num_threads(cfg.threads)
        self.net: ResNetDrift | None = None
        self.ema_net: ResNetDrift | None = None
        self.hist: list[dict] = []

    def _make_drift(self, p, mu_y, sd_y, mu_ce, sd_ce, v_loc, v_scale):
        cfg = self.cfg
        return make_drift(cfg.arch, p, mu_y, sd_y, mu_ce, sd_ce, v_loc, v_scale,
                           hidden=cfg.hidden, depth=cfg.depth, d=cfg.d, nhead=cfg.nhead, N=cfg.N)

    def _outer_lr0(self, k: int) -> float:
        """Ceiling lr for outer iteration k's own (inner) CosineAnnealingLR refit -- see toys/scsi.py's version
        for the full rationale. 'const' keeps the original fixed ceiling cfg.lr*0.5; 'cosine' (default) anneals
        it across outer iterations 1..n_outer down to cfg.lr*0.5*cfg.lr_outer_floor_ratio at k=n_outer."""
        cfg = self.cfg
        lr_max = cfg.lr * 0.5
        if cfg.lr_outer_schedule == "const" or cfg.n_outer <= 1:
            return lr_max
        lr_min = lr_max * cfg.lr_outer_floor_ratio
        frac = min(max((k - 1) / (cfg.n_outer - 1), 0.0), 1.0)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * frac))

    def _sample_ref(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.S.p, dtype=DT, generator=self.gen).to(self.device)
        return self.ref_sd * z

    def _calibrate_velocity(self, Y_sample: torch.Tensor):
        base = self._sample_ref(Y_sample.shape[0])
        R = Y_sample - base
        return R.mean(0), R.std(0).clamp_min(0.08)

    # ---- initial sampleable prior: Gaussian in Y-coordinates, fitted to the observed Ye ---- #
    def _fit_init_prior(self, Ye: torch.Tensor):
        """Sampleable pi_0: Gaussian in Y-coordinates (unchanged from toys/scsi.py; agnostic to whether S is
        `Sym` or `Chol` -- it only calls S.encode/S.decode).

        init='raw'    : mean/cov of the observed Ye (over-dispersed by the sampling noise).
        init='deconv' : moment-matching deconvolution using only the known forward channel: the mean is
                        corrected for the bias of encode(Ce), and the covariance has the simulated channel-noise
                        covariance (at a typical C) subtracted, then inflated by `log_prior_inflate` and floored.
        Neither uses clean covariances; both are declared, data-informed initialisations."""
        cfg, S = self.cfg, self.S
        m = Ye.mean(0)
        Xc = Ye - m
        cov = Xc.T @ Xc / max(len(Ye) - 1, 1)
        if cfg.init == "deconv":
            for _ in range(3):
                Cty = S.decode(m).expand(20000, cfg.d, cfg.d)
                Yn = S.encode(sample_covariance_from_factors(Cty, cfg.N, self.gen), cfg.eps0)
                m = Ye.mean(0) - (Yn.mean(0) - m)
            Yn = Yn - Yn.mean(0)
            noise_cov = Yn.T @ Yn / (len(Yn) - 1)
            w, V = torch.linalg.eigh(cov - noise_cov)
            w_floor = 0.05 * w.abs().max()
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
            idx = torch.randint(0, n, (B,), generator=self.gen).to(Yp.device)
            return Yp[idx], Cp[idx]
        return f

    # ---- one supervised regression fit on pairs generated on the fly from a target bank ---- #
    def _fit(self, target_sampler, steps: int, lr0: float, lr1: float):
        cfg, S = self.cfg, self.S
        net = self.net
        net.train()            # sample_posterior may have called .eval() on this same net in between outer iters
        if cfg.persist_optimizer and self.opt is not None:
            opt = self.opt                                    # keep exp_avg/exp_avg_sq/step across outer iters
        else:
            opt = torch.optim.AdamW(net.parameters(), lr=lr0, weight_decay=cfg.weight_decay)
            if cfg.persist_optimizer:
                self.opt = opt
        for g in opt.param_groups:                             # restart this refit's own cosine cycle at lr0
            g["lr"] = lr0                                      # regardless of where the Adam momentum/variance
            g.pop("initial_lr", None)                          # state (kept, if persist_optimizer) left off
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr1)
        ema = self.ema_net
        t0 = time.time()
        run = 0.0
        for it in range(steps):
            Y, Sigma = target_sampler(cfg.batch)                                 # clean targets (B,p), (B,*S.sigma_shape)
            Ce = S.sample_Ce(Sigma, cfg.N, self.gen)
            Ce_tril = S.flatten(Ce, sym=True)                                    # conditioning only (not I(0))
            t = torch.rand(cfg.batch, 1, dtype=DT, generator=self.gen).to(self.device)
            if cfg.sampler == "follmer":
                I, R = follmer_interpolate(Ce_tril, Y, t, cfg.follmer_epsilon)   # bridge ANCHORED at Ye
            else:
                base = self._sample_ref(cfg.batch)                          # I(0): independent of Ye
                I = (1 - t) * base + t * Y                                   # deterministic straight-line interpolant
                R = Y - base                                                # constant-in-t velocity target
            pred = net(t.float(), I.float(), Ce_tril.float())
            loss = ((pred - R.float()) ** 2).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            ## Clip gradient for more stability
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(cfg.ema).add_(pn.detach(), alpha=1 - cfg.ema)
            run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
        return run, time.time() - t0

    @torch.no_grad()
    def _integrate_follmer(self, net, ye: torch.Tensor, T: int, epsilon: float, gen) -> torch.Tensor:
        """Stochastic Follmer bridge SDE, ANCHORED at ye (g starts AT Ye, not at an independent base) --
        matches spectra-space/covsi_pd_em_spectral.py's integrate_sde exactly (b_F=(1+t)*b_theta-g+lh,
        sigma_t=epsilon*sqrt(1-t^2)), the real noise injected at every step being the mechanism this
        module's deterministic ODE sampler lacks (see cfg.sampler's docstring)."""
        g_ = ye.clone().to(DT)
        dt = 1.0 / T
        sqrt_dt = math.sqrt(dt)
        for i in range(T):
            ti = i * dt
            tt = torch.full((g_.shape[0], 1), ti, device=self.device)
            b_theta = net(tt.float(), g_.float(), ye.float()).to(DT)
            b_F = (1.0 + ti) * b_theta - g_ + ye
            sigma_t = epsilon * math.sqrt(max(1.0 - ti ** 2, 0.0))
            xi = torch.randn(g_.shape, dtype=DT, generator=gen).to(self.device)
            g_ = g_ + b_F * dt + sigma_t * sqrt_dt * xi
        return g_

    # ---- deterministic ODE sampler (Euler / Ralston), or the stochastic Follmer SDE (cfg.sampler) ---- #
    @torch.no_grad()
    def sample_posterior(self, Ce: torch.Tensor, n_draws: int = 1, net: ResNetDrift | None = None,
                         n_steps: int | None = None, chunk: int = 4000, gen=None) -> torch.Tensor:
        """Ce: (M,d,d) observations -> draws of shape (M,n_draws,*S.sigma_shape) in `self.S`'s own factor
        space, NOT C -- (d,d) Cholesky factors for Chol (see Chol.decode / sample_covariance_from_factors
        to go on from here, no Cholesky needed either way), or (d,) eigenvalues for Spec (see
        SCSI.sample_posterior_sigma / Spec.decode to reconstruct an actual matrix -- needs Ce's own
        eigenvectors as a reference basis, which this method's chunking never retains, by design: nothing
        downstream of here, in SCSI.fit's self-consistent loop, ever needs one). Each draw integrates the
        learned ODE dY/dt = net(t,Y,Ye) from an independent Y(0) = base ~ N(0, ref_sd^2 I) to t=1,
        conditioned throughout on Ye = flatten(Ce) -- the randomness across the n_draws repeats of a
        single Ce comes only from the independent `base` draws, not from the (deterministic) integration."""
        cfg, S = self.cfg, self.S
        net = net or self.ema_net
        net.eval()            # dropout=0. now anyway, but inference must never run in train() mode
        gen = gen or self.gen
        J = n_steps or cfg.n_ode_steps
        Ce = Ce.to(self.device)
        Ce_tril = S.flatten(Ce, sym=True)                                          # (M,p) -- no Cholesky: Ce stays a matrix
        M = Ce_tril.shape[0]
        Ce_tril_rep = Ce_tril.repeat_interleave(n_draws, dim=0)          # (M*n,p)
        out = []
        Jf = n_steps or cfg.follmer_T
        for s in range(0, Ce_tril_rep.shape[0], chunk):
            ye = Ce_tril_rep[s:s + chunk]
            if cfg.sampler == "follmer":
                y = self._integrate_follmer(net, ye, Jf, cfg.follmer_epsilon, gen)
                out.append(y)
                continue
            #### ------- Initial condition
            z = torch.randn(ye.shape[0], S.p, dtype=DT, generator=gen).to(self.device)
            y = self.ref_sd * z                                             # independent I(0)
            dt = 1.0 / J
            for j in range(J):
                t = j * dt
                tt = torch.full((y.shape[0], 1), t, device=self.device)
                if cfg.ode_solver == "euler":
                    v = net(tt.float(), y.float(), ye.float()).to(DT)
                    y = y + dt * v
                else:                                                       # Ralston: two-stage, 2nd-order RK
                    v1 = net(tt.float(), y.float(), ye.float()).to(DT)
                    t2 = torch.full((y.shape[0], 1), t + 2.0 * dt / 3.0, device=self.device)
                    y2 = y + (2.0 * dt / 3.0) * v1
                    v2 = net(t2.float(), y2.float(), ye.float()).to(DT)
                    y = y + dt * (0.25 * v1 + 0.75 * v2)
            out.append(y)
        Y1 = torch.cat(out)
        Sigma = S.unflatten(Y1)                                          # factor space
        return Sigma.reshape(M, n_draws, *S.sigma_shape).cpu()

    @torch.no_grad()
    def sample_posterior_sigma(self, Ce: torch.Tensor, n_draws: int = 1, orientation_chunk: int = 20000,
                                orientation_steps: int = 30000, orientation_burn_in: int = 10000,
                                orientation_thin: int = 100, orientation_step_size: float = 0.2,
                                orientation_seed: int | None = None, **kwargs) -> torch.Tensor:
        """Like sample_posterior, but ALWAYS returns genuine (M,n_draws,d,d) lower-triangular Cholesky
        factors of the predicted covariance, regardless of `cfg.encoding` -- a representation-agnostic
        entry point for anything that needs to score draws with covutils.wishart_logscore (which assumes
        a triangular factor). For encoding="chol" this is sample_posterior unchanged (already a Cholesky
        factor).

        For encoding="spectral", sample_posterior's raw (d,) eigenvalues need an eigenBASIS to become an
        actual matrix. A FIXED V_ref = Ce's own eigenvectors for every draw (an earlier version of this
        method) is wrong: it collapses the genuinely random relative orientation R = V_Ce^T V_C between
        C's and Ce's eigenbases to a point mass at R=I, which only holds for the Bayes-MEAN estimator
        (the classical RIE result -- see Spec's docstring), not for actual posterior draws -- confirmed
        empirically (iw_ri_d8_inf: eigenvector-dependent stats ldv_top/ldv_bot had bias_z +4.7/-6.1 while
        eigenvalue-only stats were fine). Each draw's R is instead sampled from its own exact conditional
        law given (that draw's lam, Ce's eigenvalues) via utils_spectra.orientation_mcmc -- a
        Metropolis-Hastings sampler on O(d) targeting the Bingham-type kernel implied by the Wishart
        channel's Tr(C^-1 Ce) likelihood term. orientation_mcmc's cost is dominated by orientation_steps
        sequential MCMC steps, essentially INDEPENDENT of how many rows are batched together (same
        fixed-wall-clock-per-chain phenomenon as the HMC oracle in log_score_oracle.py) -- so orientation_chunk
        should be as LARGE as GPU memory allows (its kept-sample buffer is the memory bottleneck, not the
        MCMC itself), not small like the ODE-integration `chunk` kwarg above. Defaults
        (steps=30000/burn_in=10000/thin=100) are the ones verified by an SBC rank test to calibrate
        correctly at d=8, N=40 (the untuned defaults in utils_spectra.py itself under-mix and fail that
        test) -- re-verify if d or N change a lot. Only ONE kept sample (the chain's last) is used per
        row, since each row needs exactly one posterior draw of R to pair with its already-sampled lam."""
        cfg, S = self.cfg, self.S
        lam_or_sigma = self.sample_posterior(Ce, n_draws, **kwargs)              # (M,n_draws,*S.sigma_shape)
        if cfg.encoding == "chol":
            return lam_or_sigma
        Ce_dev = Ce.to(self.device).to(DT)
        Lambda_e, V_Ce = torch.linalg.eigh(Ce_dev)                               # (M,d),(M,d,d), ascending -- matches Spec's convention
        M, R = lam_or_sigma.shape[0], lam_or_sigma.shape[1]
        lam = lam_or_sigma.to(self.device).to(DT)                                # (M,n_draws,d)

        lam_flat = lam.reshape(M * R, cfg.d)
        Lambda_e_flat = Lambda_e.unsqueeze(1).expand(M, R, cfg.d).reshape(M * R, cfg.d)
        V_Ce_flat = V_Ce.unsqueeze(1).expand(M, R, cfg.d, cfg.d).reshape(M * R, cfg.d, cfg.d)
        Rot = torch.empty(M * R, cfg.d, cfg.d, dtype=DT, device=self.device)
        for s in range(0, M * R, orientation_chunk):
            sl = slice(s, s + orientation_chunk)
            chain, _ = orientation_mcmc(lam_flat[sl].float(), Lambda_e_flat[sl].float(), cfg.N,
                                         n_steps=orientation_steps, step_size=orientation_step_size,
                                         burn_in=orientation_burn_in, thin=orientation_thin,
                                         seed=orientation_seed)
            Rot[sl] = chain[-1].to(DT)                                           # one posterior draw per row

        V_C = V_Ce_flat.matmul(Rot)                                              # V_C = V_Ce @ R
        C_hat = (V_C * lam_flat.unsqueeze(-2)).matmul(V_C.transpose(-1, -2))     # (M*n_draws,d,d)
        eye = torch.eye(cfg.d, dtype=DT, device=self.device) * 1e-10
        Sigma = torch.linalg.cholesky(C_hat + eye)
        return Sigma.reshape(M, R, cfg.d, cfg.d).cpu()

    # ---- Algorithm 1 (unchanged control flow; see toys/scsi.py) ---- #
    def fit(self, Ce_obs, callback=None, infinite_batch: int | None = None, resume: dict | None = None, verbose: bool=False):
        """Ce_obs: either
             - a (M,d,d) tensor of recorded empirical covariances (finite-data mode): the SAME ensemble is
               reconditioned on at every outer iteration, or
             - a callable B -> (B,d,d) tensor (infinite-data mode): a brand-new (C, Ce) ensemble of size
               `infinite_batch` is drawn fresh at the E-step of *every* outer iteration (including the very
               first, used to fit the initial prior), instead of reusing one fixed observed set.
           `infinite_batch` is required (and only used) when Ce_obs is a callable.

           resume: optional dict to continue a previous run instead of starting from the sampleable-prior
           init -- {"net_state", "ema_state", "mu_y", "sd_y", "mu_ce", "sd_ce", "velocity_loc", "velocity_scale",
           "gen_state" (optional, this SCSI's own self.gen, CPU ByteTensor), "start_k"} -- SIX normalisation
           tensors (not two: see Drift's docstring), unlike toys/scsi.py's two-tensor contract. Skips
           _fit_init_prior and the step-1 fit entirely; the outer loop resumes at start_k+1. Network
           architecture (hidden/depth) must already match cfg."""
        cfg, S = self.cfg, self.S
        infinite = callable(Ce_obs)
        if resume is not None:
            if verbose:
                print("Loading checkpoint model")
            if not infinite:
                Ce_cur = Ce_obs.to(DT).to(self.device)
            mu_y, sd_y = resume["mu_y"].to(self.device), resume["sd_y"].to(self.device)
            mu_ce, sd_ce = resume["mu_ce"].to(self.device), resume["sd_ce"].to(self.device)
            v_loc, v_scale = resume["velocity_loc"].to(self.device), resume["velocity_scale"].to(self.device)
            self.net = self._make_drift(S.p, mu_y, sd_y, mu_ce, sd_ce, v_loc, v_scale).to(self.device)
            self.net.load_state_dict(resume["net_state"])
            self.net.train()
            self.ema_net = self._make_drift(S.p, mu_y, sd_y, mu_ce, sd_ce, v_loc, v_scale).to(self.device)
            self.ema_net.load_state_dict(resume["ema_state"])
            for q in self.ema_net.parameters():
                q.requires_grad_(False)
            self.ema_net.eval()            # never trained directly (EMA-only) -- always used for sampling
            if resume.get("gen_state") is not None:
                self.gen.set_state(resume["gen_state"])
            start_k = resume["start_k"]
        else:
            if verbose:
                print("No model to load")
            if infinite:
                assert infinite_batch is not None, "infinite_batch must be given when Ce_obs is a sampler (infinite-data mode)"
                Ce_cur = Ce_obs(infinite_batch).to(DT).to(self.device)
            else:
                Ce_cur = Ce_obs.to(DT).to(self.device)
            Ce_tril = S.flatten(Ce_cur, sym=True)             # conditioning stats: Ce stays a matrix, no Cholesky at all
            self._fit_init_prior(Ce_tril)
            mu_ce, sd_ce = Ce_tril.mean(0), Ce_tril.std(0).clamp_min(1e-3)

            Y0 = self._sample_init_prior(cfg.pool_init)
            Sigma0 = S.canonicalize(S.unflatten(Y0))
            Y0 = S.flatten(Sigma0)
            mu_y, sd_y = Y0.mean(0), Y0.std(0).clamp_min(1e-3)
            v_loc, v_scale = self._calibrate_velocity(Y0)

            self.net = self._make_drift(S.p, mu_y, sd_y, mu_ce, sd_ce, v_loc, v_scale).to(self.device)
            self.net.train()
            self.ema_net = self._make_drift(S.p, mu_y, sd_y, mu_ce, sd_ce, v_loc, v_scale).to(self.device)
            self.ema_net.load_state_dict(self.net.state_dict())

            ####------- Freeze EMA parameters
            for q in self.ema_net.parameters():
                q.requires_grad_(False)
            self.ema_net.eval()            # never trained directly (EMA-only) -- always used for sampling

            # step 1: initialise with the sampleable prior pi_0 -- (Y, Sigma) pairs, NOT (Y, C): _fit turns
            # Sigma into a noisy Ce itself (sample_covariance_from_factors), no Cholesky needed here.
            if verbose:
                print("Start warm-up fit")
            pool0 = (Y0, Sigma0)
            loss, dt = self._fit(self._pool_sampler(pool0), cfg.steps_first, cfg.lr, cfg.lr_end)
            self._log(0, loss, dt, note="init prior")
            if verbose:
                print("Ended warm-up fitting")
            if callback:
                callback(self, 0)
            start_k = 0

        # steps 2-3: reconstruct at recorded observations, re-corrupt, refit
        if verbose:
            print("Start EM loops")
        for k in tqdm(range(start_k + 1, cfg.n_outer + 1)):
            t0 = time.time()
            if infinite:
                Ce_cur = Ce_obs(infinite_batch).to(DT).to(self.device)       # brand-new (C, Ce) ensemble this E-step
            #Sample posterior for training dataset
            Sigma_rec = self.sample_posterior(Ce_cur, cfg.n_recon).to(self.device)   # (M,R,*S.sigma_shape)
            Sigma_bank = S.canonicalize(Sigma_rec.reshape(-1, *S.sigma_shape))
            Y_bank = S.flatten(Sigma_bank)
            bank = (Y_bank, Sigma_bank)                                      # clean targets (Y, Sigma)
            #Compute normalization constants
            if cfg.refresh_every and k % cfg.refresh_every == 0:
                Ce_tril_now = S.flatten(Ce_cur, sym=True)
                mu_ce_now, sd_ce_now = Ce_tril_now.mean(0), Ce_tril_now.std(0).clamp_min(1e-3)
                mu_y_now, sd_y_now = Y_bank.mean(0), Y_bank.std(0).clamp_min(1e-3)
                v_loc_now, v_scale_now = self._calibrate_velocity(Y_bank)
                self.net.refresh_norm(mu_y_now, sd_y_now, mu_ce_now, sd_ce_now, v_loc_now, v_scale_now)
                self.ema_net.refresh_norm(mu_y_now, sd_y_now, mu_ce_now, sd_ce_now, v_loc_now, v_scale_now)
            t_rec = time.time() - t0
            lr0_k = self._outer_lr0(k)
            lr1_k = lr0_k * (cfg.lr_end / cfg.lr)          # keep the same inner ceiling:floor ratio as cfg.lr/cfg.lr_end
            loss, dt = self._fit(self._pool_sampler(bank), cfg.steps_outer, lr0_k, lr1_k)
            self._log(k, loss, dt + t_rec, note=f"bank={bank[0].shape[0]} lr0={lr0_k:.1e}" + (" [fresh]" if infinite else ""))
            if callback:
                callback(self, k)
        if verbose:
            print("End of training")
        return self

    def _log(self, k, loss, dt, note=""):
        self.hist.append(dict(k=k, loss=loss, sec=dt))
        if self.cfg.verbose:
            print(f"  [SCSI] outer {k}: loss={loss:.4f}  ({dt:.0f}s)  {note}", flush=True)

    # ---- supervised control: train on true (C, Ce) pairs, for diagnostics only ---- #
    def fit_supervised(self, C_true_sampler, Ce_ref: torch.Tensor, steps: int | None = None):
        """C_true_sampler: B -> (B,d,d) true, clean prior draws. The ONE Cholesky call here (S.encode(Cp))
        mirrors toy_train.py's Ce_tr closure: a prior sample is turned into a factor exactly once, then
        everything downstream (here, inside _fit via sample_covariance_from_factors) stays in factor/matrix
        space -- no further decomposition."""
        cfg, S = self.cfg, self.S
        Ce_tril = S.flatten(Ce_ref.to(DT).to(self.device), sym=True)  # conditioning stats, same convention as fit()
        mu_ce, sd_ce = Ce_tril.mean(0), Ce_tril.std(0).clamp_min(1e-3)
        Cp = C_true_sampler(cfg.pool_init).to(self.device)
        Sigma_p = S.encode(Cp, cfg.eps0)                               # the one allowed Cholesky: true C -> Sigma
        Y_p = S.flatten(Sigma_p)
        mu_y, sd_y = Y_p.mean(0), Y_p.std(0).clamp_min(1e-3)           # real Y-space samples available here, use them
        v_loc, v_scale = self._calibrate_velocity(Y_p)

        self.net = self._make_drift(S.p, mu_y, sd_y, mu_ce, sd_ce, v_loc, v_scale).to(self.device)
        self.net.train()
        self.ema_net = self._make_drift(S.p, mu_y, sd_y, mu_ce, sd_ce, v_loc, v_scale).to(self.device)
        self.ema_net.load_state_dict(self.net.state_dict())
        for q in self.ema_net.parameters():
            q.requires_grad_(False)
        self.ema_net.eval()            # never trained directly (EMA-only) -- always used for sampling
        loss, dt = self._fit(self._pool_sampler((Y_p, Sigma_p)), steps or cfg.steps_first, cfg.lr, cfg.lr_end)
        self._log(0, loss, dt, note="supervised")
        return self
