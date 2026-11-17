"""The data of the evaluation scripts, drawn from seeded generators -- the single place where they are defined, so that
the log-score scripts (toy_logscore_oracle.py, toy_logscore_scsi.py), the W1 scripts (toy_W1_oracle.py, toy_W1_scsi.py)
and toy_dt_study.py all score the same pairs or test tasks and fit the same IW-ML baseline.

Test tasks of W1: make_testset, 128 (C, Ce) from Generator(777).

Validation pairs (Ce_cal, Ce_val) of the log-score: n_val latent C from the prior, two noisy observations each. Two
roles: "select" (the trajectory along training and, with finite data, the choice of best_k) and "report" (the value at
best_k in {toy}_logscore.json, read by the tables). val_seed(toy, role):
  M=inf  : None for both roles = the pairs REPLAYED from toy_train.py: Generator(1000+seed), the first draws of its
           stream (nothing is drawn before C_val in infinite mode). One set: trajectory and report share it.
  finite : independent pairs, Generator(2000) to select, Generator(2001) to report, so the reported value carries no
           selection bias. (To move M=inf to independent pairs as well, give it SEEDS here.)

IW-ML fit ensemble:
  W1 (toy_W1_oracle.py)         : Generator(1000+seed), M noisy Ce (finite: exactly toy_train.py's training ensemble),
                                  or m_inf=5000 for M=inf; fit cached in {toy}_iwfit_ref.pt
  log-score, finite             : the same (same cache)
  log-score, M=inf              : the m_inf Ce that follow the replayed pairs in toy_train.py's stream, fitted afresh on
                                  the GPU -- kept as computed for the paper (it differs from the W1 fit ensemble)
"""
import os
import torch
from scsi import wishart_channel
from covutils import IWPrior
import paths

SEEDS = {"select": 2000, "report": 2001}           # independent validation pairs (finite data)
M_INF = 5000                                       # IW-ML fit ensemble size for M=inf (toy_train's E-step batch)


def make_testset(toy, n=128, seed=777):
    """(C, Ce) of the n test tasks of the W1 evaluation"""
    g = torch.Generator().manual_seed(seed)
    C = toy.sample_prior(n, g)
    return C, wishart_channel(C, toy.N, g)


def val_seed(toy, role):
    """generator seed of the validation pairs of a role, None = replayed from toy_train.py"""
    assert role in SEEDS, role
    return None if toy.M == -1 else SEEDS[role]


def _pairs(toy, n_val, gen, device):
    C = toy.sample_prior(n_val, gen)
    return wishart_channel(C, toy.N, gen).to(device), wishart_channel(C, toy.N, gen).to(device)


def val_pairs(toy, role, n_val=10, seed=0, device="cpu"):
    """(Ce_cal, Ce_val) of a role (see the module docstring)"""
    vs = val_seed(toy, role)
    if vs is not None:
        return _pairs(toy, n_val, torch.Generator().manual_seed(vs), device)
    assert toy.M == -1, "replayed pairs of finite-data runs would need toy_train.py's C_tr draw first"
    return _pairs(toy, n_val, torch.Generator().manual_seed(1000 + seed), device)


def fit_iw_cached(Ce_tr, N, cache, iters=800):
    """IW-ML baseline fit, cached per toy: its training ensemble is seeded and identical for every checkpoint of a
    toy, so refitting it at every evaluation (~30 s, 800 Adam steps on the CPU) is pure waste."""
    if os.path.exists(cache):
        z = torch.load(cache, map_location="cpu")
        if z["M"] == Ce_tr.shape[0] and z["iters"] == iters:
            iw = IWPrior.__new__(IWPrior)
            iw.N, iw.d, iw.Psi0, iw.nu0, iw.loglik = N, Ce_tr.shape[-1], z["Psi0"], z["nu0"], z["loglik"]
            return iw
    iw = IWPrior(Ce_tr, N, iters=iters)
    torch.save({"M": Ce_tr.shape[0], "iters": iters, "Psi0": iw.Psi0, "nu0": iw.nu0, "loglik": iw.loglik}, cache)
    return iw


def iwml_w1(toy, seed=0, m_inf=M_INF):
    """IW-ML of the W1 evaluation (and of the finite-data log-score): fitted on Generator(1000+seed), M (or m_inf) Ce"""
    g = torch.Generator().manual_seed(1000 + seed)
    Ce_tr = wishart_channel(toy.sample_prior(toy.M if toy.M > 0 else m_inf, g), toy.N, g)
    return fit_iw_cached(Ce_tr, toy.N, f"{paths.res_dir(toy.name)}/{toy.name}_iwfit_ref.pt")


def iwml_logscore_fit(toy, n_val=10, seed=0, device="cpu", m_inf=M_INF):
    """IW-ML of the log-score (see the module docstring)"""
    if toy.M != -1:
        return iwml_w1(toy, seed)
    g = torch.Generator().manual_seed(1000 + seed)
    _pairs(toy, n_val, g, device)                                        # replay: the pairs come first in the stream
    return IWPrior(wishart_channel(toy.sample_prior(m_inf, g), toy.N, g).to(device), toy.N, iters=800)
