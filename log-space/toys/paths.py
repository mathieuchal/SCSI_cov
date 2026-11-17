"""Folders, production runs and evaluation grid of the d=8 toy experiments, by data regime -- the single place where
they are defined.

Regimes (the suffix of the toy name after "_d8_"):
  "inf"   : infinite data (toy.M = -1), toys {base}_d8_inf    -> results/toys_M=-1,   figures and tables in figs/
  "M5000" : finite data, one fixed training ensemble of M=5000 noisy Ce, toys {base}_d8_M5000
                                                                -> results/toys_M=5000, figures and tables in figs_M=5000/
The scripts that handle one toy derive the folder from its name (res_dir(name)); the scripts that aggregate toys into
tables or figures take --data (add_data_arg) and use res_dir(regime) / figs_dir(regime).
"""
BASES = ("iw_ri", "iw_nonri", "factor", "iso_diag", "ar1", "diag", "hetero")
# production architecture of each toy (toy_train.py --tag), the same in every regime
ARCH = {"iw_ri": "mlp_w512_d4", "iw_nonri": "mlp_w512_d4", "factor": "mlp_w512_d4", "iso_diag": "mlp_w1024_d4",
        "ar1": "resnet_w1024_d4", "diag": "mlp_w512_d4", "hetero": "mlp_w512_d4"}
RES = {"inf": "results/toys_M=-1", "M5000": "results/toys_M=5000"}
FIGS = {"inf": "figs", "M5000": "figs_M=5000"}
REGIMES = tuple(RES)
# evaluation grid of the M=inf production runs (log-score and W1 trajectories, --ks grid; the finite-data runs are
# evaluated on every checkpoint)
GRID = [2, 4, 6, 8, 10, 12, 16, 20, 24, 60, *range(100, 1000, 40), 1000, *range(1100, 2001, 100)]


def toy_name(base, regime="inf"):
    return f"{base}_d8_{regime}"


def regime(name):
    """'inf' or 'M5000' from a toy name ({base}_d8_{regime}) or a regime string itself."""
    r = name if name in RES else name.rsplit("_d8_", 1)[-1]
    if r not in RES:
        raise ValueError(f"{name}: unknown data regime (expected a toy name ending in one of _d8_{'/_d8_'.join(RES)})")
    return r


def base(name):
    return name.rsplit("_d8_", 1)[0]


def res_dir(name_or_regime="inf"):
    return RES[regime(name_or_regime)]


def figs_dir(name_or_regime="inf"):
    return FIGS[regime(name_or_regime)]


def tag(name):
    """production run tag of a toy, e.g. tag('iso_diag_d8_M5000') = 'mlp_w1024_d4'."""
    return ARCH[base(name)]


def tags(regime_="inf"):
    """{toy name: tag} of the seven production runs of a regime."""
    return {toy_name(b, regime_): ARCH[b] for b in BASES}


def add_data_arg(ap):
    ap.add_argument("--data", default="inf", choices=REGIMES,
                    help="data regime: inf (results/toys_M=-1, figs/) or M5000 (results/toys_M=5000, figs_M=5000/)")
