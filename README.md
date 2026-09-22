# SC-SI covariance posterior — experiments

Code accompanying the working draft **"Self-Consistent Stochastic Interpolants for Covariance Prior
and Posterior Learning"** (`CovariancePosteriorPrior.pdf`, an ICLR 2027 submission that itself reports
no numerical results — Section 5 of the paper only proposes experiments). This repository implements
the method and runs it on controlled toy models and two real-data applications.

The full write-up of every experiment — protocols, tables, calibration checks, and caveats — is
**`experiments/REPORT.md`**. This README only covers installation and how to run things.

## What's here

- `experiments/scsi.py` — the method: Wishart-channel simulator, matrix-logarithm coordinates, the
  conditional stochastic interpolant (Föllmer sampler), and the self-consistent EM loop (Algorithm 1
  of the paper; the general matrix-log version — Algorithm 2, spectral + orientation MCMC, is not
  implemented).
- `experiments/covutils.py` — baselines and metrics shared across experiments: a conjugate
  inverse-Wishart prior fitted by marginal likelihood, OAS linear shrinkage, Ledoit–Wolf analytical
  nonlinear shrinkage, predictive log-scores, directional-variance PIT/coverage, Riemannian distance.
- `experiments/toys.py` — controlled models with a *reference* posterior to validate against: an
  inverse-Wishart prior (rotationally invariant or not, exact posterior) and a 2-factor +
  idiosyncratic-noise model (no conjugate posterior; a batched HMC sampler serves as reference,
  validated by R-hat and simulation-based calibration).
- `experiments/bench_iw.py`, `toy_train.py`, `toy_eval.py`, `toy_traj.py` — train/validate SC-SI
  against exact or reference posteriors.
- `experiments/eeg_*.py` — EEG experiment (PhysioNet Motor Movement/Imagery): does quantifying the
  covariance posterior improve predictive calibration and downstream classification over short,
  noisy windows, across subjects?
- `experiments/fin_*.py` — finance experiment (Ken French 48-industry daily returns): does the
  posterior improve out-of-sample minimum-variance portfolios and dynamic selection of "meaningful"
  eigendirections in a factor model, over point-estimate baselines?
- `experiments/*_figs.py`, `plotstyle.py` — figures for each experiment.
- `experiments/REPORT.md` — the results.

## Setup

Python 3.11. Install the pinned dependencies (see `requirements.txt` for why the pins matter — in
short, a plain `pip install mne` will upgrade numpy in a way that breaks this torch build):

```bash
pip install -r requirements.txt
```

## Data

Neither dataset is checked into the repo (`data/` is gitignored). Fetch and preprocess them once,
from `experiments/`:

```bash
cd experiments
./data/download_eeg.sh                # ~450MB from PhysioNet; pass a subject count for a quick smoke test
python data/prepare_finance.py         # Ken French 48-industry daily returns
python eeg_data.py 4                   # turn the raw EEG into windowed + split-half covariances
```

## Running the experiments

All commands are run from `experiments/`. See `experiments/REPORT.md` §7 ("Reproduce") for the full
list, including the toy-model pipeline; the essentials:

```bash
cd experiments
python bench_iw.py --d 8 --Nfac 5 --M 2000 --kind ri --init deconv --inflate 1.5 \
    --steps_first 4000 --steps_outer 800 --n_outer 20   # validate against the exact IW posterior
python eeg_exp.py --win 4 --n_outer 12 --neff_scale 0.65
python eeg_exp2.py eeg_w4_s0_k1.0_e0.65 12 0.65
python fin_exp.py --d 12 --n_outer 12
python eeg_figs.py eeg_w4_s0_k1.0_e0.65; python fin_figs.py; python bench_figs.py
```

Each `*_exp*.py` / `toy_train.py` script trains its own SC-SI model and caches checkpoints under
`results/`; re-running with the same arguments reuses them. Training runs are CPU-bound and take
roughly 20–45 minutes each on 2–6 threads; there is no GPU dependency.

Small result summaries (`results/**/*.json`, `*.txt`) are checked in — they're the source of the
numbers in `REPORT.md`. Large binary artifacts (posterior draws, model checkpoints: `*.npz`, `*.pt`)
are gitignored and regenerate from the commands above.

## Status

This is a research codebase for one paper's experiments, not a general-purpose library. No license
has been chosen yet.
