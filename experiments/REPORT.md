# Real-data experiments for "Self-Consistent Stochastic Interpolants for Covariance Prior and Posterior Learning"

Everything here is reproducible from `experiments/` (see the last section). All numbers below come from the
result files in `results/`; nothing is copied from the paper, which claims no numerical results.

## 0. Bottom line

| | Verdict | What the posterior buys you (measured) | What it does **not** buy you |
|---|---|---|---|
| **EEG (PhysioNet, 109 subjects, alpha band, 4-s windows, d = 8)** | **Good fit, worth a main-text experiment.** | Calibrated predictive intervals for *nonlinear functionals of C*: the 80 % interval for the log condition number covers 0.80 (split 1) / 0.76 (split 2), against 0.53 / 0.54 for the sample covariance, 0.67 / 0.66 for Ledoit–Wolf nonlinear shrinkage and 0.56 / 0.57 for a fitted inverse-Wishart. Predictive log-score gain over the sample covariance: +5.7 nats/window in a common-C split-half test (both splits) and +16.5 / +21.6 nats across consecutive windows. Against the strongest point-estimate baseline, Ledoit–Wolf nonlinear shrinkage (NLS), SC-SI is ahead by +3.0 / +2.9 nats (split-half) and +8.0 / +8.9 nats (next window), all paired CIs excluding 0. | Not calibrated when the covariance itself drifts between windows (next-window coverage 0.38/0.65/0.83 at nominal 0.5/0.8/0.95). NLS is a point estimate and gives no intervals: its coverage equals the sample covariance's. The open-vs-closed Riemannian distance stays under-covered for every method (0.43 / 0.46 at 80 %). Posterior-averaged eyes-open/closed classification improves NLL slightly (−0.04 / −0.03) but the accuracy-at-50 %-coverage gain did **not** replicate across splits. |
| **Finance (Ken French 48 industries, d = 12 baskets, 1995–2026)** | **Weaker fit; useful as a secondary experiment, and only after the reframing below.** | The SC-SI Stein-optimal covariance lowers out-of-sample minimum-variance portfolio variance by 2.5 % (N_eff = 19) / 2.3 % (N_eff = 40) relative to OAS shrinkage, by 6.5 % / 3.9 % relative to Ledoit–Wolf nonlinear shrinkage, and by 12–13 % relative to the sample covariance. Factor count chosen by posterior expected Stein loss beats the Gavish–Donoho hard threshold by 2.8 % / 3.7 % and the IW-based Bayes rule by 2.9 % / 1.2 %. | **Conclusions about "dynamic thresholding" depend on the assumed N_eff.** At N_eff = 19 the Bayes-K rule is no better than a fixed K = 3 (−0.4 %, CI [−1.1, +0.4]); at N_eff = 40 it is (−2.9 %) but chooses K ≈ 5 on average and changes K by 1.4 per month (hard threshold: 0.4). The posterior-*mean* advantage over OAS is not robust (−3.1 % → −0.7 %, n.s.). Ledoit–Wolf NLS is worse than OAS as a minimum-variance input (+3.9 % / +1.6 %). N_eff = 19 (the value that calibrates forecasts) mostly absorbs covariance drift, not sampling noise. |
I would put EEG in the main text and finance in an appendix or a "limitations of the Wishart channel" discussion.
Section 5 lists five applications I think fit better than finance.

## 1. What was implemented and how it was validated

`scsi.py` implements **Algorithm 1** (general, matrix-logarithm version): Wishart-channel simulator, `svec(log C)`
coordinates, the conditional Brownian interpolant (eq. 5), the Föllmer sampler (eq. 7, Euler–Maruyama, 64 steps),
and the self-consistent reconstruct → re-corrupt → refit loop. The drift is an MLP with the exact `t → 1` limit
`b_1(y, Ye) = y − Ye` as a skip connection. Not implemented: Algorithm 2 (spectral learning + orientation MCMC).

Initial prior π₀ is a log-Gaussian fitted to the observed ensemble by moment-matching deconvolution
(uses only the ensemble and the known channel; no clean covariances). σ = √(2/N).

**Validation against the exact inverse-Wishart posterior** (`bench_iw.py`, Appendix H diagnostics; the learner sees only noisy Ce; rows 1–2 use 256 test tasks × 512 draws, rows 3–4 the intermediate 64 × 128 evaluation):

| setting | outer iters | e_M (SCM: ·) | coverage 50/80/95 (PIT) | orientation 2nd-moment ratio (target 1) |
|---|---|---|---|---|
| supervised control, d=4, N=8 | – | 0.026 (0.507) | 0.508 / 0.809 / 0.954 | 0.89 |
| self-consistent, d=4, N=8, M=5000 | 30 | 0.049 (0.51) | 0.485 / 0.782 / 0.938 | 1.00 |
| self-consistent, d=8, N=40, M=2000 (the EEG regime; 64 test tasks, 128 draws) | 3 | 0.065 (0.21) | 0.494 / 0.794 / 0.945 | 0.96 |
| same, run longer (64 test tasks) | 20 | 0.125 (0.21) | 0.519 / 0.818 / 0.954 | 0.80 |

The last two rows matter for the real-data protocol. With a finite ensemble (M = 2000), too many EM iterations
**over-sharpen** the learned prior (log-det sd ratio 0.92, orientation ratio 0.80), the nonparametric-MLE-tends-to-atoms
effect. All real-data runs therefore select the outer iteration on held-out *validation subjects/periods*
(predictive log-score), never on the test set. Figure: `figs/bench_iw_convergence.png`.

**Nonlinear-shrinkage baseline (added after review).** `covutils.nls_shrink` implements the analytical nonlinear shrinkage of
Ledoit & Wolf (Ann. Statist. 48(5), 2020), the fast successor of the QuEST estimator cited in the paper (2012). It is rotation-equivariant, keeps the sample
eigenvectors and replaces each sample eigenvalue l_i by l_i / [(πc·l_i·f(l_i))² + (1 − c − πc·l_i·Hf(l_i))²], c = d/N, with f an Epanechnikov-kernel
estimate of the sample spectral density (local bandwidth N^(−1/3)·l_j) and Hf its Hilbert transform. Like OAS it uses only (Ce, N); N is the effective N.
Checked in `test_nls.py` against known truth (Frobenius error ‖Ĉ − C‖):

| setting | sample cov. | OAS | **NLS** | oracle (u_iᵀCu_i) / Bayes |
|---|---|---|---|---|
| d = 200, n = 600, spikes (20, 10) + bulk 1 | 9.37 | 8.52 | **4.68** | 4.48 |
| d = 200, n = 600, three levels {1, 2, 5} | 18.43 | 14.41 | **13.46** | 13.28 |
| d = 200, n = 600, identity | 8.17 | 0.09 | 0.58 | 0.00 |
| RI inverse-Wishart, **d = 8, N = 28** (EEG size) | 1.63 | 1.35 | **1.44** | 1.29 (exact posterior mean, uses the prior) |

In the large-d regime NLS reaches the oracle, as in the literature. At the small d of our experiments it is only a moderate baseline (it estimates a spectral density from 8–12 eigenvalues,
and on a spherical prior OAS's target is favourable), which should be kept in mind when reading the tables below.

## 2. EEG experiment

**Data.** PhysioNet EEG Motor Movement/Imagery, runs R01 (eyes open) and R02 (eyes closed), 109 subjects, 160 Hz.
8 channels (Fz, C3, Cz, C4, Pz, O1, Oz, O2), common-average reference over the 64 channels, 8–13 Hz band-pass,
2 s trimmed at each end, 4-s non-overlapping windows (14 per run). Task = (subject, condition, window), condition
unlabeled in the prior (pooled). Subject-disjoint split: 70 train / 10 validation / 29 test; the prior is learned
from the 1960 training-window covariances only.

**Effective N (an assumption the paper flags, tested here).** Band-limited samples are serially correlated, so the
Wishart N must be an effective sample size. The autocorrelation formula gives N_eff ≈ 43 for 4 s (matches the
time-bandwidth product 2·B·T = 40). The directly measured test–retest variability of interleaved half-windows on
*training subjects* gives N ≈ 24–36 (about 27; the range is the spread over channels and estimators), because alpha bursting makes window power more variable than a stationary
Gaussian. I use **N = 28** (N/d = 3.5). The first run with N = 43 gives the same ranking of methods (Sec. 2.4).

**Baselines** (all from the same noisy ensemble and N): sample covariance; OAS linear shrinkage (the Gaussian-model
form of Ledoit–Wolf, shrinks toward a scaled identity); **Ledoit–Wolf analytical nonlinear shrinkage (NLS)**; a conjugate inverse-Wishart prior fitted by marginal likelihood (the paper's parametric
baseline); and, for reference only, a "long-run proxy" that averages ~50 s of other windows (not causal). The three shrinkage/plug-in
baselines are turned into predictive distributions the same way: Wishart(N) noise around the point estimate. They therefore carry no estimation uncertainty of their own.

### 2.1 Protocol A: split-half (common C holds)
Calibration half A and validation half B are the even/odd 1-s chunks of the same 8-s epoch. The posterior is computed
from A and the predictive law of B (draw C, then Wishart(N) noise, eq. 169) is scored. 29 test subjects, 812 cases,
subject-level bootstrap CIs. `results/eeg_w4_s0_k1.0_e0.65_splithalf.json`, figure `figs/eeg_w4_s0_k1.0_e0.65_E1_calibration.png`.

| | log-score gain vs sample cov. (nats/window) | directional variances, coverage 50/80/95 |
|---|---|---|
| Sample cov. | 0 | 0.383 / 0.647 / 0.834 |
| OAS shrinkage | −16.0 [−22.4, −10.8] | 0.362 / 0.628 / 0.818 |
| **LW nonlinear shrinkage** | **+2.8 [+2.2, +3.4]** | 0.378 / 0.648 / 0.834 |
| IW conjugate (ML-fitted) | +0.1 [−1.8, +1.8] | 0.464 / 0.761 / 0.922 |
| **SC-SI** | **+5.8 [+4.2, +7.1]** | 0.480 / 0.764 / 0.917 |

Paired: SC-SI − NLS = **+3.0 [+1.5, +4.1]** nats; SC-SI − IW = +5.6 [+4.1, +7.2]. NLS is the best non-Bayesian baseline here (it is the only one that improves on the sample covariance), but its intervals are as under-covered as the sample covariance's.

Coverage of the central 80 % predictive interval for **nonlinear functionals of C** (nominal 0.80):

| functional | sample cov. | LW NLS | OAS | IW conj. | **SC-SI** |
|---|---|---|---|---|---|
| log det C | 0.34 | 0.46 | 0.17 | 0.43 | **0.68** |
| log power at Oz | 0.66 | 0.66 | 0.64 | 0.74 | **0.77** |
| top-eigenvalue share | 0.72 | 0.69 | 0.56 | 0.82 | **0.81** |
| log condition number | 0.53 | 0.67 | 0.14 | 0.56 | **0.80** |
| open-vs-closed Riemannian distance | 0.25 | 0.38 | 0.39 | 0.36 | 0.43 |

Reading: on directional variances the parametric IW baseline is nearly as calibrated as SC-SI, so a reviewer will
say the gain is small there. The separation appears on **spectral functionals** (log det, condition number), where the
one-parameter IW shrinkage is the wrong shape. NLS corrects the spectral bias of the sample covariance (log condition number 0.53 → 0.67) but, being a point estimate, still under-covers.
That is where I would place the paper's claim. The Riemannian distance is the honest failure: no method reaches nominal coverage.

### 2.2 Protocol B: next window (drift included)
Calibrate on window j, validate on window j+1 of the same run (`results/eeg_w4_s0_k1.0_e0.65_results.json`).

| | log-score gain (nats) | directional coverage 50/80/95 |
|---|---|---|
| Sample cov. | 0 | 0.308 / 0.541 / 0.717 |
| OAS | −7.9 [−14.8, −2.7] | 0.298 / 0.526 / 0.704 |
| **LW nonlinear shrinkage** | **+8.6 [+7.1, +10.0]** | 0.309 / 0.545 / 0.719 |
| IW conjugate | +8.9 [+5.8, +11.8] | 0.396 / 0.664 / 0.847 |
| **SC-SI** | **+16.5 [+13.4, +19.5]** | 0.382 / 0.651 / 0.828 |
| long-run proxy (reference) | +23.5 | – |

SC-SI closes 70 % of the gap between the sample covariance and the (non-causal) long-run reference; NLS 36 %, IW 38 %.
Paired: SC-SI − NLS = **+8.0 [+6.0, +9.8]** nats, SC-SI − IW = +7.6 [+5.4, +9.7].
Coverage is below nominal for every method: the covariance moves between windows and the common-C Wishart channel has no
term for that. This is the limitation the paper already states for time-indexed data. The gap between Protocols A and B
is itself a measure of window-to-window nonstationarity.

### 2.3 Point estimation (sanity check) and a downstream decision
Against the long-run proxy: Stein loss sample cov. 2.75, OAS 2.99, **LW NLS 1.93**, IW posterior mean 1.80, **SC-SI posterior mean 1.55** (Riemannian distance to the proxy: 1.85 / 2.69 / 1.79 / 1.75 / 1.53).

Eyes open vs closed from a single 4-s window, held-out subjects, split 1 (tangent-space logistic regression, Riemannian features):

| | accuracy | NLL | accuracy on the 50 % most confident windows |
|---|---|---|---|
| sample cov. → LR | 0.820 | 0.431 | 0.899 |
| OAS → LR | 0.781 | 0.465 | 0.904 |
| LW NLS → LR | 0.809 | 0.426 | 0.914 |
| SC-SI, LR averaged over posterior draws | 0.847 | 0.390 | 0.946 |

Paired subject-bootstrap vs the plug-in (SCM-trained LR, posterior-averaged), two independent subject splits:

| split | ΔNLL | Δaccuracy | Δaccuracy on the 50 % most confident windows |
|---|---|---|---|
| 1 (seed 0) | −0.041 [−0.077, −0.006] | +0.027 [+0.002, +0.048] | +0.047 [+0.012, +0.068] |
| 2 (seed 1) | −0.034 [−0.071, 0.000] | +0.018 [−0.005, +0.042] | −0.005 [−0.022, +0.022] |

Against the NLS plug-in (paired): split 1 ΔNLL −0.035 [−0.061, −0.012], Δaccuracy +0.038 [+0.022, +0.054]; split 2 ΔNLL −0.026 [−0.058, +0.008], Δaccuracy +0.023 [+0.001, +0.047].

So: a small, consistent NLL/accuracy gain, and **no reliable abstention benefit**. ECE went the opposite way on the two splits (0.051 → 0.062; 0.059 → 0.023).
I would present this as a sanity check that posterior averaging does no harm downstream, not as a headline. Figure `..._E3_decision.png` (split 1).

### 2.4 Robustness
* N = 43 instead of 28 (`results/eeg_w4_s0_k1.0_e1.0_*`): same ranking; next-window gain +26.2 nats for SC-SI vs +17.0 IW; split-half gain +11.4 vs +5.7.
* **Second subject split and training seed** (`results/eeg_w4_s1_k1.0_e0.65_*`; validation curve monotone up to k = 12 as before):
  split-half log-score gain SC-SI **+5.7 [+4.8, +6.7]** (split 1: +5.8), **LW NLS +2.8 [+2.0, +4.0]** (+2.8), IW **−3.0 [−6.2, −0.2]** (+0.1), OAS −14.0; paired SC-SI − NLS **+2.9 [+1.3, +4.2]**;
  split-half coverage of directional variances SC-SI 0.483/0.781/0.935, NLS 0.388/0.666/0.862; 80 % coverage for log det / log Oz / top share / log cond.: SC-SI 0.71 / 0.78 / 0.82 / 0.76, NLS 0.45 / 0.66 / 0.66 / 0.66, IW 0.36 / 0.65 / 0.82 / 0.57, sample cov. 0.36 / 0.67 / 0.71 / 0.54;
  Riemannian distance 80 % coverage: SC-SI 0.46, NLS 0.38, IW 0.38, sample cov. 0.22.
  Next-window log-score gain SC-SI +21.6 [+16.3, +30.3], **NLS +12.7 [+8.2, +20.4]**, IW +12.0 [+4.6, +22.8], long-run proxy +27.8; paired SC-SI − NLS **+8.9 [+7.4, +10.6]**. Posterior-mean Stein loss vs long-run proxy: SC-SI 1.73, NLS 2.14, IW 2.39, OAS 2.94, sample cov. 3.16.
  The IW baseline's showing is not stable across splits (−3.0 vs +0.1 nats in the split-half test); SC-SI's and NLS's are.
* The N = 43 run (first bullet) was not repeated with NLS. For the N = 28 runs, re-running with NLS added reproduced the pre-NLS numbers of the other methods: log-score gains identical, coverages within Monte-Carlo noise (third decimal). The pre-NLS result files are kept in `results/pre_nls/`.

## 3. Finance experiment

**Data and tasks.** Ken French 48 industry portfolios, daily value-weighted returns, 1926-07 to 2026-07, zero-mean
model. Task = (63-day window, basket of d = 12 industries drawn at random among those observed in the window).
Training ensemble: non-overlapping windows before the fold start, 16 random baskets each (M ≈ 4.7k–5.9k). Three
expanding folds (train < 1995 / 2005 / 2015; test 1995–2004 / 2005–2014 / 2015–2026), 8 fixed baskets per fold,
monthly forecasts (3008 cases), one SC-SI refit per fold. Block bootstrap over months.

**Effective N.** Calibrated on a validation fold that precedes all test folds (train < 1985, validate 1985–94) with the
IW proxy: forward-interval coverage improves monotonically as N shrinks and is best at the smallest value tried
(scale 0.3 → N = 19, N/d = 1.6). **This is not the sampling N.** A kurtosis-based estimate from the same data
(N_eff = 2N/(2+κ_ex)) gives 37–55. So N = 19 is mostly a forecast-inflation factor absorbing drift between the estimation
and forecast windows. Sec. 3.3 repeats everything at N = 40.

**Decision studied.** Number of factors K in `C_K = Σ_{k≤K} l_k u_kᵀu_k + s²(I − P_K)`. Rules: Gavish–Donoho hard
threshold on the sample spectrum, fixed K, and the **Bayes rule** K = argmin E[L_Stein(C_K(Ce), C) | Ce] over posterior draws.
Score: realized variance of the unconstrained minimum-variance portfolio over the next 21 days.

### 3.1 Results (month-block bootstrap, `results/fin_d12_nw63_s0_results.json`)

| estimator | ann. vol | mean log(var/var_OAS) [95 % CI] |
|---|---|---|
| sample cov. | 14.80 % | +0.093 [+0.073, +0.112] |
| OAS shrinkage | 14.34 % | 0 |
| LW nonlinear shrinkage | 14.45 % | +0.039 [+0.027, +0.052] |
| PCA, K = 1 | 15.86 % | +0.208 |
| PCA, K = 3 | 14.80 % | +0.094 |
| PCA, K = Gavish–Donoho | 14.98 % | +0.118 [+0.100, +0.136] |
| PCA, K = Bayes (IW posterior) | 15.03 % | +0.119 |
| PCA, K = Bayes (SC-SI posterior) | 14.79 % | +0.090 [+0.072, +0.108] |
| IW posterior mean | 14.40 % | +0.007 |
| **SC-SI posterior mean** | **14.13 %** | **−0.031 [−0.039, −0.023]** |
| SC-SI Stein-optimal | 14.28 % | −0.025 [−0.033, −0.018] |

Paired (differences of mean log realised variance, ≈ percent): SC-SI posterior mean vs OAS −3.1 [−3.9, −2.3] (by fold: −5.2, −2.2, −2.1, all significant; dot-com −5.9, GFC −5.3, COVID −1.4 log-points);
SC-SI Stein-optimal vs LW NLS **−6.5 [−8.0, −4.9]** (by fold −8.8, −2.9 [−6.7, +0.4], −7.5), SC-SI posterior mean vs LW NLS −7.0 [−8.2, −5.8]; NLS beats the sample covariance (−5.4) but is worse than OAS (+3.9 [+2.7, +5.2]);
Bayes-K(SC-SI) vs Gavish–Donoho −2.8 [−3.6, −2.0]; Bayes-K(SC-SI) vs Bayes-K(IW) −2.9; Bayes-K(SC-SI) vs **fixed K = 3: −0.4 [−1.1, +0.4]**.
Forward predictive log-score vs sample cov.: SC-SI +3.7 [+2.6, +5.0], OAS +3.2 [+2.1, +4.6], LW NLS +2.2 [+1.5, +3.1], IW +1.0. SC-SI and OAS are not distinguishable.
Forward directional-variance coverage 50/80/95: SC-SI 0.523/0.822/0.944; IW 0.505/0.805/0.939; LW NLS 0.464/0.757/0.907; OAS 0.472/0.766/0.909; sample cov. 0.484/0.769/0.909.
Figures `figs/fin_d12_nw63_s0_F1_timeseries.png`, `..._F2_summary.png`.

### 3.2 What the finance experiment shows and does not show
* Supported (both N_eff values): the posterior-based **Stein-optimal covariance** beats OAS and the sample covariance out of sample, and using the SC-SI posterior to pick K beats both the Gavish–Donoho threshold and the same rule run on the parametric IW posterior.
* **Not supported as stated: "dynamic thresholding helps".** Whether the Bayes-K rule beats a fixed K = 3 flips with N_eff (Sec. 3.3), and its K is markedly less stable than the hard threshold. The posterior bands on the eigenvalue shares (Fig. F1, top) show why: K is genuinely uncertain, and any hard pick discards that.
* A better use of the posterior here is to keep it soft: report P(factor k carries > x % of variance | Ce), or plug the posterior directly into a decision (Stein-optimal covariance), rather than thresholding.
* The calibration of forward intervals looks good at N_eff = 19 (0.52/0.82/0.94) only because N_eff was tuned to make it so; at N_eff = 40 it is 0.375/0.645/0.831, i.e. the drift between the estimation and forecast windows is not modelled.

### 3.3 Robustness
* **N_eff = 40** (kurtosis-based value; `results/fin_d12_nw63_s0_n40_*`; baselines that use N, OAS and the IW fit, are recomputed at N = 40 too):

  | paired mean log realised variance | N_eff = 19 | N_eff = 40 |
  |---|---|---|
  | SC-SI Stein-optimal vs OAS | −0.025 [−0.033, −0.018] | −0.023 [−0.029, −0.017] |
  | SC-SI Stein-optimal vs sample cov. | −0.118 [−0.138, −0.098] | −0.126 [−0.138, −0.113] |
  | SC-SI posterior mean vs OAS | −0.031 [−0.039, −0.023] | −0.007 [−0.016, +0.002] |
  | SC-SI Stein-optimal vs LW NLS | −0.065 [−0.080, −0.049] | −0.039 [−0.046, −0.032] |
  | LW NLS vs OAS | +0.039 [+0.027, +0.052] | +0.016 [+0.011, +0.021] |
  | Bayes-K (SC-SI) vs Gavish–Donoho K | −0.028 [−0.036, −0.020] | −0.037 [−0.046, −0.028] |
  | Bayes-K (SC-SI) vs Bayes-K (IW) | −0.029 [−0.040, −0.018] | −0.012 [−0.019, −0.005] |
  | Bayes-K (SC-SI) vs fixed K = 3 | −0.004 [−0.011, +0.004] | −0.029 [−0.037, −0.021] |
  | mean K, Gavish–Donoho / Bayes (SC-SI) | 1.54 / 2.19 | 2.14 / 5.20 |
  | mean abs. monthly change in K, GD / Bayes | 0.22 / 0.52 | 0.36 / 1.41 |

  Stein-optimal vs NLS is significantly negative in all three folds at N_eff = 40 (−0.040, −0.037, −0.040) and in two of three at N_eff = 19 (fold 1: −0.029 [−0.067, +0.004]).
  Stein-optimal vs OAS is negative in all three folds at both N_eff; the one non-significant case is N_eff = 19, fold 1 (test 2005–2014): −0.006 [−0.018, +0.006].
  Forward log-score gain vs sample cov. at N_eff = 40 (forward N = 13): SC-SI +13.2 [+11.4, +15.4], OAS +10.3 [+8.5, +12.7], LW NLS +10.1 [+8.5, +12.0], IW +8.6 [+7.1, +10.5]; the scores at the two N_eff are not comparable across columns because the forward N differs.

## 4. Caveats that apply to both experiments
1. **The Wishart channel is an approximation for time series.** N is estimated, not known; the paper's theory covers the exact model only. Calibration (Protocol A) is the check, and it passes for spectral functionals but not for the Riemannian distance.
2. **Tasks are dependent** (windows of one subject; baskets from one month). Subject-level or month-block bootstraps are used; the identifiability/EM guarantees assume independent tasks.
3. **Only general Algorithm 1, d ≤ 12.** Algorithm 2 (spectral + orientation MCMC) was not implemented, so d = 48 industries or 64 EEG channels are out of reach here.
4. **Finite M.** EM over-sharpens (Sec. 1); checkpoint selection on validation data is required.
5. **Baseline strength.** NLS is an asymptotic large-d estimator (Sec. 1); at d = 8–12 it is a moderate baseline. It targets Frobenius loss, whereas minimum-variance portfolios depend on the *inverse*: the natural finance competitor is Ledoit–Wolf's quadratic-inverse shrinkage (QIS, 2022), which I did not implement. Factor-model and dynamic (DCC-type) covariance forecasters were not run either, so the finance conclusions are relative to shrinkage baselines only.
6. 29 test subjects, one dataset per domain. EEG has two subject splits/training seeds; finance has one seed at two N_eff values.

## 5. Other applications that fit the submission
Selection criteria: many related tasks, modest N with N ≥ d, near-Gaussian data, a known forward simulator, and a way to validate on future/held-out data.

1. **Motor-imagery BCI with real sessions** (BCI Competition IV 2a / MOABB `BNCI2014_001`: 9 subjects × 2 sessions × 288 trials, 22 channels). Same pipeline as here but with genuine session structure: learn the prior on session 1 and validate on session 2. Directly matches the "multiple sessions/patients/trials" idea, and covariance-based classifiers (Riemannian) are the standard downstream.
2. **fMRI functional connectivity, test–retest** (Midnight Scan Club, 10 sessions per subject, OpenNeuro; or HCP if you have access). Second session is genuine held-out data for eq. 169; the paper already cites PoSCE and covariate-assisted covariance as comparisons; subject covariates (age, site) exercise Appendix I.
3. **Ensemble data assimilation** (Lorenz-96 first, then ERA5 ensemble). Forecast-error covariances from N = 20–50 members in local patches are precisely a Wishart channel with a known simulator, and sampling-error covariance is a central problem there. Downstream: Kalman gain with covariance uncertainty.
4. **Cosmology covariances from mock catalogues** (Quijote: 15 000 fiducial simulations). Uniquely, this gives a near-exact ground-truth covariance to subsample from (n = 50–500 mocks), i.e. the only real-data setting where posterior accuracy can be checked against the truth, and the cosmological parameters are natural covariates. Downstream: propagating covariance uncertainty into parameter constraints.
5. **Neural noise correlations** (Allen Brain Observatory / Steinmetz Neuropixels): tasks = session × stimulus, d = 10–30 neurons, N = 20–100 trials, stimulus as covariate. Needs a variance-stabilizing transform for counts.

Finance is better placed on **intraday realized covariances** (one task per day, N = 78 five-minute returns), where Gaussianity and stationarity within the task are much more plausible than for daily returns; that data is not freely downloadable here.

## 6. Controlled toy-model validation, densified
Complements Sec. 1: full marginal distributions of six covariance statistics (log det C, log condition number,
top-eigenvalue share, log directional variance along the empirical top/bottom eigenvector, one correlation),
not just PIT/coverage summaries, on four toys (`toys.py`, `toy_train.py`, `toy_eval.py`, `toy_traj.py`, `toy_figs.py`).
128 held-out tasks per toy; SC-SI and a fitted-by-marginal-likelihood IW baseline are each compared against a
*reference* posterior with J = 2048 draws.

| toy | d, N | reference | SC-SI checkpoint |
|---|---|---|---|
| `iw_ri_d4` | 4, 8 | exact IW posterior | k=16 of 24 |
| `iw_ri_d8` | 8, 40 | exact IW posterior | k=10 of 10 |
| `iw_nonri_d8` | 8, 28 | exact IW posterior | k=10 of 12 |
| `factor_d8` | 8, 28 (2-factor + idiosyncratic; no conjugate posterior) | batched HMC, 8 chains × 256 draws, R-hat ≤ 1.02 on all 6 statistics, all 128 tasks | k=24 of 24 |

Wasserstein-1 distance to the reference, normalised by the reference's own sd (mean over the 128 tasks; a
reference-vs-itself split-half floor is about 0.05–0.06 everywhere from finite-J noise):

| statistic | IW toys, SC-SI | IW toys, IW-fit (floor) | Factor toy, SC-SI | Factor toy, IW-fit (misspecified) |
|---|---|---|---|---|
| log det C | 0.23–0.25 | ~0.05 | 0.28 | 0.30 |
| log condition number | 0.23–0.55 | ~0.05 | 0.58 | 1.01 |
| top-eigenvalue share | 0.18–0.41 | ~0.05 | 0.26 | 0.63 |
| log dir. variance (top emp. eigvec) | 0.16–0.45 | ~0.05 | 0.32 | 0.87 |
| log dir. variance (bottom emp. eigvec) | 0.35–0.38 | ~0.05 | 0.85 | 1.15 |
| correlation (entry 0,1) | 0.12–0.23 | ~0.05 | 0.37 | 0.45 |

Takeaways:
* On the three IW toys the correctly specified fitted-IW baseline sits at the sampling floor everywhere, as it
  should: it estimates two hyperparameters from thousands of tasks under the exactly right family. SC-SI is a
  nonparametric fit and pays for it, most on log condition number and the bottom-eigenvector directional
  variance — the low-eigenvalue end of the spectrum, the hardest part of any covariance to pin down (Ledoit–Péché).
* On the factor toy (no conjugate posterior; a genuinely different covariance class from the IW family) the
  ranking flips: SC-SI tracks the HMC reference much more closely than the misspecified IW-fit, worst on the same
  low-eigenvalue statistics (log condition number, bottom directional variance) but by a smaller relative margin
  than the IW-fit's misspecification error there.
* Simulation-based-calibration (SBC) rank histograms of the *true* covariance's statistic under each posterior
  (`figs/toys_T3_sbc.png`) and pooled PIT of learner draws against the reference (`figs/toys_T2_pit.png`) tell the
  same story at the distributional level, not just via a single scalar distance.
* `figs/toys_T5_trajectory.png` shows accuracy against the self-consistent EM iteration next to the validation
  log-score used for checkpoint selection: on the d=8 IW toys, W1 to the reference bottoms out around the selected
  checkpoint and *worsens* with more iterations (the over-sharpening effect of Sec. 1), which is exactly why
  checkpoint selection uses held-out validation data rather than the last iteration.

Figures: `figs/toys_T1_posterior_marginals.png` (per-statistic histograms at one fixed test task, all four toys),
`toys_T2_pit.png`, `toys_T3_sbc.png`, `toys_T4_w1_heatmap.png`, `toys_T5_trajectory.png`.

## 7. Reproduce
All commands run from this directory (`experiments/`); see the top-level `README.md` for setup and data download.

```bash
# --- data (once) ---
./data/download_eeg.sh                                 # ~450MB, PhysioNet eegmmidb (or pass a subject count for a smoke test)
python data/prepare_finance.py                          # Ken French 48-industry daily returns
python eeg_data.py 4                                     # windowed + split-half EEG covariances (4s windows)

# --- Sec. 1: validation against the exact IW posterior ---
python bench_iw.py --d 8 --Nfac 5 --M 2000 --kind ri --init deconv --inflate 1.5 \
    --steps_first 4000 --steps_outer 800 --n_outer 20 --tag _v2
python bench_figs.py

# --- Sec. 2: EEG (~25 min train + a few min eval, per command) ---
python eeg_exp.py --win 4 --n_outer 12 --neff_scale 0.65      # Protocol B, classification, checkpoints
python eeg_exp2.py eeg_w4_s0_k1.0_e0.65 12 0.65                # Protocol A (split-half)
python eeg_figs.py eeg_w4_s0_k1.0_e0.65
python test_nls.py                                             # sanity tests of the Ledoit-Wolf NLS baseline

# --- Sec. 3: finance (~45 min per neff_scale) ---
python fin_exp.py --d 12 --n_outer 12                          # N_eff calibrated on a validation fold
python fin_exp.py --d 12 --n_outer 12 --neff_scale 0.63 --tag _n40
python fin_figs.py; python fin_figs.py fin_d12_nw63_s0_n40

# --- controlled toy models (Sec. 6): IW (RI/non-RI) and a factor model ---
python toy_train.py --toy iw_ri_d4 --n_outer 24            # also: iw_ri_d8, iw_nonri_d8, factor_d8
python toy_eval.py --toy iw_ri_d4                           # exact IW reference; HMC reference for factor_d8
python toy_traj.py iw_ri_d4 iw_ri_d8 iw_nonri_d8 factor_d8  # accuracy vs EM iteration (needs all --toy checkpoints)
python toy_figs.py                                          # histograms, PIT, SBC, W1 heatmap, trajectories
```
Files: `scsi.py` (method), `covutils.py` (baselines/metrics), `toys.py` (controlled models + HMC reference),
`bench_iw.py`/`bench_figs.py`, `eeg_*.py`, `fin_*.py`, `toy_*.py`, `plotstyle.py`, `results/`, `figs/`.
