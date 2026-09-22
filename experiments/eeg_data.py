"""EEG Motor Movement/Imagery (PhysioNet eegmmidb) -> ensemble of short-window covariances.

Task unit  : (subject, condition, window).  Conditions: R01 = eyes open, R02 = eyes closed (60 s each).
Model      : zero-mean Gaussian.  We use 8 channels, common-average reference (over all 64), band-pass
             8-13 Hz (alpha), 4 s non-overlapping windows.  Successive samples of a band-limited signal
             are strongly correlated, so the Wishart degrees of freedom N is an *effective* sample size,
             N_eff = n / sum_k rho_k^2   (variance of a sample variance of a stationary Gaussian process),
             estimated from the data (see `estimate_neff`) and then validated by predictive calibration.
"""
import glob, os, sys, json
import numpy as np
import mne

mne.set_log_level("ERROR")
CH = ["Fz", "C3", "Cz", "C4", "Pz", "O1", "Oz", "O2"]
BAND = (8.0, 13.0)
WIN_S = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
TRIM_S = 2.0
NW = int(56 // WIN_S)  # windows per run after trimming 2 s at each end
RUNS = {"open": "R01", "closed": "R02"}


def load_run(path):
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.rename_channels(lambda n: n.strip(".").upper())
    fs = raw.info["sfreq"]
    X = raw.get_data() * 1e6                          # microvolts, (64, T)
    names = raw.ch_names
    X = X - X.mean(0, keepdims=True)                  # common average reference
    idx = [names.index(c.upper()) for c in CH]
    return X[idx], fs


def bandpass(X, fs):
    return mne.filter.filter_data(X, fs, BAND[0], BAND[1], l_trans_bandwidth=2.0, h_trans_bandwidth=2.0,
                                  fir_design="firwin", verbose="ERROR")


def neff_ratio(x, max_lag):
    """Effective-sample-size ratio for the variance of a stationary Gaussian series x (1-D):
    Var(sample variance) = (2/n) sum_k rho_k^2  ->  N_eff / n = 1 / (1 + 2 sum_{k>=1} rho_k^2)."""
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:max_lag + 1] / n
    rho = ac / ac[0]
    return 1.0 / (1.0 + 2.0 * np.sum(rho[1:] ** 2))


def main(data_dir="data/eeg", out=None):
    out = out or f"data/eeg_scm_{int(WIN_S)}s.npz"
    subj, scm_win, scm_full, ratios, fs_all, scm_il = [], [], [], [], [], []
    for s in range(1, 110):
        S = f"S{s:03d}"
        try:
            per_cond_w, per_cond_f, per_cond_il = [], [], []
            for cond, r in RUNS.items():
                X, fs = load_run(f"{data_dir}/{S}{r}.edf")
                if abs(fs - 160.0) > 1e-6 or X.shape[1] < 9000:
                    raise ValueError(f"fs={fs} n={X.shape[1]}")
                X = bandpass(X, fs)
                lo, hi = int(TRIM_S * fs), X.shape[1] - int(TRIM_S * fs)
                X = X[:, lo:hi]
                n = int(WIN_S * fs)
                nw = X.shape[1] // n
                Wd = np.stack([X[:, i * n:(i + 1) * n] @ X[:, i * n:(i + 1) * n].T / n for i in range(nw)])
                per_cond_w.append(Wd[:NW])
                per_cond_f.append(X @ X.T / X.shape[1])
                # interleaved split-half: 8-s epochs of eight 1-s chunks; A = even chunks, B = odd chunks
                ch = int(fs)
                ne = X.shape[1] // (8 * ch)
                il = []
                for e in range(ne):
                    E = X[:, e * 8 * ch:(e + 1) * 8 * ch].reshape(X.shape[0], 8, ch)
                    A_ = E[:, 0::2].reshape(X.shape[0], -1); B_ = E[:, 1::2].reshape(X.shape[0], -1)
                    il.append(np.stack([A_ @ A_.T / A_.shape[1], B_ @ B_.T / B_.shape[1]]))
                per_cond_il.append(np.stack(il[:7]))
                ratios += [neff_ratio(X[c], 80) for c in range(X.shape[0])]
            subj.append(s)
            scm_win.append(np.stack(per_cond_w))       # (2, 14, 8, 8)
            scm_full.append(np.stack(per_cond_f))      # (2, 8, 8)
            scm_il.append(np.stack(per_cond_il))       # (2, 7, 2, 8, 8)
            fs_all.append(fs)
        except Exception as e:  # incomplete download / anomalous recording
            print(f"skip {S}: {e}", flush=True)
    scm_win = np.stack(scm_win)                        # (S, 2, 14, d, d)
    scm_full = np.stack(scm_full)
    ratios = np.array(ratios)
    print("subjects kept:", len(subj), " median N_eff/n:", np.median(ratios),
          " quartiles:", np.percentile(ratios, [25, 75]))
    np.savez(out, scm_il=np.stack(scm_il), subj=np.array(subj), scm_win=scm_win, scm_full=scm_full, neff_ratio=ratios,
             n_win=int(WIN_S * 160), channels=np.array(CH), band=np.array(BAND))


if __name__ == "__main__":
    main()
