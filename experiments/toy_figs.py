import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from plotstyle import *
from toy_eval import STATS, LABELS, pit_vs_ref

setup()
TOYS = [t for t in ("iw_ri_d4", "iw_ri_d8", "iw_nonri_d8", "factor_d8") if os.path.exists(f"results/toys/{t}_stats.npz")]
TITLE = {"iw_ri_d4": "IW, RI\nd=4, N=8", "iw_ri_d8": "IW, RI\nd=8, N=40", "iw_nonri_d8": "IW, non-RI\nd=8, N=28", "factor_d8": "Factor model\nd=8, N=28"}
SHORT = {"logdet": "log det C", "logcond": "log cond. number", "top_share": "top-eig. share",
         "ldv_top": "log dir. var.\n(top emp. eigvec)", "ldv_bot": "log dir. var.\n(bottom emp. eigvec)", "corr01": "correlation C01"}
D = {t: np.load(f"results/toys/{t}_stats.npz") for t in TOYS}
MET = {t: json.load(open(f"results/toys/{t}_metrics.json")) for t in TOYS}
OBS = int(sys.argv[1]) if len(sys.argv) > 1 else 0          # displayed test observation (index into the seeded test set; not selected)
nT, nS = len(TOYS), len(STATS)

# ------------------------------------------------------------------ T1: posterior marginals at one observation
fig, axs = plt.subplots(nT, nS, figsize=(2.55 * nS, 2.15 * nT), squeeze=False)
for i, t in enumerate(TOYS):
    for j, s in enumerate(STATS):
        ax = axs[i, j]
        ref, si, iw = D[t][f"ref__{s}"][OBS], D[t][f"scsi__{s}"][OBS], D[t][f"iwfit__{s}"][OBS]
        allv = np.concatenate([ref, si, iw]); lo, hi = np.percentile(allv, [0.3, 99.7])
        tv, sv = D[t][f"truth__{s}"][OBS], D[t][f"scm__{s}"][OBS]
        lo, hi = min(lo, tv), max(hi, tv); pad = 0.05 * (hi - lo); lo, hi = lo - pad, hi + pad
        bins = np.linspace(lo, hi, 36)
        ax.hist(ref, bins=bins, density=True, color=GRAY, alpha=0.40, lw=0, label="reference posterior")
        ax.hist(iw, bins=bins, density=True, histtype="step", color=ORANGE, lw=1.6, label="IW conjugate (ML-fitted)")
        ax.hist(si, bins=bins, density=True, histtype="step", color=BLUE, lw=1.9, label="SC-SI (ours)")
        ax.axvline(tv, color=INK, lw=1.4, ls=(0, (4, 2)), label="true value")
        ax.set_xlim(lo, hi)
        if lo <= sv <= hi:
            ax.axvline(sv, color=INK3, lw=1.4, ls=(0, (1, 1.5)), label="sample covariance (arrow: off-scale)")
        else:                                       # off-scale plug-in value: arrow on the nearest edge
            ax.plot([lo if sv < lo else hi], [0.5], marker="<" if sv < lo else ">", color=INK2, ms=8, transform=ax.get_xaxis_transform(), clip_on=False,
                    ls="none", label="_nolegend_")
        ax.set_yticks([]); ax.grid(axis="y", visible=False)
        if i == 0: ax.set_title(SHORT[s], fontsize=9)
        if j == 0: ax.set_ylabel(TITLE[t], fontsize=9, rotation=0, ha="right", va="center", labelpad=42)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("figs/toys_T1_posterior_marginals.png"); plt.close(fig)

# ------------------------------------------------------------------ T2: pooled PIT of learner draws under the reference marginals
fig, axs = plt.subplots(nT, nS, figsize=(2.55 * nS, 1.95 * nT), squeeze=False)
nb = 20
for i, t in enumerate(TOYS):
    for j, s in enumerate(STATS):
        ax = axs[i, j]
        ref = D[t][f"ref__{s}"]
        for nm, c, lw in (("iwfit", ORANGE, 1.6), ("scsi", BLUE, 1.9)):
            u = pit_vs_ref(D[t][f"{nm}__{s}"], ref).ravel()
            cnt, _ = np.histogram(u, bins=nb, range=(0, 1)); ax.stairs(cnt / cnt.sum() * nb, np.linspace(0, 1, nb + 1), color=c, lw=lw)
        ax.axhline(1.0, color=INK, lw=1.0, ls=(0, (4, 2)))
        ax.set_ylim(0, 2.2); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 1, 2])
        if i == 0: ax.set_title(SHORT[s], fontsize=9)
        if j == 0: ax.set_ylabel(TITLE[t], fontsize=9, rotation=0, ha="right", va="center", labelpad=42)
        if i == nT - 1: ax.set_xlabel("reference CDF of a draw", fontsize=8)
h = [plt.Line2D([], [], color=BLUE, lw=1.9), plt.Line2D([], [], color=ORANGE, lw=1.6), plt.Line2D([], [], color=INK, lw=1, ls=(0, (4, 2)))]
fig.legend(h, ["SC-SI (ours)", "IW conjugate (ML-fitted)", "uniform = identical marginal"], loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig("figs/toys_T2_pit.png"); plt.close(fig)

# ------------------------------------------------------------------ T3: SBC ranks of the true value
fig, axs = plt.subplots(nT, nS, figsize=(2.55 * nS, 1.95 * nT), squeeze=False)
nb = 8
for i, t in enumerate(TOYS):
    n_obs = D[t][f"truth__{STATS[0]}"].shape[0]
    mean, sd = n_obs / nb, np.sqrt(n_obs / nb * (1 - 1 / nb))
    for j, s in enumerate(STATS):
        ax = axs[i, j]
        ax.axhspan(mean - 1.96 * sd, mean + 1.96 * sd, color=INK3, alpha=0.18, lw=0)
        for nm, c, lw in (("ref", GRAY, 1.4), ("iwfit", ORANGE, 1.6), ("scsi", BLUE, 1.9)):
            r = (D[t][f"{nm}__{s}"] < D[t][f"truth__{s}"][:, None]).mean(1)
            cnt, _ = np.histogram(r, bins=nb, range=(0, 1)); ax.stairs(cnt, np.linspace(0, 1, nb + 1), color=c, lw=lw)
        ax.set_ylim(0, mean * 2.3); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1])
        if i == 0: ax.set_title(SHORT[s], fontsize=9)
        if j == 0: ax.set_ylabel(TITLE[t], fontsize=9, rotation=0, ha="right", va="center", labelpad=42)
        if i == nT - 1: ax.set_xlabel("rank of true value", fontsize=8)
h = [plt.Line2D([], [], color=BLUE, lw=1.9), plt.Line2D([], [], color=ORANGE, lw=1.6), plt.Line2D([], [], color=GRAY, lw=1.4),
     plt.Rectangle((0, 0), 1, 1, color=INK3, alpha=0.18)]
fig.legend(h, ["SC-SI (ours)", "IW conjugate (ML-fitted)", "reference posterior", "95% band for uniform ranks"], loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig("figs/toys_T3_sbc.png"); plt.close(fig)

# ------------------------------------------------------------------ T4: W1 heatmap
cmap = LinearSegmentedColormap.from_list("seq", ["#eef3fb", "#2a78d6", "#0d2b57"])
rows, vals, floors = [], [], []
for t in TOYS:
    for nm, lab in (("scsi", "SC-SI"), ("iwfit", "IW fit")):
        rows.append(f"{TITLE[t].replace(chr(10), ', ')} - {lab}")
        vals.append([MET[t]["metrics"][s][nm]["w1"] for s in STATS])
    floors.append([MET[t]["metrics"][s]["floor_w1"] for s in STATS])
vals = np.array(vals)
fig, ax = plt.subplots(figsize=(1.55 * nS + 3.8, 0.5 * len(rows) + 1.6))
im = ax.imshow(vals, cmap=cmap, vmin=0, vmax=max(0.35, np.nanpercentile(vals, 97)), aspect="auto")
ax.set_xticks(range(nS)); ax.set_xticklabels([SHORT[s].replace("\n", " ") for s in STATS], rotation=20, ha="right", fontsize=8.5)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8.5); ax.grid(False)
for a in range(vals.shape[0]):
    for b in range(vals.shape[1]):
        ax.text(b, a, f"{vals[a, b]:.2f}", ha="center", va="center", fontsize=8.5, color="white" if vals[a, b] > 0.5 * im.norm.vmax else INK)
for k in range(1, len(TOYS)):
    ax.axhline(2 * k - 0.5, color=SURFACE, lw=3)
ax.set_title("Wasserstein-1 distance to the reference posterior / reference sd (mean over 128 test tasks; sampling floor ~ 0.055)", fontsize=10)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("W1 / sd", fontsize=8.5)
fig.tight_layout(); fig.savefig("figs/toys_T4_w1_heatmap.png"); plt.close(fig)
print("toy figures written for", TOYS)

# ------------------------------------------------------------------ T5: accuracy along EM iterations vs the validation score
if os.path.exists("results/toys/trajectory.json"):
    TR = json.load(open("results/toys/trajectory.json"))
    names = [t for t in TOYS if t in TR]
    fig, axs = plt.subplots(2, len(names), figsize=(3.5 * len(names), 5.3), squeeze=False, sharex="col")
    for j, t in enumerate(names):
        r = TR[t]; ks = sorted(int(k) for k in r["w1"])
        a = axs[0, j]
        for s in STATS:
            a.plot(ks, [r["w1"][str(k)][s] for k in ks], color=INK3, lw=0.9, alpha=0.6)
        a.plot(ks, [np.mean(list(r["w1"][str(k)].values())) for k in ks], "-o", color=BLUE, lw=2, ms=4, label="mean over statistics")
        a.axhline(r["floor"], color=INK, lw=1.1, ls=(0, (4, 2)), label="sampling floor")
        a.axvline(r["best_k"], color=ORANGE, lw=1.4, label="selected by validation score")
        a.set_title(TITLE[t].replace("\n", ", "), fontsize=9.5); a.set_ylim(0, None)
        if j == 0: a.set_ylabel("W1 to reference / sd")
        b = axs[1, j]
        vk = sorted(int(k) for k in r["val_curve"]); vv = np.array([r["val_curve"][str(k)] if str(k) in r["val_curve"] else r["val_curve"][k] for k in vk])
        b.plot(vk, vv - vv.max(), "-o", color=BLUE, lw=1.8, ms=4); b.axvline(r["best_k"], color=ORANGE, lw=1.4)
        b.set_xlabel("outer iteration k")
        if j == 0: b.set_ylabel("validation log-score\n(nats / task, minus best)")
    axs[0, 0].legend(fontsize=7.4, loc="upper right")
    fig.tight_layout(); fig.savefig("figs/toys_T5_trajectory.png"); plt.close(fig)
    print("trajectory figure written for", names)
