"""Log-score + W1 convergence-curve figure and a plain-text summary table for the two actively-maintained
spectral_follmer toys (iw_ri_d8_inf, iso_diag_d8_inf) -- reads the per-toy/per-tag files produced by
log_score_oracle.py/log_score_scsi.py ({toy}_logscore.json, {toy}_{tag}_train.json's val_curve) and
toy_W1_mean.py ({toy}_W1_mean.json, {toy}_{tag}_train.json's w1_curve). Replaces the deleted
toy_figs_common.py T5 block (which read the deleted results/toys/trajectory.json) for this campaign,
rebuilt against the current per-toy/per-tag schema instead. Run as the final step of
train_iw_ri_d8.sh/train_iso_diag_d8.sh/train_all_spectra.sh, after training + log-score + W1 eval."""
import json, os
import matplotlib.pyplot as plt
from plotstyle import *

setup()
RESULTS = "results/toys"
PNG = "figs"
TXT = f"{PNG}/table_spectral_follmer.txt"
os.makedirs(PNG, exist_ok=True)

TAG = "spectral_follmer"
TOYS = [("iw_ri_d8_inf", "IW, RI"), ("iso_diag_d8_inf", "Iso-diagonal")]


def load(toy):
    tr = json.load(open(f"{RESULTS}/{toy}_{TAG}_train.json"))
    o = json.load(open(f"{RESULTS}/{toy}_logscore.json"))
    w = json.load(open(f"{RESULTS}/{toy}_W1_mean.json"))
    return tr, o, w


def table_block(toy, title, tr, o, w):
    k = tr["best_k"]
    orc, iw_ls, scsi_ls = o["oracle"]["value"], o["iw_ml"]["value"], o["scsi"]["value"]
    floor, iw_w1, scsi_w1 = w["floor"], w["iw_ml_w1"], w["scsi"]["value"]
    return (f"=== {TAG}: {toy} ({title}, N={tr['N']}, d={tr['d']}) ===\n"
            f"  best_k = {k}\n"
            f"  log-score   oracle={orc:9.4f}   iw_ml={iw_ls:9.4f}   scsi={scsi_ls:9.4f}   gap={scsi_ls - orc:+8.4f}\n"
            f"  W1          floor={floor:8.4f}   iw_ml={iw_w1:8.4f}   scsi={scsi_w1:8.4f}\n")


def fig_and_table_spectral_follmer():
    fig, axs = plt.subplots(2, len(TOYS), figsize=(3.6 * len(TOYS), 5.4), squeeze=False)
    blocks = []
    for j, (toy, title) in enumerate(TOYS):
        tr, o, w = load(toy)
        best_k = tr["best_k"]
        blocks.append(table_block(toy, title, tr, o, w))

        # row 0: validation log-score curve, as a gap to the oracle (0 = matches oracle exactly)
        ls = {int(k): v for k, v in tr["val_curve"].items()}; ks = sorted(ls)
        orc = o["oracle"]["value"]; iw_gap = o["iw_ml"]["value"] - orc
        gap = [ls[k] - orc for k in ks]
        a0 = axs[0, j]
        a0.axhline(0, color=INK, lw=1.3)
        a0.axhline(iw_gap, color=ORANGE, lw=1.4, ls=(0, (4, 2)))
        a0.plot(ks, gap, "-o", color=BLUE, ms=2.6)
        a0.plot([best_k], [ls[best_k] - orc], "o", ms=7, mfc="none", mec=INK, mew=1.2)
        a0.set_title(f"{title}\n({toy})", fontsize=9.5)
        if j == 0:
            a0.set_ylabel("validation log-score\n(nats/task, vs oracle)")

        # row 1: mean W1 vs the reference posterior
        wc = tr["w1_curve"]
        w1 = {int(k): v for k, v in wc["checkpoints"].items()}; wks = sorted(w1)
        a1 = axs[1, j]
        a1.axhline(w["iw_ml_w1"], color=ORANGE, lw=1.4, ls=(0, (4, 2)))
        a1.axhline(w["floor"], color=INK, lw=1.1, ls=(0, (2, 1.5)))
        a1.plot(wks, [w1[k] for k in wks], "-o", color=BLUE, ms=2.6)
        a1.plot([best_k], [w1[best_k]], "o", ms=7, mfc="none", mec=INK, mew=1.2)
        a1.set_xlabel("EM iteration $k$")
        if j == 0:
            a1.set_ylabel("mean W1\n(vs reference posterior)")

    h = [plt.Line2D([], [], color=BLUE, marker="o", ms=3.5), plt.Line2D([], [], color=ORANGE, ls=(0, (4, 2))),
         plt.Line2D([], [], color=INK, ls=(0, (2, 1.5))), plt.Line2D([], [], color=INK, marker="o", mfc="none", mew=1.2, ms=6.5, ls="none")]
    fig.legend(h, ["SC-SI", "IW conjugate (ML-fitted)", "sampling-noise floor (W1 only)", "selected checkpoint (best_k)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05), columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{PNG}/fig_spectral_follmer.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{PNG}/fig_spectral_follmer.png")

    with open(TXT, "w") as f:
        f.write("\n".join(blocks))
    print("wrote", TXT)


if __name__ == "__main__":
    fig_and_table_spectral_follmer()
