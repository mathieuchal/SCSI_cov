import json
import numpy as np, matplotlib.pyplot as plt
from plotstyle import *
setup()
runs = {"d=4, N=8 (N/d=2), M=5000": "results/iw_ri_d4_N8_M5000_v2.json", "d=8, N=40 (N/d=5), M=2000": "results/iw_ri_d8_N40_M2000_v2.json"}
cols = {"d=4, N=8 (N/d=2), M=5000": ORANGE, "d=8, N=40 (N/d=5), M=2000": BLUE}
fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
for n, f in runs.items():
    r = json.load(open(f)); bi = r["by_iter"]; ks = sorted(int(k) for k in bi)
    c = cols[n]
    ax[0].plot(ks, [bi[str(k)]["pit_cov80"] for k in ks], "-o", color=c, ms=4, label=n)
    ax[1].plot(ks, [bi[str(k)]["offdiag_2nd_moment_ratio"] for k in ks], "-o", color=c, ms=4, label=n)
    ax[2].plot(ks, [bi[str(k)]["e_M"] / bi[str(k)]["e_M_SCM"] for k in ks], "-o", color=c, ms=4, label=n)
ax[0].axhline(0.8, color=INK3, ls=(0, (4, 3)), lw=1.2); ax[0].set_ylim(0.65, 0.9)
ax[0].set_title("80% interval coverage vs exact posterior"); ax[0].set_ylabel("coverage of exact IW posterior (PIT)")
ax[1].axhline(1.0, color=INK3, ls=(0, (4, 3)), lw=1.2); ax[1].set_ylim(0.7, 1.35)
ax[1].set_title("Orientation second-moment ratio (eq. 143)"); ax[1].set_ylabel("sampled / exact, target 1")
ax[2].set_title("Posterior-mean error, relative to sample cov."); ax[2].set_ylabel("||M_hat - M*|| / ||SCM - M*||")
for a in ax: a.set_xlabel("outer self-consistent iteration k")
ax[0].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("figs/bench_iw_convergence.png"); plt.close(fig)
print("ok")
