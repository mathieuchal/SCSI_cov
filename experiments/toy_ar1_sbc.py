"""Simulation-based calibration of the AR(1) toy's quadrature posterior (toys.AR1Toy.ref_posterior).

For each of n prior draws C -> Ce, the rank of the true log(gamma) among 512 posterior draws must be uniform if the quadrature
posterior is exact. Reports the KS distance of the ranks to uniformity (95% critical value ~1.36/sqrt(n)), the mean rank, and the
median posterior s.d. of log(gamma); writes results/toys/ar1_sbc.json."""
import json
import numpy as np, torch

from scsi import wishart_channel
from toys import make_toy

n, J = 4000, 512
toy = make_toy("ar1_d8_M2000")
g = torch.Generator().manual_seed(0)
C = toy.sample_prior(n, g); Ce = wishart_channel(C, toy.N, g)
post = toy.ref_posterior(Ce, J, g)
lg = lambda X: torch.log(-torch.log(X[..., 0, 1]) / toy.dt)                       # C_01 = exp(-gamma dt)
rank = (lg(post) < lg(C)[:, None]).double().mean(1).numpy()
ks = float(np.max(np.abs(np.sort(rank) - np.arange(1, n + 1) / n)))
out = dict(n=n, J=J, ks=ks, ks_crit95=float(1.36 / np.sqrt(n)), mean_rank=float(rank.mean()), post_sd_logg_median=float(lg(post).std(1).median()), prior_sd_logg=toy.sigma)
json.dump(out, open("results/toys/ar1_sbc.json", "w"), indent=1); print(out)
