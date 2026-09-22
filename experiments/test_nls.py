"""Sanity tests for nls_shrink: (1) large-d asymptotic regime with known population spectrum,
(2) exact-Bayes benchmark: RI inverse-Wishart prior at the EEG-like (d=8, N=28) size."""
import math, torch, numpy as np
from covutils import nls_shrink, oas_shrink, sample_iw
from scsi import wishart_channel, DT
torch.manual_seed(0)
fro = lambda A, B: torch.linalg.norm(A - B, dim=(-1, -2))

print("(1) asymptotic regime: d=200, n=600 (c=1/3); Frobenius error ||C_hat - C||_F")
d, n = 200, 600
for name, ev in (("identity", np.ones(d)), ("3 levels {1,2,5}", np.repeat([1., 2., 5.], [d // 2, d // 4, d - d // 2 - d // 4])),
                 ("2 spikes (20,10) + bulk 1", np.r_[20., 10., np.ones(d - 2)])):
    C = torch.tensor(np.diag(ev), dtype=DT)
    errs = {"SCM": [], "OAS": [], "NLS": [], "oracle m_i": []}
    for _ in range(10):
        Ce = wishart_channel(C.expand(1, d, d), n)[0]
        l, U = torch.linalg.eigh(Ce)
        xi = torch.einsum("ji,jk,ki->i", U, C, U)                               # u_i' C u_i
        errs["SCM"].append(fro(Ce, C)); errs["OAS"].append(fro(oas_shrink(Ce, n), C))
        errs["NLS"].append(fro(nls_shrink(Ce, n), C)); errs["oracle m_i"].append(fro((U * xi) @ U.T, C))
    print(f"   {name:28s} " + "  ".join(f"{k}={float(torch.stack(v).mean()):.3f}" for k, v in errs.items()))

print("(2) RI inverse-Wishart prior, d=8, N=28 (Bayes-optimal Frobenius estimator = eq. 134, uses the prior)")
d, N, kap0 = 8, 28, 16.0
C = sample_iw(d + 1 + kap0, kap0 * torch.eye(d, dtype=DT), 4000)
Ce = wishart_channel(C, N)
est = {"SCM": Ce, "OAS": oas_shrink(Ce, N), "NLS (LW 2020)": nls_shrink(Ce, N),
       "Bayes, known prior (eq. 134)": N / (N + kap0) * Ce + kap0 / (N + kap0) * torch.eye(d, dtype=DT)}
for k, v in est.items():
    print(f"   {k:30s} Frobenius error = {float(fro(v, C).mean()):.4f}   min eig = {float(torch.linalg.eigvalsh(v).min(-1).values.min()):.3f}")
# ordering/positivity checks
print("   NLS positive definite in all tasks:", bool((torch.linalg.eigvalsh(est['NLS (LW 2020)']) > 0).all()))
