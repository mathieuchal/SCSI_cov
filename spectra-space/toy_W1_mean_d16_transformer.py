"""d=16, N=80, Transformer (tfeig) SC-SI runs -- per-checkpoint mean-W1 production driver, calling
toy_W1_mean.w1_mean_dict for the d16 campaign. Results land in {toy}_W1_mean.json (one per toy) and in
{toy}_tfeig_train.json's "w1_curve" -- see toy_W1_mean.py's module docstring (resumable via a disposable
per-toy cache file: safe to kill/restart).

iso_diag_d16_inf excluded: its cached reference posterior (results/toys/iso_diag_d16_inf_ref.pt) comes
from the HMC non-reproducibility investigation this session -- even the best 32-chain recompute still
has R-hat up to 8.4 on logcond/ldv_bot, so its W1-vs-reference numbers would not be trustworthy."""
from toy_W1_mean import w1_mean_dict

TAG = "tfeig"
TOYS = ["iw_ri_d16_inf", "iw_nonri_d16_inf", "factor_d16_inf", "ar1_d16_inf",
        "hetero_d16_inf", "diag_d16_inf"]

if __name__ == "__main__":
    w1_mean_dict(TOYS, TAG)
    print("\nALL TOYS DONE", flush=True)
