"""d=8, N=40, spectral-encoding SC-SI runs trained with the Foellmer bridge sampler (tag="spectral_follmer")
-- per-checkpoint median-per-statistic mean-W1 production driver (toy_W1_mean.w1_mean_dict). Replaces the
earlier ODE-sampler "spectral" tag campaign (deleted). No single shared progress file anymore (see
toy_W1_mean.py's module docstring): results land in {toy}_W1_mean.json (one per toy, at the results/toys/
root like {toy}_logscore.json -- not per-encoding) and in {toy}_spectral_follmer_train.json's "w1_curve"."""
from toy_W1_mean import w1_mean_dict

TOYS = ["iw_ri_d8_inf", "iso_diag_d8_inf"]

if __name__ == "__main__":
    w1_mean_dict(TOYS, "spectral_follmer")
    print("\nALL SPECTRAL_FOLLMER W1-MEAN DONE", flush=True)
