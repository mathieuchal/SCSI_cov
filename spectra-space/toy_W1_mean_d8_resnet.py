"""d=8, N=40, ResNet/MLP (flat-MLP "ResNet" convention, see toy_figs/toy_figs_d8_resnet.py) SC-SI runs --
per-checkpoint mean-W1 production driver, calling toy_W1_mean.w1_mean_dict for the 7-toy d8 campaign.
Results land in {toy}_W1_mean.json (one per toy) and each tag's own {toy}_{tag}_train.json "w1_curve" --
see toy_W1_mean.py's module docstring (resumable via a disposable per-(toy,tag) cache file).

Not all 7 toys use the same tag: iw_ri/iw_nonri/ar1/diag/factor only have a full per-checkpoint sweep
under the bare (untagged) stem (hidden=256, depth=4); hetero/iso_diag's bare sweep is a smaller,
superseded run -- their real "ResNet" campaign (hidden=512, depth=4, matching toy_figs_d8_resnet.py's
titles) is tagged "bignet". w1_mean_dict is called once per tag group; NOTE a toy appearing under more
than one tag across driver scripts (e.g. iw_ri_d8_inf here, untagged, vs. the actively-maintained
iw_ri_d8_inf_spectral_follmer campaign) shares the SAME {toy}_W1_mean.json -- its "scsi" entry reflects
whichever tag was merged there most recently, same "last tag written wins" convention as
{toy}_logscore.json's own "scsi" slot."""
from toy_W1_mean import w1_mean_dict

BARE_TOYS = ["iw_ri_d8_inf", "iw_nonri_d8_inf", "ar1_d8_inf", "diag_d8_inf", "factor_d8_inf"]
BIGNET_TOYS = ["hetero_d8_inf"]

if __name__ == "__main__":
    w1_mean_dict(BARE_TOYS, "")
    w1_mean_dict(BIGNET_TOYS, "bignet")
    print("\nALL TOYS DONE", flush=True)
