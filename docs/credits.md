# Licence and data

Code and documentation: GNU GPLv3 ([`LICENSE`](../LICENSE)).

**Figures.** Every data figure is drawn by [`writeup/figures/make_figures.py`](../writeup/figures/make_figures.py) from the tables in `writeup/data/` and carries the same licence as the code. The one
conceptual picture, the airport under `fig_taxi_schematic`, is an image-model illustration: its prompt, tool and licence note are in
[`writeup/figures/illustrations/PROMPTS.md`](../writeup/figures/illustrations/PROMPTS.md), and every caption that uses it says "illustration".

**Data.** EUROCONTROL Performance Review Unit (challenge data, ansperformance.eu aggregates, and OPDI with the OpenSky
Network); ADS-B data from adsb.lol (ODbL); © OpenStreetMap contributors
(ODbL); the METAR archive of the Iowa Environmental Mesonet; OurAirports.

## Methods borrowed

| method | where we use it | source |
|---|---|---|
| denoising autoencoder with swap noise, then a supervised head on its encoder | the DAE member of the blend (`overnight/dae.py`) | the recipe of the 1st-place solution of the Kaggle *Porto Seguro's Safe Driver Prediction* competition (M. Jahrer); denoising autoencoders: Vincent et al., ICML 2008 |
| stack of dilated 1D convolutions over a time-ordered chunk | an alternative trunk of the stream network (`overnight/stx.py`, `--arch conv`); tested and not in the final model ([what did not help](results.md#what-did-not-help-and-why)) | the 1D-CNN member of the 1st-place blend of the Kaggle *Optiver Realized Volatility Prediction* competition |
| gradient-boosted trees | the copy classifier, the regressors and the stackers | XGBoost (Chen and Guestrin, 2016), CatBoost (Prokhorenkova et al., 2018), LightGBM (Ke et al., 2017) |
| non-negative least squares blend weights, cross-fitted | the blend of the eleven regressors | standard stacking practice; the cross-fit is ours |

Everything else, the label mixture, the recording-regime rules, the post-steps and the validation design, is our own work on this data.
