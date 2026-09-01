# Continental functional traits, functional diversity and productivity across the United States

Code to reproduce the main figures of

> Liu, S. et al. *Continental functional trait and functional diversity maps reveal
> aridity-mediated relationships with productivity.*

The paper maps nine foliar traits across the contiguous United States (CONUS) at 30 m,
summarises them as functional composition (the first three principal components of a
10-trait space, with canopy height as the tenth trait) and functional diversity (trait
probability density: richness, divergence, dissimilarity across scales), and relates both
to gross primary productivity (GPP) with linear mixed-effects models and with
Gaussian-process boosted regression trees.

## Data

All inputs come from the Zenodo archive **https://doi.org/10.5281/zenodo.22215583**:
the 30-m trait maps, the analysis-ready grid-cell tables, the 0.05-degree functional
composition, functional diversity, predictor and mask rasters, and the fitted PCA model.
Download the archive into `data/` and run `data/organize_zenodo_files.py` once
(see `data/README.md`). The functional diversity rasters are distributed as data; the
code that computed them (R package `TPD`, described in the Methods) is not part of this
release.

## Layout and run order

Each script is run from inside its own folder and writes to a `results/` (or `models/`)
sub-folder that it creates.

```text
01_functional_composition/   Fig. 2
  01_fit_pca.ipynb                 fits the 10-trait PCA on the sample points; prints the
                                   variance explained by PC1-PC3 (85.3 %)
  02_plot_trait_space_2d.ipynb     Fig. 2b, 2c
  03_plot_trait_space_3d.ipynb     Fig. 2a
  04_apply_pca_to_trait_maps.py    applies the PCA pixel-wise to the 30-m trait maps
  05_mosaic_pc_tiles.py            mosaics the tiles into one 30-m PC raster
                                   (Fig. 2d-g are drawn from the 0.05-degree PC rasters in the archive)

02_mixed_effects_models/     Fig. 3 and Fig. 4
  01-04_fit_*.R                    candidate models: composition, diversity, and their
                                   aridity interactions (nlme, spatial correlation, ecoprovince
                                   random intercept)
  05/06 and 07/08                  forward selection of the full model (R2m = 60.0 %) and of the
                                   aridity-interaction model (R2m = 71.7 %), and export of their results
  09_evaluate_models.Rmd           coefficients, delta AIC and marginal R2 of every model
  10_evaluate_pft_models.Rmd       the plant-functional-type model of Fig. 3f
  11_plot_fig3.py                  draws Fig. 3 (marimo notebook: `python 11_plot_fig3.py`)
  aridity_classes/                 the per-aridity-class models and Fig. 4 (run 01 -> 02 -> 03 -> 04)

03_gpboost_shap/             Fig. 5
  01_fit_full_model.py             GPBoost model with Bayesian hyper-parameter optimisation
  02_model_performance.py          R2 values quoted in the text (marimo notebook)
  03_shap_maps.ipynb               full-grid SHAP attribution; Fig. 5b, 5c, 5e, 5f rasters
  04_decision_plots.ipynb          Fig. 5d

functions/plot_style.py      shared figure styling
data/                        the Zenodo archive goes here
```

Fig. 2d-g and Fig. 5a-c, e-f are map layouts; the scripts write the rasters, and the
published panels were composed in a GIS from them.

## Environment

```bash
uv sync                        # Python, pinned in pyproject.toml / uv.lock
```

```r
install.packages(c("nlme", "dplyr", "foreach", "doParallel", "performance", "MuMIn", "knitr", "rmarkdown"))
```

Key Python packages: `scikit-learn`, `gpboost` 1.5.5, `shap`, `optuna`, `rasterio`,
`matplotlib`, `seaborn`. The mixed-effects models are R scripts and R Markdown documents.

Note: the Bayesian hyper-parameter search in `03_gpboost_shap/01_fit_full_model.py` is not
seeded, so refitted models differ slightly from run to run.

## License

MIT, see `LICENSE`.
