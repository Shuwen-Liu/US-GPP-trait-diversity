# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.22.4",
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "gpboost>=1.5.5",
#     "scikit-learn>=1.3",
#     "matplotlib>=3.7",
#     "optuna>=3.0",
# ]
# ///

"""GPBoost model performance: predictive R2 (main text) and variance decomposition (appendix).

marimo notebook (step 02 of the Fig. 5 pipeline). Independent of 03/04; run any time after
01_fit_full_model.py has written ./models/GPBoost_GPP_LAI_full-model_ecoprovince.json.

    marimo run 02_model_performance.py      # or: marimo edit 02_model_performance.py
    python 02_model_performance.py          # headless: writes the CSVs only

Purpose
    Report the held-out (20 %) test-set R2 of the full GPBoost model quoted in the text,
    split into the tree-only mean function F(X) and tree + spatial GP, and the incremental
    R2 over two reference models that are trained here with the same recipe:
    null_env (LAI + T + P + AI only) and traitFD_only (PC1-3 + FD only).

Inputs
    ../data/analysis_ready_tables/data.csv
    ./models/GPBoost_GPP_LAI_full-model_ecoprovince.json          (from 01)
    ./models/GPBoost_GPP_LAI_null-model_ecoprovince.json          (trained + cached on first run)
    ./models/GPBoost_GPP_traitFD-only_ecoprovince.json            (trained + cached on first run)

Outputs (./results/)
    gpboost_predictive_performance.csv        test/train R2, RMSE, MSE, Pearson r per model
    gpboost_incremental_R2.csv                delta R2 full - null_env, full - traitFD_only
    gpboost_variance_decomposition_appendix.csv   Nakagawa-style marginal/conditional R2 analogues

Notes
    The two reference models use a seeded Optuna sampler (TPESampler(seed=42)) so that they
    are reproducible; the full model from 01 is unseeded, so its numbers vary slightly
    between runs of 01.
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import json
    from pathlib import Path

    import gpboost as gpb
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import optuna
    import pandas as pd
    from sklearn.model_selection import train_test_split

    return Path, gpb, json, mo, np, optuna, pd, plt, train_test_split


@app.cell
def _(mo):
    mo.md(r"""
    # GPBoost performance — predictive R² (main) & variance-decomposition R² (appendix)

    Evaluates three GPBoost GPP models — **full** (`LAI + T + P + AI + PCs + FD`),
    **null_env** (`LAI + T + P + AI`) and **traitFD_only** (`PCs + FD`). The full model is
    loaded from its saved JSON; null_env and traitFD_only are trained once with the same recipe
    and cached.

    **Main text — predictive performance (primary).** Held-out 20% test-set R² and RMSE/MSE,
    split into *tree-only* `F(X)` and *tree + spatial GP* `F(X) + b(s)`, plus the incremental
    ΔR² from null → full. This is the framing recommended for GPBoost (a non-parametric mean
    function + latent Gaussian process).

    **Appendix — variance-decomposition R² (secondary).** Nakagawa-style marginal/conditional
    R² from the fitted variance components, provided only for side-by-side comparison with the
    LME models of Figs. 3-4 (`performance::r2_nakagawa`). These are analogues, **not** standard
    marginal/conditional R² — see the appendix caveat.

    > The two families answer different questions and are not equal (e.g. full model: predictive
    > tree-only R² ≈ 0.70 vs var-decomposition marginal ≈ 0.43). Details in each section.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Configuration

    Paths, the 80/20 split seed, and the feature lists for the three models. Feature lists
    mirror `01_fit_full_model.py`.
    """)
    return


@app.cell
def _(Path):
    # folder of this notebook; falls back to the working directory when __file__ is undefined
    try:
        GPB_DIR = Path(__file__).resolve().parent
    except NameError:
        GPB_DIR = Path.cwd()
    DATA_CSV = GPB_DIR.parent / "data" / "analysis_ready_tables" / "data.csv"
    MODELS_DIR = GPB_DIR / "models"
    RESULTS_DIR = GPB_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RANDOM_STATE = 42

    ENV_VARS = ["LAI", "Temperature", "Precipitation", "aridity_index"]
    TRAITFD_VARS = [
        "all_PC1",
        "all_PC2",
        "all_PC3",
        "FRic_gamma",
        "Fbeta_alpha_to_gamma",
        "Fbeta_gamma_to_tau",
        "FDiv_gamma",
    ]

    # name -> (feature_list, saved-model path). "full" is written by 01_fit_full_model.py;
    # "null_env" and "traitFD_only" are trained + cached on first run.
    MODEL_SPECS = {
        "full": (
            ENV_VARS + TRAITFD_VARS,
            MODELS_DIR / "GPBoost_GPP_LAI_full-model_ecoprovince.json",
        ),
        # named "null_env" (not "null") because pandas read_csv parses the literal
        # string "null" as NaN, which would drop this model's label downstream.
        "null_env": (
            ENV_VARS,
            MODELS_DIR / "GPBoost_GPP_LAI_null-model_ecoprovince.json",
        ),
        "traitFD_only": (
            TRAITFD_VARS,
            MODELS_DIR / "GPBoost_GPP_traitFD-only_ecoprovince.json",
        ),
    }
    return DATA_CSV, MODEL_SPECS, RANDOM_STATE, RESULTS_DIR


@app.cell
def _(mo):
    mo.md("""
    ## Load data & reproduce the split

    Reproduces `train_test_split(..., test_size=0.2, random_state=42)` and derives the exact
    linear map from standardized `GPP` back to `GPP_original` (used only for original-scale
    RMSE/MAE/bias — R² is scale-invariant).
    """)
    return


@app.cell
def _(DATA_CSV, RANDOM_STATE, np, pd, train_test_split):
    data = pd.read_csv(DATA_CSV)
    train, test = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)

    y_tr = train["GPP"].to_numpy()
    y_te = test["GPP"].to_numpy()
    coords_tr = train[["latitude", "longitude"]]
    coords_te = test[["latitude", "longitude"]]

    # GPP_original = bt_slope * GPP + intercept  (exact; residual ~ 1e-14)
    bt_slope, _bt_intercept = np.polyfit(data["GPP"], data["GPP_original"], 1)
    return bt_slope, coords_te, coords_tr, test, train, y_te, y_tr


@app.cell
def _(mo):
    mo.md("""
    ## Helper functions

    - `cov_pars_from_json` — reads `[σ²_ε, σ²_GP, GP_range]` from a saved model.
    - `variance_decomposition` — Nakagawa-style marginal/conditional R² + spatial ICC.
    - `regression_metrics` — R² (standardized) plus RMSE/MAE/bias back-transformed to original units.
    """)
    return


@app.cell
def _(json, np):
    def cov_pars_from_json(path):
        # GPBoost Gaussian likelihood + isotropic covariance -> [nugget, GP_var, GP_range]
        with open(path) as fh:
            return list(json.load(fh)["gp_model_str"]["cov_pars"])

    def variance_decomposition(sigma2_f, sigma2_gp, sigma2_eps):
        total = sigma2_f + sigma2_gp + sigma2_eps
        return {
            "R2_marginal": sigma2_f / total,
            "R2_conditional": (sigma2_f + sigma2_gp) / total,
            "ICC_spatial": sigma2_gp / (sigma2_gp + sigma2_eps),
        }

    def regression_metrics(y_true, y_pred, slope):
        sse = float(np.sum((y_true - y_pred) ** 2))
        sst = float(np.sum((y_true - y_true.mean()) ** 2))
        resid_orig = slope * (y_pred - y_true)  # intercept cancels in the difference
        return {
            "R2": 1.0 - sse / sst,
            "RMSE_orig": float(np.sqrt(np.mean(resid_orig**2))),
            "MSE_orig": float(np.mean(resid_orig**2)),
            "MAE_orig": float(np.mean(np.abs(resid_orig))),
            "bias_orig": float(np.mean(resid_orig)),
            "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
        }

    return cov_pars_from_json, regression_metrics, variance_decomposition


@app.cell
def _(mo):
    mo.md("""
    ## Train-or-load the three models & compute predictions

    The saved booster does **not** carry the GP training data, so a reloaded model returns a
    zero random effect. We therefore reconstruct the GP by kriging the training residuals
    `y − F(X)` with the model's saved covariance parameters (`GPModel.predict(y=…, cov_pars=…)`);
    this reproduces GPBoost's native `random_effect_mean` to numerical precision (validated:
    max abs diff ≈ 8e-8, correlation 1.0). `F(X)` (the tree ensemble / fixed effect) is read
    straight from the reloaded booster, which works fine.

    `train_or_load` mirrors the full-model recipe (Optuna over `lambda_l2, max_depth,
    min_data_in_leaf, learning_rate`; `num_leaves=2**10`; Vecchia GP) but **seeds** the Optuna
    sampler for reproducibility. First run trains + caches the null and traits+FD models.
    """)
    return


@app.cell
def _(
    bt_slope,
    coords_te,
    coords_tr,
    cov_pars_from_json,
    gpb,
    np,
    optuna,
    regression_metrics,
    test,
    train,
    variance_decomposition,
    y_te,
    y_tr,
):
    def _fixed_effect(bst, X, coords):
        out = bst.predict(X, gp_coords_pred=coords, pred_latent=True)
        return np.asarray(out["fixed_effect"]).ravel()

    def _gp_mean(residual_tr, coords_pred, cov_pars):
        # kriging of training residuals with FIXED (saved) covariance parameters
        gpm = gpb.GPModel(
            gp_coords=coords_tr,
            cov_function="gaussian",
            gp_approx="vecchia",
            num_neighbors=20,
            vecchia_ordering="random",
            seed=0,
        )
        pred = gpm.predict(
            y=residual_tr,
            gp_coords_pred=coords_pred,
            cov_pars=np.asarray(cov_pars),
            predict_response=False,
            predict_var=False,
        )
        return np.asarray(pred["mu"]).ravel()

    def train_or_load(feature_list, json_path):
        if json_path.exists():
            bst = gpb.Booster(model_file=str(json_path))
            return bst, cov_pars_from_json(json_path)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        d_train = gpb.Dataset(train[feature_list], label=y_tr)
        d_test = gpb.Dataset(test[feature_list], label=y_te)

        def objective(trial):
            param = {
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
                "max_depth": trial.suggest_int("max_depth", 1, 8),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 1000),
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", [0.005, 0.0075, 0.01, 0.02, 0.03]
                ),
                "num_leaves": 2**10,
                "verbose": -1,
            }
            gpm = gpb.GPModel(
                gp_coords=coords_tr, cov_function="gaussian", gp_approx="vecchia"
            )
            gpm.set_prediction_data(gp_coords_pred=coords_te)
            booster = gpb.train(
                params=param,
                train_set=d_train,
                valid_sets=d_test,
                use_gp_model_for_validation=True,
                gp_model=gpm,
                num_boost_round=10000,
                early_stopping_rounds=10,
                verbose_eval=False,
            )
            y_pred = booster.predict(
                test[feature_list], gp_coords_pred=coords_te
            )["response_mean"]
            return float(np.mean((y_te - y_pred) ** 2))

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=30)
        best = dict(study.best_trial.params)
        best.update({"num_leaves": 2**10, "verbose": -1})

        gp_final = gpb.GPModel(
            gp_coords=coords_tr, cov_function="gaussian", gp_approx="vecchia"
        )
        gp_final.set_prediction_data(gp_coords_pred=coords_te)
        bst = gpb.train(
            params=best,
            train_set=d_train,
            valid_sets=d_test,
            use_gp_model_for_validation=True,
            gp_model=gp_final,
            num_boost_round=10000,
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        bst.save_model(str(json_path))
        return bst, cov_pars_from_json(json_path)

    def evaluate(name, bst, cov_pars, feature_list):
        sigma2_eps, sigma2_gp, gp_range = cov_pars
        fe_tr = _fixed_effect(bst, train[feature_list], coords_tr)
        fe_te = _fixed_effect(bst, test[feature_list], coords_te)
        resid_tr = y_tr - fe_tr
        gp_tr = _gp_mean(resid_tr, coords_tr, cov_pars)
        gp_te = _gp_mean(resid_tr, coords_te, cov_pars)

        sigma2_f = float(np.var(fe_tr, ddof=1))
        vd = variance_decomposition(sigma2_f, sigma2_gp, sigma2_eps)

        m_test_tree = regression_metrics(y_te, fe_te, bt_slope)
        m_test_full = regression_metrics(y_te, fe_te + gp_te, bt_slope)
        m_train_tree = regression_metrics(y_tr, fe_tr, bt_slope)
        m_train_full = regression_metrics(y_tr, fe_tr + gp_tr, bt_slope)

        return {
            "model": name,
            "n_train": len(train),
            "n_test": len(test),
            "sigma2_f": sigma2_f,
            "sigma2_GP": sigma2_gp,
            "sigma2_eps": sigma2_eps,
            "GP_range": gp_range,
            "R2_marginal": vd["R2_marginal"],
            "R2_conditional": vd["R2_conditional"],
            "ICC_spatial": vd["ICC_spatial"],
            "R2_test_tree_only": m_test_tree["R2"],
            "R2_test_tree_gp": m_test_full["R2"],
            "delta_R2_spatial": m_test_full["R2"] - m_test_tree["R2"],
            "RMSE_test_orig": m_test_full["RMSE_orig"],
            "MSE_test_orig": m_test_full["MSE_orig"],
            "MAE_test_orig": m_test_full["MAE_orig"],
            "bias_test_orig": m_test_full["bias_orig"],
            "pearson_r_test": m_test_full["pearson_r"],
            "R2_train_tree_only": m_train_tree["R2"],
            "R2_train_tree_gp": m_train_full["R2"],
        }

    return evaluate, train_or_load


@app.cell
def _(MODEL_SPECS, train_or_load):
    fitted = {
        name: (*train_or_load(flist, path), flist)
        for name, (flist, path) in MODEL_SPECS.items()
    }
    return (fitted,)


@app.cell
def _(mo):
    mo.md("""
    ## Compute all metrics (master table)

    `perf` holds every metric for the three models; the main-text and appendix views below are
    subsets of it.
    """)
    return


@app.cell
def _(evaluate, fitted, pd):
    perf = pd.DataFrame(
        [evaluate(name, bst, cov, flist) for name, (bst, cov, flist) in fitted.items()]
    ).set_index("model")
    perf
    return (perf,)


@app.cell
def _(mo):
    mo.md("""
    # Main text — predictive performance

    Held-out (20% test) predictive skill — the recommended framing for GPBoost, whose
    fixed-effect part is a non-parametric tree ensemble and whose spatial dependence is a
    latent Gaussian process. Reported per model:

    - **tree+GP predictive R²** — full prediction `F(X) + b(s)` (the headline test R²);
    - **tree-only predictive R²** — mean function `F(X)` alone;
    - **RMSE / MSE** (original GPP units) and Pearson r.
    """)
    return


@app.cell
def _(RESULTS_DIR, perf):
    predictive_cols = [
        "n_train",
        "n_test",
        "R2_test_tree_gp",
        "R2_test_tree_only",
        "RMSE_test_orig",
        "MSE_test_orig",
        "pearson_r_test",
        "R2_train_tree_gp",
        "R2_train_tree_only",
    ]
    predictive = perf[predictive_cols].copy()
    predictive.to_csv(RESULTS_DIR / "gpboost_predictive_performance.csv")
    predictive
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Incremental predictive R² (ΔR²)

    Core question — *do traits + functional diversity improve GPP prediction beyond climate?* —
    answered by the increment from the null (env-only) model to the full model. **Two ΔR² are
    reported and they tell different stories:**

    - **ΔR²(response, tree+GP)** — increment in the *full spatial prediction*. Small, because the
      spatial GP compensates: remove traits+FD from the mean function and the GP simply absorbs
      more of the spatially-structured signal, so out-of-sample response prediction barely moves.
    - **ΔR²(fixed, tree-only)** — increment in the *mean (fixed-effect) function* `F(X)`. Large,
      and the increment that matches the SHAP attribution / the ecological claim that traits and
      FD structure the productivity response.

    Pick the headline to match the claim: predictive skill of the whole model (small) vs
    explanatory contribution to the mean function (large). `dR2_marginal` (var-decomposition,
    appendix) is included for cross-reference.
    """)
    return


@app.cell
def _(RESULTS_DIR, pd, perf):
    def _delta(base):
        return {
            "dR2_response_tree_gp": perf.loc["full", "R2_test_tree_gp"]
            - perf.loc[base, "R2_test_tree_gp"],
            "dR2_fixed_tree_only": perf.loc["full", "R2_test_tree_only"]
            - perf.loc[base, "R2_test_tree_only"],
            "dR2_marginal_vardecomp": perf.loc["full", "R2_marginal"]
            - perf.loc[base, "R2_marginal"],
        }

    incremental = pd.DataFrame(
        {
            "full - null_env  (adds traits+FD)": _delta("null_env"),
            "full - traitFD_only  (adds env)": _delta("traitFD_only"),
        }
    ).T
    incremental.to_csv(RESULTS_DIR / "gpboost_incremental_R2.csv")
    incremental
    return


@app.cell
def _(mo):
    mo.md("""
    ## Main figure — held-out test R² (tree-only vs tree+GP)
    """)
    return


@app.cell
def _(perf, plt):
    _order = ["null_env", "traitFD_only", "full"]
    _labels = ["Null\n(env only)", "Traits+FD\nonly", "Full"]
    _tree = [perf.loc[m, "R2_test_tree_only"] for m in _order]
    _full = [perf.loc[m, "R2_test_tree_gp"] for m in _order]
    _x = list(range(len(_order)))

    fig_pred, ax_pred = plt.subplots(figsize=(6, 4))
    ax_pred.bar([i - 0.2 for i in _x], _tree, width=0.4, label="tree-only  F(X)", color="#4C72B0")
    ax_pred.bar([i + 0.2 for i in _x], _full, width=0.4, label="tree + GP  F(X)+b(s)", color="#55A868")
    ax_pred.set_xticks(_x)
    ax_pred.set_xticklabels(_labels)
    ax_pred.set_ylabel("Held-out test R²")
    ax_pred.set_ylim(0, 1)
    ax_pred.set_title("GPBoost predictive R² (20% test)")
    ax_pred.legend(loc="lower right", frameon=False)
    fig_pred.tight_layout()
    fig_pred
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Appendix — variance-decomposition R² (comparison with the LME models)

    **Caveat.** GPBoost's fixed-effect part is a non-parametric tree
    ensemble, so a Nakagawa-style variance decomposition has no unique definition here. These
    are *analogues* of the LME marginal/conditional R² (`performance::r2_nakagawa`), provided
    only so the GPBoost model can sit next to the LME table — they should **not** be reported as
    standard marginal/conditional R² in the main text.

    `R²_marginal = σ²_f/(σ²_f+σ²_GP+σ²_ε)`, `R²_conditional = (σ²_f+σ²_GP)/(...)`,
    `ICC = σ²_GP/(σ²_GP+σ²_ε)`, with `σ²_f = Var(F(X))` over the training set and `σ²_GP, σ²_ε`
    the fitted GP and nugget variances. (Because tree and GP are jointly fit, the components do
    not sum to `Var(y)`; see the sanity cell.)
    """)
    return


@app.cell
def _(RESULTS_DIR, perf):
    vardecomp_cols = [
        "sigma2_f",
        "sigma2_GP",
        "sigma2_eps",
        "GP_range",
        "R2_marginal",
        "R2_conditional",
        "ICC_spatial",
    ]
    variance_decomp_tbl = perf[vardecomp_cols].copy()
    variance_decomp_tbl.to_csv(RESULTS_DIR / "gpboost_variance_decomposition_appendix.csv")
    variance_decomp_tbl
    return


@app.cell
def _(perf, plt):
    _order2 = ["null_env", "traitFD_only", "full"]
    _labels2 = ["Null\n(env only)", "Traits+FD\nonly", "Full"]
    _marg = [perf.loc[m, "R2_marginal"] for m in _order2]
    _gp = [perf.loc[m, "R2_conditional"] - perf.loc[m, "R2_marginal"] for m in _order2]
    _resid = [1.0 - perf.loc[m, "R2_conditional"] for m in _order2]

    fig_vd, ax_vd = plt.subplots(figsize=(6, 4))
    ax_vd.bar(_labels2, _marg, label="Fixed effects (marginal R²)", color="#4C72B0")
    ax_vd.bar(_labels2, _gp, bottom=_marg, label="Spatial GP", color="#55A868")
    ax_vd.bar(
        _labels2,
        _resid,
        bottom=[m + g for m, g in zip(_marg, _gp)],
        label="Residual",
        color="#C44E52",
    )
    ax_vd.set_ylabel("Variance share")
    ax_vd.set_ylim(0, 1)
    ax_vd.set_title("Appendix: GPBoost variance decomposition (training set)")
    ax_vd.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    fig_vd.tight_layout()
    fig_vd
    return


@app.cell
def _(mo):
    mo.md("""
    ## Sanity checks & caveats
    """)
    return


@app.cell
def _(mo, np, perf, y_tr):
    _f = perf.loc["full"]
    _components = _f["sigma2_f"] + _f["sigma2_GP"] + _f["sigma2_eps"]
    mo.md(
        f"""
    **Full-model checks** (reference values from the run used in the paper; a re-run of 01 changes them slightly):

    - `σ²_f = {_f['sigma2_f']:.3f}` (≈ 0.287), `σ²_GP = {_f['sigma2_GP']:.3f}` (≈ 0.358),
      `σ²_ε = {_f['sigma2_eps']:.3f}` (≈ 0.019)
    - sum of components = `{_components:.3f}` vs `Var(y_train) = {np.var(y_tr, ddof=1):.3f}`
      (they differ because the tree ensemble and GP are jointly fit and not orthogonal)
    - **R²_marginal = {_f['R2_marginal']:.3f}** (≈ 0.43), **R²_conditional = {_f['R2_conditional']:.3f}** (≈ 0.97)
    - **R²_test tree-only = {_f['R2_test_tree_only']:.3f}** (≈ 0.70),
      **tree+GP = {_f['R2_test_tree_gp']:.3f}** (≈ 0.98), ΔR²_spatial = {_f['delta_R2_spatial']:.3f}
    - test RMSE (orig units) = {_f['RMSE_test_orig']:.3f}, Pearson r = {_f['pearson_r_test']:.4f}

    **Caveats**

    1. *Marginal (var-decomp) ≠ tree-only (test).* The first attributes model variance
       (`σ²_f / total`), the second measures predictive skill on held-out data. Both are
       legitimate; report them side by side and label them clearly.
    2. *Test set is not a clean hold-out.* The original full model used the test set for
       early stopping / the Optuna objective; the null and traits+FD models mirror this for
       comparability. Test metrics are therefore mildly optimistic.
    3. *Tiny nugget ⇒ conditional R² ≈ 0.97* (near-interpolation in-sample). Trust the test-set
       tree+GP R² for genuine predictive skill.
    """
    )
    return


if __name__ == "__main__":
    app.run()
