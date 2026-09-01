# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.22.4",
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "matplotlib>=3.7",
#     "seaborn>=0.12",
# ]
# ///
"""11_plot_fig3.py -- Fig. 3: model comparison of the linear mixed-effects models.

Purpose
    Combine the result tables of the baseline, functional-composition (PC1-3),
    functional-diversity, aridity-interaction, forward-selected full models and
    the PFT alternative into one vertically stacked figure: marginal R2 bars
    (left column) and coefficient forest plots (right column).

Inputs (all produced by the scripts in this folder)
    results/LME_GPP_LAI_T_P_AI_ecoprovince_SA_results.csv            (09_evaluate_models.Rmd)
    results/LME_GPP_LAI_T_P_AI_trait_ecoprovince_SA_results.csv      (09)
    results/LME_GPP_LAI_T_P_AI_FD_ecoprovince_SA_results.csv         (09)
    results/LME_GPP_LAI_T_P_AI_trait_AI_ecoprovince_SA_results.csv   (09)
    results/LME_GPP_LAI_T_P_AI_FD_AI_ecoprovince_SA_results.csv      (09)
    results/LME_GPP_LAI_T_P_AI_IGBP_ecoprovince_SA_results.csv       (10_evaluate_pft_models.Rmd)
    results/LME_GPP_LAI_T_P_AI_IGBP_AI_ecoprovince_SA_results.csv    (10)
    nlme_forward_selection_PC3_final_model_performance.csv           (06_export_full_model.R)
    nlme_AI_forward_selection_PC3_final_model_performance.csv        (08_export_interaction_model.R)

Output
    results/fig3_model_comparison.png

Run order
    Last step of this folder (after 01-10). This is a marimo notebook; run it
    either as a plain script (``python 11_plot_fig3.py``) or interactively
    (``marimo edit 11_plot_fig3.py``). Working directory: this folder.
    Plot styling comes from ``../functions/plot_style.py``.
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from pathlib import Path

    return Path, mo, np, pd


@app.cell
def _(Path):
    # Base directory = the folder containing this script (falls back to the
    # current working directory when __file__ is not defined, e.g. in some
    # interactive sessions)
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    return base_dir, results_dir


@app.cell
def _(mo):
    mo.md(r"""
    # LME Model Results — Combined Stacked Figure

    Merges the main-effect and aridity-interaction result figures into **one**
    vertically-stacked figure. Axes are swapped so bars/forests are horizontal
    and every model step stacks down a single ruler, making each model's
    marginal $R^2$ directly comparable to the Null model.

    Layout (top → bottom blocks): **Main effects** → **Aridity interaction** →
    **Alternative: PFT**. Left column = marginal-$R^2$ bars; right column =
    coefficient forest plots aligned to the same rows.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Load raw result CSVs
    """)
    return


@app.cell
def _(pd, results_dir):
    # Benchmark (null model): GPP ~ LAI + T + P + AI
    raw_benchmark = pd.read_csv(
        results_dir / "LME_GPP_LAI_T_P_AI_ecoprovince_SA_results.csv"
    ).set_index("model")

    # Functional composition (main effect): GPP ~ LAI + T + P + AI + trait
    raw_fc = pd.read_csv(
        results_dir / "LME_GPP_LAI_T_P_AI_trait_ecoprovince_SA_results.csv"
    ).set_index("trait")

    # Functional diversity (main effect): GPP ~ LAI + T + P + AI + FD
    raw_fd = pd.read_csv(
        results_dir / "LME_GPP_LAI_T_P_AI_FD_ecoprovince_SA_results.csv"
    ).set_index("FD")

    # Functional composition x AI interaction
    raw_fc_ai = pd.read_csv(
        results_dir / "LME_GPP_LAI_T_P_AI_trait_AI_ecoprovince_SA_results.csv"
    ).set_index("trait")

    # Functional diversity x AI interaction
    raw_fd_ai = pd.read_csv(
        results_dir / "LME_GPP_LAI_T_P_AI_FD_AI_ecoprovince_SA_results.csv"
    ).set_index("FD")

    # +PFT (IGBP) — main effect: GPP ~ LAI + T + P + AI + IGBP
    raw_pft = pd.read_csv(
        results_dir / "LME_GPP_LAI_T_P_AI_IGBP_ecoprovince_SA_results.csv"
    ).set_index("model")

    # +PFT (IGBP) x AI interaction
    raw_pft_ai = pd.read_csv(
        results_dir / "LME_GPP_LAI_T_P_AI_IGBP_AI_ecoprovince_SA_results.csv"
    ).set_index("model")
    return (
        raw_benchmark,
        raw_fc,
        raw_fc_ai,
        raw_fd,
        raw_fd_ai,
        raw_pft,
        raw_pft_ai,
    )


@app.cell
def _(base_dir, pd):
    # Full model (forward selection) performance & estimates
    raw_full_perf = pd.read_csv(
        base_dir / "nlme_forward_selection_PC3_final_model_performance.csv"
    ).set_index("indicator")

    # Full model with AI interaction
    raw_full_ai_perf = pd.read_csv(
        base_dir / "nlme_AI_forward_selection_PC3_final_model_performance.csv"
    ).set_index("indicator")
    return raw_full_ai_perf, raw_full_perf


@app.cell
def _(raw_pft, raw_pft_ai):
    # +PFT main effect (Null + IGBP)
    pft_R2m = float(raw_pft.loc["LAI_T_P_AI_IGBP", "R2_marginal"])
    pft_delta_AIC = float(raw_pft.loc["LAI_T_P_AI_IGBP", "delta_AIC_vs_null"])

    # +PFT x AI (Null + IGBP*AI)
    pft_ai_R2m = float(raw_pft_ai.loc["Null + IGBP*AI", "R2_marginal"])
    pft_ai_delta_AIC = float(raw_pft_ai.loc["Null + IGBP*AI", "delta_AIC_vs_null"])
    return pft_R2m, pft_ai_R2m, pft_delta_AIC


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Organize: Main Effect Results
    """)
    return


@app.cell
def _(np, pd, raw_benchmark, raw_fc, raw_fd, raw_full_perf):
    # --- Trait and FD name mappings ---
    trait_list = [
        "Carbon", "Cellulose", "ChlorophyllsArea", "EWT", "Lignin",
        "Nitrogen", "NSC", "Phenolics", "SLA", "canopy_height",
        "all_PC1", "all_PC2", "all_PC3",
    ]

    fd_list = [
        "FRic_alpha", "FRic_gamma", "FRic_tau",
        "FDiv_alpha", "FDiv_gamma", "FDiv_tau",
        "Fbeta_alpha_to_gamma", "Fbeta_gamma_to_tau",
    ]

    rename_map = {
        "canopy_height": "Canopy height",
        "ChlorophyllsArea": "Chlorophyll a + b",
        "all_PC1": "PC1",
        "all_PC2": "PC2",
        "all_PC3": "PC3",
        "FRic": "Functional richness",
        "FDiv": "Functional divergence",
        "Fbeta": "Functional dissimilarity",
    }

    # --- Benchmark ---
    benchmark_R2m = raw_benchmark.loc["LAI", "R2_marginal"]
    benchmark_R2c = raw_benchmark.loc["LAI", "R2_conditional"]

    # --- Functional Composition (main effect) ---
    rows_fc = []
    for t in trait_list:
        display_name = rename_map.get(t, t)
        rows_fc.append({
            "category": "Functional Composition",
            "variable": display_name,
            "coef": raw_fc.loc[t, "coef_trait"],
            "coef_CI_lower": raw_fc.loc[t, "coef_trait_CI_lower"],
            "coef_CI_upper": raw_fc.loc[t, "coef_trait_CI_upper"],
            "p_value": raw_fc.loc[t, "p_value_trait"],
            "delta_AIC": raw_fc.loc[t, "delta_AIC"],
            "R2_marginal": raw_fc.loc[t, "R2_marginal"],
            "R2_conditional": raw_fc.loc[t, "R2_conditional"],
            "partial_R2": raw_fc.loc[t, "partial_R2_trait"],
        })
    df_fc_main = pd.DataFrame(rows_fc)

    # --- Functional Diversity (main effect) ---
    rows_fd = []
    for fd in fd_list:
        parts = fd.split("_", 1)
        metrics_name = rename_map.get(parts[0], parts[0])
        scale = parts[1] if len(parts) > 1 else ""
        rows_fd.append({
            "category": "Functional Diversity",
            "variable": metrics_name,
            "scale": scale,
            "coef": raw_fd.loc[fd, "coef_FD"],
            "coef_CI_lower": raw_fd.loc[fd, "coef_FD_CI_lower"],
            "coef_CI_upper": raw_fd.loc[fd, "coef_FD_CI_upper"],
            "p_value": raw_fd.loc[fd, "p_value_FD"],
            "delta_AIC": raw_fd.loc[fd, "delta_AIC"],
            "R2_marginal": raw_fd.loc[fd, "R2_marginal"],
            "R2_conditional": raw_fd.loc[fd, "R2_conditional"],
            "partial_R2": raw_fd.loc[fd, "partial_R2_FD"],
        })
    df_fd_main = pd.DataFrame(rows_fd)

    # --- Full model ---
    full_R2m = float(raw_full_perf.loc["R2_marginal", "value"])
    full_R2c = float(raw_full_perf.loc["R2_conditional", "value"])
    full_partial_R2 = float(raw_full_perf.loc["partial_R2_trait", "value"])
    full_delta_AIC = float(raw_full_perf.loc["delta_AIC", "value"])

    # --- Combined main-effect table (numeric, ready for plotting) ---
    df_main = pd.concat([df_fc_main, df_fd_main], ignore_index=True)
    df_main["significant"] = df_main["p_value"] < 0.05
    df_main["partial_R2_pct"] = df_main["partial_R2"] * 100
    df_main["R2_marginal_pct"] = df_main["R2_marginal"] * 100
    # Benchmark partial R2 = 0 by definition
    df_main["delta_R2_marginal_pct"] = (df_main["R2_marginal"] - benchmark_R2m) * 100

    # Separate PC-based vs individual-trait subsets
    pc_vars = {"PC1", "PC2", "PC3"}
    individual_traits = {
        "Carbon", "Cellulose", "Chlorophyll a + b", "EWT", "Lignin",
        "Nitrogen", "NSC", "Phenolics", "SLA", "Canopy height",
    }
    df_main["var_type"] = np.where(
        df_main["variable"].isin(pc_vars), "PC",
        np.where(df_main["variable"].isin(individual_traits), "Trait", "FD")
    )
    return (
        benchmark_R2m,
        df_main,
        fd_list,
        full_R2m,
        full_delta_AIC,
        rename_map,
        trait_list,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Organize: Aridity Index Interaction Results
    """)
    return


@app.cell
def _(
    fd_list,
    pd,
    raw_fc_ai,
    raw_fd_ai,
    raw_full_ai_perf,
    rename_map,
    trait_list,
):
    # --- FC x AI interaction ---
    _rows_fc_ai = []
    for _t in trait_list:
        _display_name = rename_map.get(_t, _t)
        _rows_fc_ai.append({
            "category": "FC x AI",
            "variable": _display_name,
            "coef_main": raw_fc_ai.loc[_t, "coef_trait"],
            "coef_interaction": raw_fc_ai.loc[_t, "coef_interaction"],
            "coef_interaction_CI_lower": raw_fc_ai.loc[_t, "coef_interaction_CI_lower"],
            "coef_interaction_CI_upper": raw_fc_ai.loc[_t, "coef_interaction_CI_upper"],
            "p_value_interaction": raw_fc_ai.loc[_t, "p_value_interaction"],
            "p_value_main": raw_fc_ai.loc[_t, "p_value_trait"],
            "delta_AIC": raw_fc_ai.loc[_t, "delta_AIC"],
            "R2_marginal": raw_fc_ai.loc[_t, "R2_marginal"],
            "R2_conditional": raw_fc_ai.loc[_t, "R2_conditional"],
            "partial_R2": raw_fc_ai.loc[_t, "partial_R2_trait"],
        })
    df_fc_interaction = pd.DataFrame(_rows_fc_ai)

    # --- FD x AI interaction ---
    _rows_fd_ai = []
    for _fd in fd_list:
        _parts = _fd.split("_", 1)
        _metrics_name = rename_map.get(_parts[0], _parts[0])
        _scale = _parts[1] if len(_parts) > 1 else ""
        _rows_fd_ai.append({
            "category": "FD x AI",
            "variable": _metrics_name,
            "scale": _scale,
            "coef_main": raw_fd_ai.loc[_fd, "coef_FD"],
            "coef_interaction": raw_fd_ai.loc[_fd, "coef_interaction"],
            "coef_interaction_CI_lower": raw_fd_ai.loc[_fd, "coef_interaction_CI_lower"],
            "coef_interaction_CI_upper": raw_fd_ai.loc[_fd, "coef_interaction_CI_upper"],
            "p_value_interaction": raw_fd_ai.loc[_fd, "p_value_interaction"],
            "p_value_main": raw_fd_ai.loc[_fd, "p_value_FD"],
            "delta_AIC": raw_fd_ai.loc[_fd, "delta_AIC"],
            "R2_marginal": raw_fd_ai.loc[_fd, "R2_marginal"],
            "R2_conditional": raw_fd_ai.loc[_fd, "R2_conditional"],
            "partial_R2": raw_fd_ai.loc[_fd, "partial_R2_FD"],
        })
    df_fd_interaction = pd.DataFrame(_rows_fd_ai)

    # --- Full model with AI interaction ---
    full_ai_R2m = float(raw_full_ai_perf.loc["R2_marginal", "value"])
    full_ai_R2c = float(raw_full_ai_perf.loc["R2_conditional", "value"])
    full_ai_partial_R2 = float(raw_full_ai_perf.loc["partial_R2_trait", "value"])
    full_ai_delta_AIC = float(raw_full_ai_perf.loc["delta_AIC", "value"])

    # --- Combined interaction table ---
    df_interaction = pd.concat(
        [df_fc_interaction, df_fd_interaction], ignore_index=True
    )
    df_interaction["significant"] = df_interaction["p_value_interaction"] < 0.05
    df_interaction["partial_R2_pct"] = df_interaction["partial_R2"] * 100
    df_interaction["R2_marginal_pct"] = df_interaction["R2_marginal"] * 100
    return df_interaction, full_ai_R2m, full_ai_delta_AIC


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Prepare plotting subsets (PC & FD, main + interaction)
    """)
    return


@app.cell
def _(Path):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    import seaborn as sns
    import sys

    # Shared styling module lives at the repository root: ../functions/plot_style.py
    try:
        _repo_root = Path(__file__).resolve().parents[1]
    except NameError:
        _repo_root = Path.cwd().resolve().parent
    sys.path.insert(0, str(_repo_root))
    from functions.plot_style import apply_style, FIG_FULL

    apply_style(font="Helvetica", context="paper")
    return FIG_FULL, Line2D, gridspec, plt, sns


@app.cell
def _(benchmark_R2m, df_main, full_R2m):
    # --- Main-effect subsets ---
    df_pc = df_main[df_main["var_type"] == "PC"].copy()
    df_fd_plot = df_main[df_main["var_type"] == "FD"].copy()

    _fd_short = {
        "Functional richness": "FRic",
        "Functional divergence": "FDiv",
        "Functional dissimilarity": "FDis",
    }
    df_fd_plot["short_name"] = df_fd_plot["variable"].map(_fd_short)

    benchmark_R2m_pct = benchmark_R2m * 100
    full_R2m_pct = full_R2m * 100
    return benchmark_R2m_pct, df_fd_plot, df_pc, full_R2m_pct


@app.cell
def _(benchmark_R2m, df_interaction, full_ai_R2m, np):
    # --- Interaction subsets ---
    _pc_vars = {"PC1", "PC2", "PC3"}
    _individual_traits = {
        "Carbon", "Cellulose", "Chlorophyll a + b", "EWT", "Lignin",
        "Nitrogen", "NSC", "Phenolics", "SLA", "Canopy height",
    }
    df_interaction["var_type"] = np.where(
        df_interaction["variable"].isin(_pc_vars), "PC",
        np.where(df_interaction["variable"].isin(_individual_traits), "Trait", "FD")
    )
    df_interaction["delta_R2_marginal_pct"] = (
        df_interaction["R2_marginal"] - benchmark_R2m
    ) * 100

    df_pc_int = df_interaction[df_interaction["var_type"] == "PC"].copy()
    df_fd_int = df_interaction[df_interaction["var_type"] == "FD"].copy()

    _fd_short = {
        "Functional richness": "FRic",
        "Functional divergence": "FDiv",
        "Functional dissimilarity": "FDis",
    }
    df_fd_int["short_name"] = df_fd_int["variable"].map(_fd_short)

    benchmark_R2m_pct_int = benchmark_R2m * 100
    full_ai_R2m_pct = full_ai_R2m * 100
    return benchmark_R2m_pct_int, df_fd_int, df_pc_int, full_ai_R2m_pct


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Combined stacked figure

    One vertically-stacked figure where **each model component is its own axes**
    with a subtitle (Null → +FC → +FD → Full → +FC×AI → +FD×AI → Full → +PFT).
    All left (bar) axes share the marginal-$R^2$ ruler and a vertical dashed Null
    reference line; each component's right forest axis shares its left neighbour's
    y positions. Functional-diversity metrics are drawn as **side-by-side grouped
    bars** (scales offset within each FRic / FDiv / FDis group), as in the
    original figures.
    """)
    return


@app.cell
def _(
    FIG_FULL,
    Line2D,
    benchmark_R2m_pct,
    benchmark_R2m_pct_int,
    df_fd_int,
    df_fd_plot,
    df_pc,
    df_pc_int,
    full_R2m_pct,
    full_ai_R2m_pct,
    full_ai_delta_AIC,
    full_delta_AIC,
    gridspec,
    pft_R2m,
    pft_ai_R2m,
    pft_delta_AIC,
    plt,
    results_dir,
    sns,
):
    # ---------------- style / constants ----------------
    color_fc = "#266489"       # FC significant
    color_fc_ns = "#AAC4E0"    # FC non-significant
    color_full = "#2c2c2c"     # Full model
    color_pft = "#5b8c5a"      # PFT
    errorbar_color = "#8F97A4"

    scale_color = {
        "alpha": "#FF6347", "gamma": "#F4A460", "tau": "#FFD700",
        "alpha_to_gamma": "#6A5ACD", "gamma_to_tau": "#469eb4",
    }
    scale_label = {
        "alpha": r"$\bar{\alpha}$", "gamma": r"$\bar{\gamma}$", "tau": r"$\tau$",
        "alpha_to_gamma": r"$\bar{\beta}_{\alpha \rightarrow \gamma}$",
        "gamma_to_tau": r"$\beta_{\gamma \rightarrow \tau}$",
    }
    # FD metric groups: scales sit side-by-side within each group (as in the
    # original figures)
    FD_GROUPS = [
        ("FRic", "Functional\nrichness", ["alpha", "gamma", "tau"]),
        ("FDiv", "Functional\ndivergence", ["alpha", "gamma", "tau"]),
        ("FDis", "Functional\ndissimilarity", ["alpha_to_gamma", "gamma_to_tau"]),
    ]

    BARH = 0.7          # bar thickness for single / PC / PFT bars
    FD_BARH = 0.375      # bar thickness for each FD scale (grouped)
    GROUP_STEP = 1.35   # y-distance between FD metric groups

    # font sizes (paper context; see ../functions/plot_style.py)
    TITLE_FS = 9        # panel subtitles
    LABEL_FS = 10       # panel labels a, b, c ...
    TICK_FS = 8         # y-tick labels (variable / group names)
    VAL_FS = 7          # R2 value inside bar
    ANNOT_FS = 6        # delta-R2 / delta-AIC annotations
    FD_VAL_FS = 6       # R2 value inside the thinner FD bars
    FD_ANNOT_FS = 5     # delta annotations on FD bars
    LEG_FS = 7          # legend text

    def _fd_offsets(n):
        return [(j - (n - 1) / 2) * FD_BARH for j in range(n)]

    # shared x-range for all R2 bar axes (headroom for outside dR2 text)
    _all_vals = (
        list(df_pc["R2_marginal_pct"]) + list(df_fd_plot["R2_marginal_pct"])
        + list(df_pc_int["R2_marginal_pct"]) + list(df_fd_int["R2_marginal_pct"])
        + [full_R2m_pct, full_ai_R2m_pct, pft_R2m * 100, pft_ai_R2m * 100,
           benchmark_R2m_pct]
    )
    x_max = max(_all_vals) + 6.0

    # ---------------- annotation helper ----------------
    def _annot(ax, y, value, delta, aic, val_fs, annot_fs):
        if value > 3:
            ax.text(value / 2, y, f"{value:.1f}", ha="center", va="center",
                    color="white", fontsize=val_fs, fontweight="bold")
        if delta is not None:
            _sign = "+" if delta >= 0 else ""
            ax.text(value + 0.6, y, f"{_sign}{delta:.1f}", ha="left",
                    va="center", fontsize=annot_fs)
            if aic is not None and value > 6:
                ax.text(value - 0.6, y, rf"$\Delta$AIC={aic:.0f}", ha="right",
                        va="center", color="white", fontsize=annot_fs)

    def _finish_bar_ax(ax, benchmark_pct, title, ylo, yhi):
        ax.axvline(benchmark_pct, color="gray", ls="--", lw=0.8, zorder=1)
        ax.set_ylim(ylo, yhi)
        ax.invert_yaxis()
        ax.set_xlim(0, x_max)
        ax.set_title(title, loc="center", fontsize=TITLE_FS, fontweight="bold", pad=3)

    # ---------------- left-column bar drawers ----------------
    def _draw_single_bar(ax, value, color, delta, aic, benchmark_pct, title,
                         formula=None):
        ax.barh(0, value, height=BARH, color=color, zorder=2)
        _annot(ax, 0, value, delta, aic, VAL_FS, ANNOT_FS)
        ax.set_yticks([])
        if formula:
            # model formula at the top-left of the axes (single line)
            ax.text(0.01, 0.75, formula, transform=ax.transAxes, ha="left",
                    va="bottom", fontsize=6.5, fontstyle="italic")
        _finish_bar_ax(ax, benchmark_pct, title, -0.7, 0.7)

    def _draw_fc_bars(ax, df_pc_b, benchmark_pct, title):
        for _i, (_, r) in enumerate(df_pc_b.iterrows()):
            _color = color_fc if r["significant"] else color_fc_ns
            ax.barh(_i, r["R2_marginal_pct"], height=BARH, color=_color, zorder=2)
            _annot(ax, _i, r["R2_marginal_pct"], r["delta_R2_marginal_pct"],
                   r["delta_AIC"], VAL_FS, ANNOT_FS)
        ax.set_yticks(range(len(df_pc_b)))
        ax.set_yticklabels(df_pc_b["variable"], fontsize=TICK_FS)
        _finish_bar_ax(ax, benchmark_pct, title, -0.7, len(df_pc_b) - 1 + 0.7)

    def _draw_fd_bars(ax, df_fd_b, benchmark_pct, title):
        _yticks, _ylabels = [], []
        for _gi, (_metric, _gname, _scales) in enumerate(FD_GROUPS):
            _center = _gi * GROUP_STEP
            _offs = _fd_offsets(len(_scales))
            for _j, _sc in enumerate(_scales):
                _row = df_fd_b[(df_fd_b["short_name"] == _metric)
                               & (df_fd_b["scale"] == _sc)].iloc[0]
                _y = _center + _offs[_j]
                ax.barh(_y, _row["R2_marginal_pct"], height=FD_BARH,
                        color=scale_color[_sc], zorder=2)
                _annot(ax, _y, _row["R2_marginal_pct"],
                       _row["delta_R2_marginal_pct"], _row["delta_AIC"],
                       FD_VAL_FS, FD_ANNOT_FS)
            _yticks.append(_center)
            _ylabels.append(_gname)
        ax.set_yticks(_yticks)
        ax.set_yticklabels(_ylabels, fontsize=TICK_FS)
        _finish_bar_ax(ax, benchmark_pct, title, -0.7,
                       (len(FD_GROUPS) - 1) * GROUP_STEP + 0.7)

    def _draw_pft_bars(ax, benchmark_pct, title):
        _val = pft_R2m * 100
        ax.barh(0, _val, height=BARH, color=color_pft, zorder=2)
        _annot(ax, 0, _val, _val - benchmark_pct, pft_delta_AIC, VAL_FS, ANNOT_FS)
        ax.set_yticks([])
        _finish_bar_ax(ax, benchmark_pct, title, -0.7, 0.7)

    # ---------------- right-column coefficient drawers ----------------
    def _coef_point(ax, x, y, mcolor, lo, hi, sig):
        _xerr = [[x - lo], [hi - x]]
        ax.errorbar(x, y, xerr=_xerr, fmt="o", color=errorbar_color,
                    markersize=0, lw=1, capsize=2, zorder=4)
        if sig:
            ax.scatter(x, y, color=mcolor, s=13, zorder=5)
        else:
            ax.scatter(x, y, color=mcolor, s=13, zorder=5, facecolors="white")

    def _draw_fc_coef(ax, df_pc_b, coef_col, lo_col, hi_col):
        for _i, (_, r) in enumerate(df_pc_b.iterrows()):
            _coef_point(ax, r[coef_col], _i, color_fc, r[lo_col], r[hi_col],
                        bool(r["significant"]))
        ax.axvline(0, color="gray", ls="--", lw=0.7)
        # category labels on the right (same rows as the left bar column)
        ax.set_yticks(range(len(df_pc_b)))
        ax.set_yticklabels(df_pc_b["variable"], fontsize=TICK_FS)
        ax.yaxis.set_ticks_position("right")
        ax.tick_params(left=False, labelleft=False, right=True, labelright=True)

    def _draw_fd_coef(ax, df_fd_b, coef_col, lo_col, hi_col):
        for _gi, (_metric, _gname, _scales) in enumerate(FD_GROUPS):
            _center = _gi * GROUP_STEP
            _offs = _fd_offsets(len(_scales))
            for _j, _sc in enumerate(_scales):
                _row = df_fd_b[(df_fd_b["short_name"] == _metric)
                               & (df_fd_b["scale"] == _sc)].iloc[0]
                _coef_point(ax, _row[coef_col], _center + _offs[_j],
                            scale_color[_sc], _row[lo_col], _row[hi_col],
                            bool(_row["significant"]))
        ax.axvline(0, color="gray", ls="--", lw=0.7)
        # metric-group labels on the right (same rows as the left bar column)
        ax.set_yticks([_gi * GROUP_STEP for _gi in range(len(FD_GROUPS))])
        ax.set_yticklabels([_g[1] for _g in FD_GROUPS], fontsize=TICK_FS)
        ax.yaxis.set_ticks_position("right")
        ax.tick_params(left=False, labelleft=False, right=True, labelright=True)

    # ---------------- figure / gridspec ----------------
    # rows (top->bottom): Null, [gap], +FC, +FD, Full, [gap], +FC*AI, +FD*AI,
    # Full(AI), +PFT. The two empty spacer rows (idx 1, 5) widen only the
    # a->b and d->e gaps to host the section headers.
    _gap = 0.5    # big gap: hosts the section headers (a->b, d->e, g->h)
    _sgap = 0.1  # small gap above each Full-model panel
    _hr = [1.4, _gap, 3.4, 4.0, _sgap, 1.4, _gap, 3.4, 4.0, _sgap, 1.4, _gap, 1.4]
    fig = plt.figure(figsize=(FIG_FULL, 10), constrained_layout=False)
    gs = gridspec.GridSpec(
        13, 2, width_ratios=[4, 2], height_ratios=_hr,
        hspace=0.15, wspace=0.0,  # <- hspace controls vertical gap between panel rows
    )

    # left column (bars) — all share the marginal-R2 x ruler
    axL_null = fig.add_subplot(gs[0, 0])
    axL_fc = fig.add_subplot(gs[2, 0], sharex=axL_null)
    axL_fd = fig.add_subplot(gs[3, 0], sharex=axL_null)
    axL_full = fig.add_subplot(gs[5, 0], sharex=axL_null)
    axL_fc_i = fig.add_subplot(gs[7, 0], sharex=axL_null)
    axL_fd_i = fig.add_subplot(gs[8, 0], sharex=axL_null)
    axL_full_i = fig.add_subplot(gs[10, 0], sharex=axL_null)
    axL_pft = fig.add_subplot(gs[12, 0], sharex=axL_null)

    # right column (coefficient forests)
    axR_fc = fig.add_subplot(gs[2, 1], sharey=axL_fc)
    axR_fd = fig.add_subplot(gs[3, 1], sharey=axL_fd, sharex=axR_fc)
    axR_fc_i = fig.add_subplot(gs[7, 1], sharey=axL_fc_i)
    axR_fd_i = fig.add_subplot(gs[8, 1], sharey=axL_fd_i, sharex=axR_fc_i)

    # ---- draw left column ----
    _draw_single_bar(axL_null, benchmark_R2m_pct, "gray", None, None,
                     benchmark_R2m_pct, "Baseline model",
                     formula=r"GPP ~ LAI + T + P + AI + (1 | Ecoprovince)")
    _draw_fc_bars(axL_fc, df_pc, benchmark_R2m_pct, "+ Functional composition")
    _draw_fd_bars(axL_fd, df_fd_plot, benchmark_R2m_pct, "+ Functional diversity")
    _draw_single_bar(axL_full, full_R2m_pct, color_full,
                     full_R2m_pct - benchmark_R2m_pct, full_delta_AIC,
                     benchmark_R2m_pct, "Full model",
                     formula="Baseline model + Functional composition"
                             " + Functional diversity")
    _draw_fc_bars(axL_fc_i, df_pc_int, benchmark_R2m_pct_int,
                  r"+ Functional composition $\times$ aridity index")
    _draw_fd_bars(axL_fd_i, df_fd_int, benchmark_R2m_pct_int,
                  r"+ Functional diversity $\times$ aridity index")
    _draw_single_bar(axL_full_i, full_ai_R2m_pct, color_full,
                     full_ai_R2m_pct - benchmark_R2m_pct_int, full_ai_delta_AIC,
                     benchmark_R2m_pct_int, "Full model with aridity index interaction",
                     formula=r"Baseline model + Functional composition $\times$ aridity index"
                             r" + Functional diversity $\times$ aridity index")
    _draw_pft_bars(axL_pft, benchmark_R2m_pct, "+ PFT")

    # ---- draw right column ----
    _draw_fc_coef(axR_fc, df_pc, "coef", "coef_CI_lower", "coef_CI_upper")
    _draw_fd_coef(axR_fd, df_fd_plot, "coef", "coef_CI_lower", "coef_CI_upper")
    _draw_fc_coef(axR_fc_i, df_pc_int, "coef_interaction",
                  "coef_interaction_CI_lower", "coef_interaction_CI_upper")
    _draw_fd_coef(axR_fd_i, df_fd_int, "coef_interaction",
                  "coef_interaction_CI_lower", "coef_interaction_CI_upper")

    # ---- axis labels: only bottom-most of each shared group ----
    _left_upper = [axL_null, axL_fc, axL_fd, axL_full, axL_fc_i, axL_fd_i, axL_full_i]
    for _ax in _left_upper:
        plt.setp(_ax.get_xticklabels(), visible=False)
    axL_pft.set_xlabel(r"Marginal $R^2$ (%)")

    plt.setp(axR_fc.get_xticklabels(), visible=False)
    axR_fd.set_xlabel("Coefficient")
    plt.setp(axR_fc_i.get_xticklabels(), visible=False)
    axR_fd_i.set_xlabel("Interaction coefficient")

    # despine everything visible
    for _ax in [axL_null, axL_fc, axL_fd, axL_full, axL_fc_i, axL_fd_i,
                axL_full_i, axL_pft, axR_fc, axR_fd, axR_fc_i, axR_fd_i]:
        sns.despine(ax=_ax)

    # drop the bottom x-axis (spine + ticks) on all panels except the
    # bottom-most of each shared-x group: e, j (coef) and l (Marginal R2)
    for _ax in [axL_null, axL_fc, axR_fc, axL_fd, axL_full, axL_fc_i, axR_fc_i,
                axL_fd_i, axL_full_i]:
        _ax.spines["bottom"].set_visible(False)
        _ax.tick_params(bottom=False)

    # coefficient panels (c, e, h, j): put the y-axis spine on the right
    for _ax in [axR_fc, axR_fd, axR_fc_i, axR_fd_i]:
        _ax.spines["left"].set_visible(False)
        _ax.spines["right"].set_visible(True)

    # ---- legends inside panels (no titles): FD scale -> d, Significance -> c ----
    _scale_handles = [
        Line2D([], [], color=scale_color[s], lw=4, label=scale_label[s])
        for s in ["alpha", "gamma", "tau", "alpha_to_gamma", "gamma_to_tau"]
    ]
    axL_fd.legend(
        handles=_scale_handles, loc="lower right", fontsize=LEG_FS, frameon=False,
        ncol=2, handlelength=1, columnspacing=0.6, handletextpad=0.4, labelspacing=0.3,
        bbox_to_anchor=(0.92, 0)
    )
    _sig_handles = [
        Line2D([], [], marker="o", color="black", lw=0, markersize=4, label=r"$p < 0.05$"),
        Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="black",
               lw=0, markersize=4, label=r"$p \geq 0.05$"),
    ]
    axR_fd.legend(
        handles=_sig_handles, loc="lower right", 
        fontsize=LEG_FS, frameon=False, handletextpad=0.1, labelspacing=0.1,
        # bbox_to_anchor=(1.02, 1.19),
    )

    # ---- panel labels (a, b, c, ...) at each axes' top-left ----
    # FD and Full-model panels are left unlabelled; the rest are lettered in order
    _panel_axes = [
        axL_null, axL_fc, axR_fc, axL_fc_i, axR_fc_i, axL_pft,
    ]
    # --- adjust panel-label placement here (axes fraction) ---
    LABEL_X_LEFT = -0.13    # labels on the wide left-column bar panels (a, b, d, f)
    LABEL_X_RIGHT = -0.03   # labels on the narrow right-column coef panels (c, e)
    LABEL_Y = 1.0          # shared vertical position for all labels
    _left_col_labels = {"a", "b", "d", "f"}
    for _lab, _ax in zip("abcdef", _panel_axes):
        _x = LABEL_X_LEFT if _lab in _left_col_labels else LABEL_X_RIGHT
        _ax.text(_x, LABEL_Y, _lab, transform=_ax.transAxes,
                 fontsize=LABEL_FS, fontweight="bold", va="top", ha="right")

    # ---- section headers: horizontal labels at the top-left of the first
    # panel of each block (b = Main effect, e = Interaction effect), placed
    # in the gap just above the panel in figure coordinates ----
    SECTION_FS = 11
    _sec_bbox = dict(facecolor="0.85", edgecolor="none", boxstyle="round,pad=0.3")
    _b_pos = axL_fc.get_position()
    fig.text(_b_pos.x0 - 0.08, _b_pos.y1 + 0.01, "Main effect",
             ha="left", va="bottom", fontsize=SECTION_FS, fontweight="bold",
             bbox=_sec_bbox)
    _e_pos = axL_fc_i.get_position()
    fig.text(_e_pos.x0 - 0.08, _e_pos.y1 + 0.01, "Interaction effect",
             ha="left", va="bottom", fontsize=SECTION_FS, fontweight="bold",
             bbox=_sec_bbox)
    _h_pos = axL_pft.get_position()
    fig.text(_h_pos.x0 - 0.08, _h_pos.y1 + 0.01, "PFT comparison",
             ha="left", va="bottom", fontsize=SECTION_FS, fontweight="bold",
             bbox=_sec_bbox)

    fig.savefig(
        results_dir / "fig3_model_comparison.png",
        dpi=600, bbox_inches="tight",
    )
    fig
    return


if __name__ == "__main__":
    app.run()
