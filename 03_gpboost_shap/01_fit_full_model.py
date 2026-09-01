"""Fit the full GPBoost model of GPP (step 01 of the Fig. 5 pipeline).

Run order: 01 -> 03 -> 04 (02 is independent and can run any time after 01).

Purpose
    Tune (Optuna, 30 trials) and fit the tree-boosted spatial Gaussian-process model

        GPP ~ F(LAI, Temperature, Precipitation, aridity_index,
                all_PC1, all_PC2, all_PC3,
                FRic_gamma, Fbeta_alpha_to_gamma, Fbeta_gamma_to_tau, FDiv_gamma)
              + GP(latitude, longitude)

    on the 4,371 analysis-grid cells (80 % train / 20 % test split). This is the model
    whose SHAP values are mapped in Fig. 5 and whose R2 values are quoted in the text
    (see 02_model_performance.py).

Inputs
    ../data/analysis_ready_tables/data.csv   z-scored predictors, one row per grid cell
                                             (see ../data/README.md)

Outputs
    ./models/GPBoost_GPP_LAI_full-model_ecoprovince.json
        saved booster; the JSON also stores the fitted GP covariance parameters.

Notes
    - The train/test split uses random_state=42 as in the paper. The Optuna sampler is
      NOT seeded (as in the original analysis), so the tuned hyperparameters and the
      downstream R2 / SHAP values vary slightly between runs.
    - The file-name suffix "ecoprovince" is historical: the random effect is a spatial
      Gaussian process on latitude/longitude (Vecchia approximation), not an
      ecoprovince grouping.
"""

import os

import gpboost as gpb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_CSV = "../data/analysis_ready_tables/data.csv"
MODEL_DIR = "./models"
MODEL_FILE = f"{MODEL_DIR}/GPBoost_GPP_LAI_full-model_ecoprovince.json"
os.makedirs(MODEL_DIR, exist_ok=True)

data = pd.read_csv(DATA_CSV)

# 80/20 split; the test set is used for early stopping and as the Optuna objective
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

baseline_var_list = ['LAI', 'Temperature', 'Precipitation', 'aridity_index']
full_model_var_list = ['all_PC1', 'all_PC2', 'all_PC3', 'FRic_gamma', 'Fbeta_alpha_to_gamma', 'Fbeta_gamma_to_tau',
                       'FDiv_gamma', ]
feature_list = baseline_var_list + full_model_var_list

data_train_gpb = gpb.Dataset(data_train[feature_list], label=data_train["GPP"])
data_test_gpb = gpb.Dataset(data_test[feature_list], label=data_test["GPP"])


def objective(trial):
    """Optuna objective: test-set MSE of the boosted trees + spatial GP."""
    param = {
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 8),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 1000),
        'learning_rate': trial.suggest_categorical('learning_rate',
                                                   [0.005, 0.0075, 0.01, 0.02, 0.03]),
        'num_leaves': 2 ** 10,
    }

    gp_model = gpb.GPModel(
        gp_coords=data_train[['latitude', 'longitude']],
        cov_function='gaussian',
        gp_approx='vecchia'
    )
    gp_model.set_prediction_data(
        gp_coords_pred=data_test[['latitude', 'longitude']],
    )

    gpboost = gpb.train(params=param, train_set=data_train_gpb,
                        valid_sets=data_test_gpb,
                        use_gp_model_for_validation=True,
                        gp_model=gp_model, num_boost_round=10000,
                        early_stopping_rounds=10, verbose_eval=False)
    y_pred = gpboost.predict(data_test[feature_list],
                             gp_coords_pred=data_test[['latitude', 'longitude']],
                             )
    mse = np.mean((data_test["GPP"] - y_pred['response_mean']) ** 2)
    return mse


# hyperparameter search (unseeded, see module docstring)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
trial = study.best_trial
params = trial.params
print(params)

# refit with the best hyperparameters and save
gp_model = gpb.GPModel(
    gp_coords=data_train[['latitude', 'longitude']],
    cov_function='gaussian',
    gp_approx='vecchia'
)
gp_model.set_prediction_data(
    gp_coords_pred=data_test[['latitude', 'longitude']]
)

gpboost = gpb.train(params=params, train_set=data_train_gpb,
                    valid_sets=data_test_gpb,
                    use_gp_model_for_validation=True,
                    gp_model=gp_model, num_boost_round=10000,
                    early_stopping_rounds=10, verbose_eval=False)
gpboost.save_model(MODEL_FILE)
print(f"saved {MODEL_FILE}")
