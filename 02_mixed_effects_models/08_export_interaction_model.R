# =============================================================================
# 08_export_interaction_model.R
#
# Purpose : Export the forward-selected interaction model
#           (07_forward_selection_interaction_model.R): fixed-effect estimates
#           with 95% CIs and p-values, and model performance (marginal /
#           conditional R2, AIC, ICC, RMSE) together with the partial R2 and
#           delta AIC relative to the baseline model.
# Inputs  : nlme_AI_forward_selection_PC3_final_model.rds   (from 07)
#           LME_GPP_LAI_T_P_AI_ecoprovince_SA.rds            (baseline, from 01)
# Outputs : nlme_AI_forward_selection_PC3_final_model_estimate.csv
#           nlme_AI_forward_selection_PC3_final_model_performance.csv
# Run     : Rscript 08_export_interaction_model.R   (working directory = this folder)
#           Run order: after 07_forward_selection_interaction_model.R
# Feeds   : Fig. 3 ("Full model with aridity index interaction" bar; read by 11_plot_fig3.py)
# =============================================================================

library(nlme)
library(performance)
library(dplyr)

# Load the interaction model and the baseline model
model <- readRDS("nlme_AI_forward_selection_PC3_final_model.rds")

model_LAI <- readRDS("LME_GPP_LAI_T_P_AI_ecoprovince_SA.rds")
cor_struct <- corGaus(form = ~ latitude + longitude, nugget = TRUE)

perf_LAI <- model_performance(model_LAI, estimator = "ML")
partial_R2_LAI <- perf_LAI$R2_marginal

# Fixed-effect terms (excluding the intercept)
predictor_names <- rownames(summary(model)$tTable)
predictor_names <- predictor_names[predictor_names != "(Intercept)"]

# Collect estimate, 95% CI and p-value for each term
results_list <- list()

for (var in predictor_names) {
  coef_value <- summary(model)$tTable[var, "Value"]
  coef_CI_lower <- intervals(model, level = 0.95, which = "fixed")$fixed[var, "lower"]
  coef_CI_upper <- intervals(model, level = 0.95, which = "fixed")$fixed[var, "upper"]
  p_value <- summary(model)$tTable[var, "p-value"]

  results_list[[var]] <- data.frame(
    variable = var,
    coef = coef_value,
    coef_CI_lower = coef_CI_lower,
    coef_CI_upper = coef_CI_upper,
    p_value = p_value
  )
}

# Combine into one data.frame
results_df <- bind_rows(results_list)

# Significance stars
results_df <- results_df %>%
  mutate(p_sig = case_when(
    p_value < 0.001 ~ "***",
    p_value < 0.01  ~ "**",
    p_value < 0.05  ~ "*",
    TRUE            ~ ""
  ))

# Export CSV
write.csv(results_df, "nlme_AI_forward_selection_PC3_final_model_estimate.csv", row.names = FALSE)

# Optional: print a markdown table
knitr::kable(results_df, format = "markdown")


# Model performance metrics
perf <- model_performance(model, estimator = "ML")
R2_marginal <- perf$R2_marginal
R2_conditional <- ifelse(is.na(perf$R2_conditional), R2_marginal, perf$R2_conditional)
AIC <- perf$AIC
ICC <- ifelse(is.null(perf$ICC), 0, perf$ICC)
RMSE <- perf$RMSE

# Partial R2 and delta AIC relative to the baseline model
partial_R2_trait <- R2_marginal - partial_R2_LAI
delta_AIC <- AIC - perf_LAI$AIC

# Model formula as a single string
formula_fixed <- formula(model)
formula_str <- paste(deparse(formula_fixed), collapse = " ")
formula_str <- gsub("\t", " ", formula_str)         # drop tabs
formula_str <- gsub(" +", " ", formula_str)         # collapse repeated spaces
formula_str <- trimws(formula_str)                  # trim leading/trailing spaces
cat(formula_str)


df <- data.frame(
  Model_structure = formula_str,
  R2_marginal = R2_marginal,
  R2_conditional = R2_conditional,
  partial_R2_trait = partial_R2_trait,
  AIC = AIC,
  delta_AIC = delta_AIC,
  ICC = ICC,
  RMSE = RMSE
)
df_t <- as.data.frame(t(df))
colnames(df_t) <- "value"
df_t$indicator <- rownames(df_t)
df_t <- df_t[, c("indicator", "value")]
# Export CSV
write.csv(df_t, "nlme_AI_forward_selection_PC3_final_model_performance.csv", row.names = FALSE)

# Optional: print a markdown table
knitr::kable(df_t, format = "markdown")
