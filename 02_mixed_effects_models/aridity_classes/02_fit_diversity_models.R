# =============================================================================
# aridity_classes/02_fit_diversity_models.R
#
# Purpose : Fit, SEPARATELY WITHIN EACH ARIDITY CLASS (Arid, Semi-arid,
#           Dry sub-humid, Sub-humid, Humid), the functional-diversity models
#           that add ONE FD metric at a time to the baseline model
#               GPP ~ LAI + Temperature + Precipitation + aridity_index + (1 | ecoprovince)
#           Predictors in the input table are standardised within each aridity
#           class. Residual spatial autocorrelation is modelled with corGaus on
#           latitude/longitude (with nugget).
# Inputs  : ../../data/analysis_ready_tables/data_aridity_index_class_scaler.csv
# Outputs : LME_AI_class_GPP_LAI_T_P_AI_FD_ecoprovince_SA_model_list_grouped.rds
#             (list[FD] -> list[aridity class] -> list(group, model))
# Run     : Rscript 02_fit_diversity_models.R   (working directory = aridity_classes/)
#           Run order: after 01_fit_composition_models.R
# Feeds   : Fig. 4 (through 03_evaluate_models.Rmd and 04_plot_fig4.ipynb)
# =============================================================================

library(nlme)
library(dplyr)
library(parallel)
library(foreach)
library(doParallel)

# Load the analysis-ready table (predictors standardised within aridity class)
data <- read.csv("../../data/analysis_ready_tables/data_aridity_index_class_scaler.csv")

# Functional diversity metrics tested one at a time
FD_list <- c('FRic_alpha', 'FDiv_alpha', 'FRic_gamma', 'FDiv_gamma', 'FRic_tau', 'FDiv_tau',
             'Fbeta_alpha_to_gamma', 'Fbeta_gamma_to_tau')

# Parallel back-end (one worker per metric, leaving one core for the OS)
numCores <- max(1, min(length(FD_list), parallel::detectCores() - 1))
cl <- makeCluster(numCores)
registerDoParallel(cl)

# For each FD metric (in parallel), fit one model per aridity class
model_list <- foreach(FD = FD_list, .packages = c("nlme", "dplyr")) %dopar% {
  print(FD)

  # Fit the model within each aridity class
  results_by_group <- lapply(split(data, data$aridity_index_classification), function(subdata) {
    formula <- as.formula(paste("GPP ~ LAI + Temperature + Precipitation + aridity_index + ", FD))
    cor_struct <- corGaus(form = ~latitude + longitude, nugget = TRUE)
    model <- lme(fixed = formula, random = ~1 | ecoprovince, data = subdata, method = "ML", correlation = cor_struct)
    return(list(group = unique(subdata$aridity_index_classification), model = model))
  })

  return(results_by_group)

}

# Save the model list
names(model_list) <- FD_list
saveRDS(model_list, "LME_AI_class_GPP_LAI_T_P_AI_FD_ecoprovince_SA_model_list_grouped.rds")

# Shut down the parallel back-end
stopCluster(cl)
