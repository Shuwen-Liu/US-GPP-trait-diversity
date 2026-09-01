# =============================================================================
# aridity_classes/01_fit_composition_models.R
#
# Purpose : Fit, SEPARATELY WITHIN EACH ARIDITY CLASS (Arid, Semi-arid,
#           Dry sub-humid, Sub-humid, Humid), the baseline model
#               GPP ~ LAI + Temperature + Precipitation + aridity_index + (1 | ecoprovince)
#           and the functional-composition models that add ONE trait (or trait
#           PC) at a time. Predictors in the input table are standardised within
#           each aridity class. Residual spatial autocorrelation is modelled with
#           corGaus on latitude/longitude (with nugget).
# Inputs  : ../../data/analysis_ready_tables/data_aridity_index_class_scaler.csv
# Outputs : LME_AI_class_GPP_LAI_T_P_AI_trait_ecoprovince_SA_model_list_grouped.rds
#             (list[trait] -> list[aridity class] -> list(group, model))
#           LME_AI_class_GPP_LAI_T_P_AI_ecoprovince_SA_model_list_grouped.rds
#             (baseline model per aridity class)
# Run     : Rscript 01_fit_composition_models.R   (working directory = aridity_classes/)
#           Run order in this folder: 01 -> 02 -> 03 (Rmd) -> 04 (ipynb)
# Feeds   : Fig. 4 (through 03_evaluate_models.Rmd and 04_plot_fig4.ipynb)
# =============================================================================

library(nlme)
library(dplyr)
library(parallel)
library(foreach)
library(doParallel)

# Load the analysis-ready table (predictors standardised within aridity class)
data <- read.csv("../../data/analysis_ready_tables/data_aridity_index_class_scaler.csv")

# Traits / trait PCs tested one at a time
trait_list <- c('Carbon', 'Cellulose', 'ChlorophyllsArea', 'EWT', 'Lignin', 'Nitrogen',
                'NSC', 'Phenolics', 'SLA', 'canopy_height', 'all_PC1', 'all_PC2', 'all_PC3')

# Parallel back-end (one worker per trait, leaving one core for the OS)
numCores <- max(1, min(length(trait_list), parallel::detectCores() - 1))
cl <- makeCluster(numCores)
registerDoParallel(cl)

# For each trait (in parallel), fit one model per aridity class
model_list <- foreach(trait = trait_list, .packages = c("nlme", "dplyr")) %dopar% {
  print(trait)

  # Fit the model within each aridity class
  results_by_group <- lapply(split(data, data$aridity_index_classification), function(subdata) {
    formula <- as.formula(paste("GPP ~ LAI + Temperature + Precipitation + aridity_index + ", trait))
    cor_struct <- corGaus(form = ~latitude + longitude, nugget = TRUE)
    model <- lme(fixed = formula, random = ~1 | ecoprovince, data = subdata, method = "ML", correlation = cor_struct)
    return(list(group = unique(subdata$aridity_index_classification), model = model))
  })

  return(results_by_group)
}

# Save the model list
names(model_list) <- trait_list
saveRDS(model_list, "LME_AI_class_GPP_LAI_T_P_AI_trait_ecoprovince_SA_model_list_grouped.rds")

# Shut down the parallel back-end
stopCluster(cl)

# Baseline model (no trait term) within each aridity class
model_LAI_list <- lapply(split(data, data$aridity_index_classification), function(subdata) {
  formula <- as.formula(paste("GPP ~ LAI + Temperature + Precipitation + aridity_index"))
  cor_struct <- corGaus(form = ~latitude + longitude, nugget = TRUE)
  model <- lme(fixed = formula, random = ~1 | ecoprovince, data = subdata, method = "ML", correlation = cor_struct)
  return(list(group = unique(subdata$aridity_index_classification), model = model))
})

saveRDS(model_LAI_list, "LME_AI_class_GPP_LAI_T_P_AI_ecoprovince_SA_model_list_grouped.rds")
