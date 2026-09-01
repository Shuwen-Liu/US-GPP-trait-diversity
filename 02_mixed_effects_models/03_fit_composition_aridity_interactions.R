# =============================================================================
# 03_fit_composition_aridity_interactions.R
#
# Purpose : Fit the functional-composition x aridity interaction models: the
#           baseline model plus ONE trait (or trait PC) and its interaction
#           with aridity_index,
#               GPP ~ LAI + Temperature + Precipitation + aridity_index
#                     + aridity_index * trait + (1 | ecoprovince)
#           with a Gaussian spatial correlation structure (corGaus, with nugget).
# Inputs  : ../data/analysis_ready_tables/data.csv
# Outputs : LME_GPP_LAI_T_P_AI_trait_AI_ecoprovince_SA_model_list.rds (one lme per trait)
# Run     : Rscript 03_fit_composition_aridity_interactions.R   (working directory = this folder)
#           Run order: after 02_fit_diversity_models.R
# Feeds   : Fig. 3 (through 09_evaluate_models.Rmd and 11_plot_fig3.py)
# =============================================================================

library(nlme)
library(dplyr)
library(parallel)
library(foreach)
library(doParallel)

# Load the analysis-ready table
data <- read.csv("../data/analysis_ready_tables/data.csv")

# Traits / trait PCs tested one at a time
trait_list <- c('Carbon', 'Cellulose', 'ChlorophyllsArea', 'EWT', 'Lignin', 'Nitrogen',
                'NSC', 'Phenolics', 'SLA', 'canopy_height', 'all_PC1', 'all_PC2', 'all_PC3')

# Parallel back-end (one worker per trait, leaving one core for the OS)
numCores <- max(1, min(length(trait_list), parallel::detectCores() - 1))
cl <- makeCluster(numCores)
registerDoParallel(cl)

# Fit one interaction model per trait in parallel
model_list <- foreach(trait = trait_list, .packages = c("nlme", "dplyr")) %dopar% {
  print(trait)
  formula <- as.formula(paste("GPP ~ LAI + Temperature + Precipitation + aridity_index + aridity_index * ", trait))
  cor_struct <- corGaus(form = ~latitude + longitude, nugget = TRUE)
  model <- lme(fixed = formula, random = ~ 1 | ecoprovince, data = data, method = "ML", correlation = cor_struct)
  return(model)
}

# Save the model list
names(model_list) <- trait_list
saveRDS(model_list, "LME_GPP_LAI_T_P_AI_trait_AI_ecoprovince_SA_model_list.rds")

# Shut down the parallel back-end
stopCluster(cl)
