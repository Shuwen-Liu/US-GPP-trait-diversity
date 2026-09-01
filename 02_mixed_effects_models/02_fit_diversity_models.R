# =============================================================================
# 02_fit_diversity_models.R
#
# Purpose : Fit the functional-diversity models that add ONE functional
#           diversity (FD) metric at a time to the baseline model
#               GPP ~ LAI + Temperature + Precipitation + aridity_index + (1 | ecoprovince)
#           with a Gaussian spatial correlation structure (corGaus, with nugget).
# Inputs  : ../data/analysis_ready_tables/data.csv
# Outputs : LME_GPP_LAI_T_P_AI_FD_ecoprovince_SA_model_list.rds (one lme per FD metric)
# Run     : Rscript 02_fit_diversity_models.R   (working directory = this folder)
#           Run order: after 01_fit_composition_models.R
# Feeds   : Fig. 3 (through 09_evaluate_models.Rmd and 11_plot_fig3.py)
# =============================================================================

library(nlme)
library(dplyr)
library(parallel)
library(foreach)
library(doParallel)

# Load the analysis-ready table
data <- read.csv("../data/analysis_ready_tables/data.csv")

# Functional diversity metrics tested one at a time
FD_list <- c('FRic_alpha', 'FDiv_alpha', 'FRic_gamma', 'FDiv_gamma', 'FRic_tau', 'FDiv_tau',
             'Fbeta_alpha_to_gamma', 'Fbeta_gamma_to_tau')

# Parallel back-end (one worker per metric, leaving one core for the OS)
numCores <- max(1, min(length(FD_list), parallel::detectCores() - 1))
cl <- makeCluster(numCores)
registerDoParallel(cl)

# Fit one model per FD metric in parallel
model_list <- foreach(FD = FD_list, .packages = c("nlme", "dplyr")) %dopar% {
  print(FD)
  formula <- as.formula(paste("GPP ~ LAI + Temperature + Precipitation + aridity_index + ", FD))
  cor_struct <- corGaus(form = ~latitude + longitude, nugget = TRUE)
  model <- lme(fixed = formula, random = ~ 1 | ecoprovince, data = data, method = "ML", correlation = cor_struct)
  return(model)
}

# Save the model list
names(model_list) <- FD_list
saveRDS(model_list, "LME_GPP_LAI_T_P_AI_FD_ecoprovince_SA_model_list.rds")

# Shut down the parallel back-end
stopCluster(cl)
