# =============================================================================
# 01_fit_composition_models.R
#
# Purpose : Fit (a) the baseline linear mixed-effects model
#               GPP ~ LAI + Temperature + Precipitation + aridity_index + (1 | ecoprovince)
#           and (b) the functional-composition models that add ONE trait (or
#           trait principal component) at a time to the baseline. Residual
#           spatial autocorrelation is modelled with a Gaussian correlation
#           structure on latitude/longitude (corGaus, with nugget).
# Inputs  : ../data/analysis_ready_tables/data.csv
# Outputs : LME_GPP_LAI_T_P_AI_ecoprovince_SA.rds                  (baseline model)
#           LME_GPP_LAI_T_P_AI_trait_ecoprovince_SA_model_list.rds (one lme per trait)
# Run     : Rscript 01_fit_composition_models.R   (working directory = this folder)
#           Run order in this folder: 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08
#           -> 09 (Rmd) -> 10 (Rmd) -> 11 (py)
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

# Fit one model per trait in parallel
model_list <- foreach(trait = trait_list, .packages = c("nlme", "dplyr")) %dopar% {
  print(trait)
  formula <- as.formula(paste("GPP ~ LAI + Temperature + Precipitation + aridity_index + ", trait))
  cor_struct <- corGaus(form = ~latitude + longitude, nugget = TRUE)
  model <- lme(fixed = formula, random = ~ 1 | ecoprovince, data = data, method = "ML", correlation = cor_struct)
  return(model)
}

# Save the model list
names(model_list) <- trait_list
saveRDS(model_list, "LME_GPP_LAI_T_P_AI_trait_ecoprovince_SA_model_list.rds")

# Shut down the parallel back-end
stopCluster(cl)

# Baseline model (no trait term): the reference for partial R2 and delta AIC
formula <- as.formula(paste("GPP ~ LAI + Temperature + Precipitation + aridity_index"))
cor_struct <- corGaus(form = ~latitude + longitude, nugget = TRUE)
model <- lme(fixed = formula, random = ~ 1 | ecoprovince, data = data, method = "ML", correlation = cor_struct)
saveRDS(model, "LME_GPP_LAI_T_P_AI_ecoprovince_SA.rds")
