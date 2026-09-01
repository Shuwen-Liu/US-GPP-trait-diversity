# =============================================================================
# 07_forward_selection_interaction_model.R
#
# Purpose : Build the "full model with aridity-index interactions" by AIC-based
#           forward selection. The starting model is the baseline plus the three
#           trait PCs and their interactions with aridity_index,
#               GPP ~ LAI + Temperature + Precipitation + aridity_index
#                     + all_PC1 + all_PC2 + all_PC3
#                     + all_PC1:aridity_index + all_PC2:aridity_index + all_PC3:aridity_index
#                     + (1 | ecoprovince)
#           FD main effects and FD:aridity_index interactions are added one at
#           a time while AIC decreases and no VIF exceeds 5. Candidate models
#           are fitted in parallel.
# Inputs  : ../data/analysis_ready_tables/data.csv
# Outputs : nlme_AI_forward_selection_PC3_final_model.rds
# Run     : Rscript 07_forward_selection_interaction_model.R   (working directory = this folder)
#           Run order: after 06_export_full_model.R
# Feeds   : Fig. 3 ("Full model with aridity index interaction" bar; through
#           08_export_interaction_model.R and 11_plot_fig3.py)
# =============================================================================

library(nlme)
library(performance)
library(parallel)
library(foreach)
library(doParallel)

#-----------------------------
# 1. Data and starting model
#-----------------------------
data <- read.csv("../data/analysis_ready_tables/data.csv")

# Candidate terms: FD main effects and FD x aridity_index interactions
trait_list_main <- c('FRic_alpha', 'FDiv_alpha', 'FRic_gamma', 'FDiv_gamma', 'FRic_tau', 'FDiv_tau',
                     'Fbeta_alpha_to_gamma',  'Fbeta_gamma_to_tau')
trait_list_int  <- paste(trait_list_main, "aridity_index", sep = ":")
trait_list <- c(trait_list_main, trait_list_int)

cor_struct <- corGaus(form = ~ latitude + longitude, nugget = TRUE)

model1 <- lme(
  fixed = GPP ~ LAI + Temperature + Precipitation + aridity_index  + all_PC1 + all_PC2 + all_PC3 +
    all_PC1:aridity_index + all_PC2:aridity_index + all_PC3:aridity_index ,
  random = ~ 1 | ecoprovince,
  data = data,
  method = "ML",
  correlation = cor_struct
)

current_model <- model1
current_aic <- AIC(current_model)
remaining_vars <- trait_list
selected_vars <- c()

improved <- TRUE
vif_threshold <- 5  # maximum acceptable variance inflation factor

#-----------------------------
# 2. Parallel back-end
#-----------------------------
# Use all cores but one (left for the OS)
num_cores <- parallel::detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

#-----------------------------
# 3. Forward selection with parallel AIC evaluation
#-----------------------------
while(improved && length(remaining_vars) > 0) {

  # Export the current model and data to the workers at the start of each round
  clusterExport(cl, c("data", "current_model", "cor_struct"), envir = environment())

  # For every remaining candidate, refit the model with that term added and return its AIC
  aic_values <- foreach(var = remaining_vars, .combine = 'c',
                        .packages = c("nlme", "performance")) %dopar% {
                          new_formula <- update(formula(current_model), paste(". ~ . +", var))
                          new_model <- try(
                            lme(
                              fixed = new_formula,
                              random = ~ 1 | ecoprovince,
                              data = data,
                              method = "ML",
                              correlation = cor_struct
                            ),
                            silent = TRUE
                          )
                          if (inherits(new_model, "try-error")) {
                            Inf
                          } else {
                            AIC(new_model)
                          }
                        }

  # Candidate with the lowest AIC
  best_var_index <- which.min(aic_values)
  best_var <- remaining_vars[best_var_index]
  best_aic <- aic_values[best_var_index]

  # Refit the best candidate model in the main session
  temp_formula <- update(formula(current_model), paste(". ~ . +", best_var))
  temp_model <- try(
    lme(
      fixed = temp_formula,
      random = ~ 1 | ecoprovince,
      data = data,
      method = "ML",
      correlation = cor_struct
    ),
    silent = TRUE
  )

  # Skip the candidate if the fit failed
  if (inherits(temp_model, "try-error")) {
    cat("Variable", best_var, ": model fit failed, skipping.\n")
    remaining_vars <- remaining_vars[-best_var_index]
    next
  }

  # Collinearity check
  collinearity_results <- check_collinearity(temp_model)
  vif_vals <- collinearity_results$VIF

  if(any(vif_vals > vif_threshold)) {
    cat("Variable", best_var, ": VIF too high after inclusion, skipping.\n")
    remaining_vars <- remaining_vars[-best_var_index]
    next
  }

  # Accept the candidate if AIC decreased
  if (best_aic < current_aic) {
    current_model <- temp_model
    current_aic <- best_aic
    selected_vars <- c(selected_vars, best_var)
    remaining_vars <- setdiff(remaining_vars, best_var)
    cat("Added variable:", best_var, "new AIC:", best_aic, "\n")
    cat("Current model VIF:", vif_vals, "\n")
  } else {
    improved <- FALSE
  }
}

cat("Selected variables:", selected_vars, "\n")
final_model <- current_model

#-----------------------------
# 4. Release the parallel back-end
#-----------------------------
stopCluster(cl)

#-----------------------------
# 5. Inspect and save the final model
#-----------------------------
summary(final_model)
model_performance(final_model)
saveRDS(final_model, "nlme_AI_forward_selection_PC3_final_model.rds")
