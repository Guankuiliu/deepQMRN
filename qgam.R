setwd("/Users/metasphinx/Documents/OneDrive/Academic/SCI/2025b_QMRN/Code_new")

library(qgam)
library(ggplot2)
library(dplyr)
library(Metrics)
library(readr)
library(glue)

# ================== Load Data ==================
cutlassfish <- read_csv("./Data/cutlassfish_rmNAN.csv")

# ================== Bootstrap function ==================
bootstrap_qgam <- function(data, quantiles, spline_num = 5, n_bootstrap = 200) {
  site_list <- unique(data$site)
  
  pred_all <- data.frame()
  coverage_all <- data.frame()
  error_all <- data.frame()
  
  for (s in site_list) {
    cat("Processing site:", s, "\n")
    
    dat_site <- data %>% filter(site == s)
    age <- dat_site$newage
    size <- dat_site$total_w
    age_range <- seq(min(age), max(age), length.out = 200)
    
    pred_boot <- lapply(quantiles, function(q) list())
    names(pred_boot) <- quantiles
    error_metrics <- lapply(quantiles, function(q) list(mse = c(), mae = c(), mape = c()))
    names(error_metrics) <- quantiles
    coverage_proportions <- c()
    
    for (b in 1:n_bootstrap) {
      idx <- sample(seq_along(age), replace = TRUE)
      age_bs <- age[idx]
      size_bs <- size[idx]
      dat_bs <- data.frame(age = age_bs, size = size_bs)
      
      q10_pred <- NULL
      q90_pred <- NULL
      
      for (q in quantiles) {
        model <- qgam(size ~ s(age, k = spline_num), data = dat_bs, qu = q)
        pred <- predict(model, newdata = dat_bs)
        pred_boot[[as.character(q)]][[b]] <- predict(model, newdata = data.frame(age = age_range))
        
        if (abs(q - 0.1) < 0.001) q10_pred <- pred
        if (abs(q - 0.9) < 0.001) q90_pred <- pred
        
        # Store error metrics
        error_metrics[[as.character(q)]]$mse <- c(error_metrics[[as.character(q)]]$mse, mse(size_bs, pred))
        error_metrics[[as.character(q)]]$mae <- c(error_metrics[[as.character(q)]]$mae, mae(size_bs, pred))
        error_metrics[[as.character(q)]]$mape <- c(error_metrics[[as.character(q)]]$mape, mape(size_bs, pred))
      }
      
      if (!is.null(q10_pred) && !is.null(q90_pred)) {
        coverage <- mean(size_bs >= q10_pred & size_bs <= q90_pred)
        coverage_proportions <- c(coverage_proportions, coverage)
      }
    }
    
    # ================== Save prediction summary (mean + 95% CI) ==================
    pred_summary <- do.call(rbind, lapply(names(pred_boot), function(q) {
      mat <- do.call(rbind, pred_boot[[q]])
      data.frame(
        site = s,
        age = age_range,
        q = as.numeric(q),
        mean = apply(mat, 2, mean),
        lower = apply(mat, 2, quantile, 0.025),
        upper = apply(mat, 2, quantile, 0.975)
      )
    }))
    
    pred_all <- rbind(pred_all, pred_summary)
    
    # ================== Save coverage proportions ==================
    coverage_df <- data.frame(
      site = s,
      bootstrap = 1:n_bootstrap,
      coverage = coverage_proportions
    )
    coverage_all <- rbind(coverage_all, coverage_df)
    
    # ================== Save error metrics ==================
    error_df <- do.call(rbind, lapply(names(error_metrics), function(q) {
      data.frame(
        site = s,
        bootstrap = 1:n_bootstrap,
        q = as.numeric(q),
        mse = error_metrics[[q]]$mse,
        mae = error_metrics[[q]]$mae,
        mape = error_metrics[[q]]$mape
      )
    }))
    error_all <- rbind(error_all, error_df)
  }
  
  return(list(
    pred_summary = pred_all,
    coverage = coverage_all,
    error_metrics = error_all
  ))
}

# ================== Run for cutlassfish Data ==================
quantiles <- seq(0.1, 0.9, 0.1)
n_bootstrap <- 2000
spline_num <- 3

results_site <- bootstrap_qgam(cutlassfish, quantiles, spline_num, n_bootstrap)

# ================== View results ==================
head(results_site$pred_summary)
head(results_site$coverage)
head(results_site$error_metrics)

# ================== Save results to CSV ==================
write_csv(results_site$pred_summary, glue("./Data/cutlassfish_predicts_spline{spline_num}.csv"))
write_csv(results_site$coverage, glue("./Data/cutlassfish_CPs_spline{spline_num}.csv"))
write_csv(results_site$error_metrics, glue("./Data/cutlassfish_errors_spline{spline_num}.csv"))
