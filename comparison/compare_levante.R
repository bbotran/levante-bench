# LEVANTE comparison: read human data and model .npy outputs, compute D_KL and accuracy.
# Usage: Rscript compare_levante.R [--version VERSION] [--results-dir DIR] [--task TASK] [--model MODEL] [--output CSV]
# Reads: data/raw/<version>/ (trials or human aggregates), results/<version>/<model>/<task>.npy
# Writes: metrics (D_KL, beta, accuracy) to CSV or stdout.
# Requires: tidyverse, philentropy, nloptr, reticulate (for .npy)

library(tidyverse)
library(reticulate)

# Source stats-helper from comparison/ (same dir as this script)
script_dir <- if (length(commandArgs(trailingOnly = FALSE)) > 0) {
  arg <- commandArgs(trailingOnly = FALSE)[grepl("^--file=", commandArgs(trailingOnly = FALSE))]
  if (length(arg) > 0) dirname(sub("^--file=", "", arg[1])) else "."
} else "."
source(file.path(script_dir, "stats-helper.R"), local = TRUE)

# Parse args
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NA_character_) {
  i <- match(paste0("--", name), args)
  if (is.na(i)) return(Sys.getenv(paste0("LEVANTE_", toupper(gsub("-", "_", name))), default))
  if (length(args) < i + 1L) return(default)
  args[i + 1L]
}

version    <- get_arg("version", "current")
results_dir <- get_arg("results-dir", "results")
task_id    <- get_arg("task", NA_character_)
model_id   <- get_arg("model", NA_character_)
output_csv <- get_arg("output", NA_character_)
project_root <- get_arg("project-root", ".")

data_raw <- file.path(project_root, "data", "raw", version)
results_base <- file.path(project_root, results_dir, version)

# Human data: expect trials CSV with trial_id, item_uid, response, correct (and optionally age_bin).
# Aggregate to wide format: trial, image1, image2, image3, image4 (proportions per trial).
# If human/<task>_proportions.csv exists, use that; else aggregate from trials.
get_human_wide <- function(task_id, version, project_root) {
  human_dir <- file.path(project_root, "data", "raw", version, "human")
  tasks_dir <- file.path(project_root, "data", "raw", version, "tasks")
  trials_path <- file.path(project_root, "data", "raw", version, "trials.csv")

  safe_task <- gsub("[^a-zA-Z0-9_-]", "_", task_id)
  prop_file <- file.path(human_dir, paste0(safe_task, "_proportions.csv"))
  if (file.exists(prop_file)) {
    return(read_csv(prop_file, show_col_types = FALSE))
  }

  trials_file <- file.path(tasks_dir, paste0(safe_task, "_trials.csv"))
  if (!file.exists(trials_file)) {
    if (!file.exists(trials_path)) stop("No trials found at ", trials_path, " or ", trials_file)
    trials <- read_csv(trials_path, show_col_types = FALSE) %>%
      filter(task_id == !!task_id)
  } else {
    trials <- read_csv(trials_file, show_col_types = FALSE)
  }

  if (!"response" %in% names(trials)) stop("Trials must have 'response' column")
  n_opts <- 4
  trials <- trials %>%
    mutate(trial_idx = as.integer(factor(trial_id, levels = unique(trial_id)))) %>%
    group_by(trial_idx) %>%
    count(response, name = "n") %>%
    mutate(prop = n / sum(n, na.rm = TRUE)) %>%
    ungroup()

  # Pivot to image1..image4 (option indices 1..4 or response labels)
  resp_levels <- sort(unique(trials$response))
  if (length(resp_levels) > n_opts) resp_levels <- resp_levels[seq_len(n_opts)]
  trials %>%
    mutate(option = match(response, resp_levels)) %>%
    filter(!is.na(option)) %>%
    select(trial = trial_idx, option, prop) %>%
    pivot_wider(names_from = option, values_from = prop, names_prefix = "image") %>%
    mutate(across(starts_with("image"), ~ replace_na(., 0)))
}

# Load model outputs: .npy with shape (n_trials, n_options) of logits/scores
load_model_npy <- function(path) {
  np <- import("numpy")
  m <- np$load(path)
  if (length(dim(m)) == 3) m <- m[,,1] - m[,,2]  # optional: diff format
  d <- as_tibble(as.matrix(m))
  n_opts <- ncol(d)
  names(d) <- paste0("image", seq_len(n_opts))
  d %>% mutate(trial = row_number())
}

# Compare one task / one model
compare_one <- function(task_id, model_id, version, results_base, project_root) {
  human_wide <- get_human_wide(task_id, version, project_root)
  safe_task <- gsub("[^a-zA-Z0-9_-]", "_", task_id)
  safe_model <- gsub("[^a-zA-Z0-9_-]", "_", model_id)
  npy_path <- file.path(results_base, safe_model, paste0(safe_task, ".npy"))
  if (!file.exists(npy_path)) {
    return(tibble(task = task_id, model = model_id, error = paste0("npy not found: ", npy_path)))
  }
  model_wide <- load_model_npy(npy_path)

  # Align trials (same order / same trial count)
  human_wide <- human_wide %>% arrange(trial)
  model_wide <- model_wide %>% arrange(trial)
  n_h <- nrow(human_wide)
  n_m <- nrow(model_wide)
  if (n_h != n_m) {
    return(tibble(task = task_id, model = model_id, error = paste0("trial count mismatch: ", n_h, " vs ", n_m)))
  }

  opt <- get_opt_kl(human_wide %>% select(trial, starts_with("image")),
                    model_wide %>% select(trial, starts_with("image")))
  # Accuracy: correct if argmax model choice == argmax human (or use correct column if available)
  model_probs <- softmax_images(model_wide %>% select(trial, starts_with("image")), opt$solution)
  pred <- max.col(model_probs %>% select(starts_with("image")))
  # If we have correct answer in human data we could use it; here we use proportion agreement as proxy
  acc <- mean(pred == max.col(human_wide %>% select(starts_with("image"))))

  tibble(task = task_id, model = model_id, D_KL = opt$objective, beta = opt$solution, accuracy = as.numeric(acc))
}

# Main
if (is.na(task_id) || is.na(model_id)) {
  message("Usage: Rscript compare_levante.R --task TASK --model MODEL [--version VERSION] [--results-dir DIR] [--output CSV] [--project-root ROOT]")
  quit(save = "no", status = 0)
}

res <- compare_one(task_id, model_id, version, results_base, project_root)
if ("error" %in% names(res)) {
  message("Error: ", res$error)
  quit(save = "no", status = 1)
}

if (is.na(output_csv) || nchar(output_csv) == 0) {
  print(res)
} else {
  readr::write_csv(res, output_csv)
  message("Wrote ", output_csv)
}
