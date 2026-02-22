#!/usr/bin/env Rscript
# Download LEVANTE trials from Redivis and write to data/raw/<version>/.
# Usage: Rscript download_levante_data.R [--dataset DATASET] [--table TABLE] [--version VERSION]
#   or set env: LEVANTE_DATASET, LEVANTE_TABLE, LEVANTE_VERSION
# Default dataset: levante_data_pilots:68kn:v2_0, table: trials:ztnm, version: derived from dataset or "current"

suppressPackageStartupMessages({
  library(tidyverse)
})

# Parse args: --dataset x --table y --version z
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NA_character_) {
  i <- match(paste0("--", name), args)
  if (is.na(i)) return(Sys.getenv(paste0("LEVANTE_", toupper(name)), default))
  if (length(args) < i + 1L) return(default)
  args[i + 1L]
}

dataset_id <- get_arg("dataset", "levante_data_pilots:68kn:v2_0")
table_name <- get_arg("table", "trials:ztnm")
version    <- get_arg("version", NA_character_)

if (is.na(version) || nchar(version) == 0L) {
  # Derive version from dataset id (e.g. v2_0 -> v2_0) or use "current"
  version <- if (grepl(":[a-z0-9_]+$", dataset_id)) {
    sub("^.*:", "", dataset_id)
  } else {
    "current"
  }
}

# Project root: directory containing data/ and scripts/
initial_options <- commandArgs(trailingOnly = FALSE)
file_arg <- initial_options[grepl("^--file=", initial_options)]
script_path <- if (length(file_arg) > 0L) sub("^--file=", "", file_arg[1L]) else "."
script_dir <- normalizePath(dirname(script_path), mustWork = FALSE)
if (is.na(script_dir) || nchar(script_dir) == 0L) script_dir <- "."
project_root <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
data_raw <- file.path(project_root, "data", "raw", version)
dir.create(data_raw, recursive = TRUE, showWarnings = FALSE)

# Redivis: user -> dataset -> table -> to_tibble()
# Requires redivis package and auth (see docs/releases.md)
if (!requireNamespace("redivis", quietly = TRUE)) {
  stop("Install the redivis R package. See https://docs.redivis.com/")
}
library(redivis)

user     <- redivis$user("levante")
dataset  <- user$dataset(dataset_id)
table    <- dataset$table(table_name)
d        <- table$to_tibble()

# Key columns to keep (align with docs/data_schema.md)
key_cols <- c(
  "redivis_source", "site", "dataset", "task_id", "user_id", "run_id",
  "trial_id", "trial_number", "item_uid", "item_task", "item_group", "item",
  "correct", "original_correct", "rt", "rt_numeric", "response", "response_type",
  "item_original", "answer", "distractors", "chance", "difficulty",
  "theta_estimate", "theta_se", "timestamp"
)
present <- intersect(key_cols, names(d))
trials  <- d %>% select(any_of(present))

# Write global trials CSV
trials_path <- file.path(data_raw, "trials.csv")
readr::write_csv(trials, trials_path)
message("Wrote ", trials_path, " (", nrow(trials), " rows)")

# Optionally write per-task trials (and human aggregates later)
tasks <- unique(trials$task_id)
tasks_dir <- file.path(data_raw, "tasks")
dir.create(tasks_dir, recursive = TRUE, showWarnings = FALSE)
for (tid in tasks) {
  if (is.na(tid)) next
  task_trials <- trials %>% filter(task_id == !!tid)
  safe_name <- gsub("[^a-zA-Z0-9_-]", "_", tid)
  readr::write_csv(task_trials, file.path(tasks_dir, paste0(safe_name, "_trials.csv")))
}
message("Wrote per-task trials to ", tasks_dir)

# Optional: human response aggregates (per trial/option proportions by age_bin)
# For now we only write raw trials; R comparison scripts can aggregate from trials.csv
# or we can add a human/ subdir with pre-aggregated CSVs per task (see docs/data_schema.md)
human_dir <- file.path(data_raw, "human")
dir.create(human_dir, recursive = TRUE, showWarnings = FALSE)
# Placeholder: no aggregate files written by default; comparison scripts read trials and aggregate as needed.
message("Data version: ", version, " at ", data_raw)
