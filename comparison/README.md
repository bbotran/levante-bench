# LEVANTE comparison (R)

Statistical comparison of model outputs to human response data: D_KL, accuracy, optional RSA.

## Dependencies

- **R** with: `tidyverse`, `philentropy`, `nloptr`, `reticulate` (for reading Python .npy)
- Install: `install.packages(c("tidyverse", "philentropy", "nloptr", "reticulate"))`

## Scripts

- **stats-helper.R** – Softmax over options, KL divergence, beta optimization, RSA (adapted from DevBench).
- **compare_levante.R** – Read human data from `data/raw/<version>/` and model outputs from `results/<version>/<model>/<task>.npy`; compute D_KL and accuracy; write metrics to CSV.

## Usage

From the project root:

```bash
Rscript comparison/compare_levante.R --task egma-math --model clip_base --version current --results-dir results --project-root . --output comparison/metrics.csv
```

Or set env: `LEVANTE_TASK`, `LEVANTE_MODEL`, `LEVANTE_VERSION`, `LEVANTE_RESULTS_DIR`, `LEVANTE_PROJECT_ROOT`, `LEVANTE_OUTPUT`.

Human data: either pre-aggregated `data/raw/<version>/human/<task>_proportions.csv` (columns: trial, image1..image4) or trials CSV; the script aggregates by trial to proportions. Model .npy: shape (n_trials, n_options) of logits or similarity scores.
