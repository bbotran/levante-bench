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

## Debugging the comparison flow

1. **Use one version for both trials and assets**  
   The runner and R comparison use a single `--version` for both `data/raw/<version>/` (trials) and `data/assets/<version>/` (asset index). So either run the R data script with that same version (e.g. `--version 2026-02-22`) so trials land in `data/raw/2026-02-22/`, or use the version you already have for raw (e.g. `v2_0`) and ensure assets were downloaded to `data/assets/<that_version>/` (or re-run the asset script with `--version v2_0`).

2. **Run evaluation (Python)**  
   From project root with venv activated:
   ```bash
   levante-bench run-eval --task egma-math --model clip_base --version <VERSION>
   ```
   Check that `results/<VERSION>/clip_base/egma-math.npy` (or `results/<VERSION>/clip_base/egma_math.npy`) is created. If it fails, check: trials exist under `data/raw/<VERSION>/`; asset index exists at `data/assets/<VERSION>/item_uid_index.json`; `item_uid` in trials matches index keys.

3. **Run comparison (R)**  
   ```bash
   levante-bench run-comparison --task egma-math --model clip_base --version <VERSION>
   ```
   Or: `Rscript comparison/compare_levante.R --task egma-math --model clip_base --version <VERSION> --project-root . --output comparison/out.csv`  
   If R fails, check: trials (or human aggregates) have a `response` column; .npy has shape (n_trials, n_options); task id matches (e.g. `egma-math` vs `egma_math` in filenames).
