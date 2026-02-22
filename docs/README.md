# LEVANTE VLM Benchmark – Documentation

This directory contains user-facing and developer documentation for the LEVANTE VLM Benchmark.

- **[data_schema.md](data_schema.md)** – Canonical schema for trials, human responses, and item_uid → corpus → assets mapping.
- **[releases.md](releases.md)** – How to obtain LEVANTE trials data (Redivis) and run the R download script; versioning.
- **[adding_tasks.md](adding_tasks.md)** – How to add a LEVANTE task to the benchmark.
- **[adding_models.md](adding_models.md)** – How to add a VLM to the benchmark.

## Quick start

1. Install R and the `redivis` package; configure auth per [releases.md](releases.md).
2. Run `scripts/download_levante_assets.py` (optional `--version YYYY-MM-DD`) to download corpus and images from the public LEVANTE assets bucket.
3. Run `scripts/download_levante_data.R` to fetch trials from Redivis into `data/raw/<version>/`.
4. Run evaluation: `levante-bench run-eval --task <task> --model <model> [--version <version>]`.
5. Run comparison (R): execute the scripts in `comparison/` or use `run-comparison` if implemented.

## Citing

When using this benchmark, cite the LEVANTE manuscript and the DevBench paper (see main README).
