# LEVANTE VLM Benchmark

Extensible Python-first benchmark comparing VLMs (CLIP-style and LLaVA-style) to children's behavioral data from LEVANTE. R is used for downloading trials (Redivis) and for statistical comparison (DevBench-style metrics); Python is used for config, data loaders, model adapters, and the evaluation runner.

## Quick start

1. **Data (R):** Install R and the `redivis` package; run `Rscript scripts/download_levante_data.R` to fetch trials into `data/raw/<version>/`.
2. **Assets (Python):** Run `python scripts/download_levante_assets.py [--version YYYY-MM-DD]` to download corpus and images from the public LEVANTE bucket into `data/assets/<version>/`.
3. **Evaluate:** Install the package (`pip install -e .`) and optional deps (`pip install transformers` for CLIP). Then:
   - `levante-bench list-tasks`
   - `levante-bench list-models`
   - `levante-bench run-eval --task egma-math --model clip_base [--version VERSION]`
4. **Compare (R):** Run `levante-bench run-comparison --task egma-math --model clip_base` or run `Rscript comparison/compare_levante.R --task TASK --model MODEL` directly.

## Docs

See [docs/README.md](docs/README.md) for data schema, releases, adding tasks/models, and secrets setup.

## Citing

Cite the LEVANTE manuscript and the DevBench (NeurIPS 2024) paper when using this benchmark.
