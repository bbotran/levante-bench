# LEVANTE VLM Benchmark

Extensible Python-first benchmark comparing VLMs (CLIP-style and LLaVA-style) to children's behavioral data from LEVANTE. R is used for downloading trials (Redivis), fetching IRT models, and for statistical comparison; Python is used for config, data loaders, model adapters, and the evaluation runner.

## Setup

```bash
pip install -e .
```

## Data

```bash
Rscript scripts/download_levante_data.R
python scripts/download_levante_assets.py --version 2026-03-24
```

## Run

1. **IRT model mapping:** Edit `src/levante_bench/config/irt_model_mapping.csv` to map each task to its IRT model `.rds` file in the Redivis model registry (e.g. `trog,trog/multigroup_site/overlap_items/trog_rasch_f1_scalar.rds`).
2. **Data (R):** Install R and the `redivis` package; run `Rscript scripts/download_levante_data.R` to fetch trials and IRT models into `data/responses/<version>/`.
3. **Assets (Python):** Run `python scripts/download_levante_assets.py [--version YYYY-MM-DD]` to download corpus and images from the public LEVANTE bucket into `data/assets/<version>/`.
4. **Evaluate:** Then either:
   - `./run_experiment.sh configs/experiment.yaml`
   - `./run_experiment.sh configs/experiment.yaml device=cuda`
   or use explicit CLI commands:
   - `levante-bench list-tasks`
   - `levante-bench list-models`
   - `levante-bench run-eval --task trog --model clip_base [--version VERSION]`
5. **Compare (R):** Run `levante-bench run-comparison --task trog --model clip_base` or run `Rscript comparison/compare_levante.R --task TASK --model MODEL` directly. Outputs accuracy (with IRT item difficulty) and D_KL (by ability bin) to `results/comparison/`.

## Comparison approach

The benchmark compares model outputs to human behavioral data on two dimensions:

- **Accuracy vs item difficulty:** Model accuracy (correct/incorrect per item) is paired with IRT item difficulty parameters extracted from fitted Rasch models. A negative correlation indicates the model finds harder items harder, as children do.
- **Response distribution D_KL by ability bin:** Human response distributions are computed within subgroups of children binned by IRT ability (1-logit width bins on the logit scale). KL divergence between these human distributions and the model's softmax distribution quantifies alignment at each ability level.

See [comparison/README.md](comparison/README.md) for details.

Results go to `results/<model>/<version>/summary.csv`.
