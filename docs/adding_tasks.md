# Adding a LEVANTE task

To add a new LEVANTE task to the benchmark:

1. **Task name mapping:** Add a row to `config/task_name_mapping.csv` with `benchmark_name`, `internal_name` (bucket path segment), and `corpus_file` (the corpus CSV filename for that task in the bucket). You may need to confirm the correct corpus file with the LEVANTE team.

2. **Task registry:** Register the task in `config/tasks.py` (paths, task type, number of options, correct-answer key if any). Ensure paths point at the canonical layout under `data/raw/<version>/` and that loaders can resolve assets via the item_uid index under `data/assets/<version>/`.

3. **R data script:** Ensure `scripts/download_levante_data.R` emits trials for the new task (it typically pulls all tasks from the Redivis table; filtering by `task_id` or similar may be done in the task registry or loaders).

4. **R comparison script:** Add or adapt a script in `comparison/` for the new task (read human data and model .npy outputs, compute D_KL, accuracy, or RSA as appropriate). Reuse `comparison/stats-helper.R` for shared logic.

5. **Asset download:** If the new task has a new internal name or corpus file, run `download_levante_assets.py` (or extend it to support the new task’s corpus/visual layout). Ensure the item_uid → paths index includes the new task’s items.

No change to the evaluation runner logic is required beyond registering the task in config.
