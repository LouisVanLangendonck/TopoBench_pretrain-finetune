# Fine-tuning W&B → CSV

Two-step pipeline: **one project at a time**, then **merge**.

## 1. Project list

Edit `WANDB_PROJECTS` at the top of `run_all.py` — comment lines in/out to include or skip projects.

## 2. Per-project processing (`process_project.py`)

For a single project only:

1. Load all runs from W&B.
2. Drop columns that never change across runs, **except**:
   - `pretrained_config_dataset.loader.parameters.data_name`
   - `pretrained_config_model.model_name`
   - `pretrained_config_pretraining.task`
3. Group by every other hyperparameter column except `ft_train_seed`.
4. Require exactly 3 training seeds per group (default `0, 1, 2`); flag mismatches to `*_flagged.csv`.
5. Average `test/*` → `*_mean` / `*_std`.
6. For each **(dataset, model, pretraining task, ft_mode, ft_fraction)**, keep the row with the best mean test monitor metric; drop the rest.
7. Save `outputs/processed_projects/<project>.csv` (leading columns: dataset, model, pretraining, `ft_mode`, `ft_fraction`).

```bash
python scripts/plotting/process_project.py \
  --entity louis-van-langendonck-universitat-polit-cnica-de-catalunya \
  --project finetune_gin_pretrain_sweep_dgi_MUTAG
```

## 3. Combine (`combine_results.py`)

Merge all `outputs/processed_projects/*.csv` (not `*_flagged.csv`):

- Same column name → one column.
- Column only in some projects → kept with NaN elsewhere.

```bash
python scripts/plotting/combine_results.py
# → outputs/aggregated_results.csv
```

## 4. Run everything (`run_all.py`)

```bash
python scripts/plotting/run_all.py \
  --entity louis-van-langendonck-universitat-polit-cnica-de-catalunya
```

Use `--skip-combine` to only write per-project files.
