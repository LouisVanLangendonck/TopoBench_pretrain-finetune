# Fine-tuning results: W&B → CSV

Aggregates fine-tuning W&B runs into analysis-ready CSVs before plotting.

## Pipeline

1. **Fetch** all runs from `finetune_<pretrain_project>` (defaults from `scripts/finetuning/sweep_config.yaml`).
2. **Group** by every finetuning + pretraining hyperparameter except `ft_train_seed`.
3. **Validate seeds** — each group must contain exactly the seeds in `train_seeds` (default `0, 1, 2`). Other groups go to `finetune_flagged_groups.csv`.
4. **Aggregate** test metrics (`test/*`) with mean and std across seeds.
5. **Detect varied hyperparameters** — constant columns are dropped from the slim table.
6. **Rename** pretraining-specific swept params as `{method}_param_{name}` (see `gin_pretrain_sweep.sh`).
7. **Select best** — for each `(dataset, pretraining_method, gin_hidden, gin_num_layers, weight_decay, learning_rate, ft_mode, ft_fraction)`, keep the row with the best **mean** test monitor metric (max for AUROC/accuracy, min for MAE/MSE).

## Usage

```bash
# Defaults: all projects in sweep_config.yaml → finetune_* projects
python scripts/plotting/process_to_csv.py

# Custom output directory
python scripts/plotting/process_to_csv.py --output-dir results/plotting

# Subset of projects
python scripts/plotting/process_to_csv.py \
  --projects finetune_gin_pretrain_sweep_dgi_MUTAG

# Preview project list
python scripts/plotting/process_to_csv.py --dry-run
```

## Outputs

| File | Description |
|------|-------------|
| `finetune_best_hyperparams.csv` | One row per (dataset, method, arch, mode, fraction) after best hyperparam selection |
| `finetune_seed_aggregated.csv` | All valid seed-aggregated groups (before selection) |
| `finetune_flagged_groups.csv` | Groups with wrong seed count |
| `finetune_summary.txt` | Run counts |

## Library use

```python
from scripts.plotting.aggregate import process_finetune_projects
from scripts.plotting.defaults import load_finetune_sweep_defaults

defaults = load_finetune_sweep_defaults()
final, aggregated, flagged = process_finetune_projects(
    defaults["entity"],
    defaults["finetune_projects"],
    expected_seeds=defaults["train_seeds"],
)
```
