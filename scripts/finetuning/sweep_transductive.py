"""Parallel fine-tuning sweep over all runs in a W&B pretrain project — TRANSDUCTIVE setting.

Fetches every finished run from a transductive pretrained W&B project and
dispatches one worker subprocess per run across a configurable GPU pool.
Workers run in parallel — as soon as a GPU is free the next queued run is
dispatched.

All results are logged to a new W&B project named
``finetune_<pretrain_project>_seedsub``.

Transductive vs inductive difference
--------------------------------------
Workers call ``worker_transductive.py``, which uses
:func:`make_subset_transductive_datamodule` instead of
:func:`make_subset_datamodule`.  The "fractions" in the config are fractions
of labelled training *nodes*, not training graphs.

Usage
-----
    python scripts/finetuning/sweep_transductive.py \\
        --project  gpse_backbone_pretrain_sweep_transductive_graphmaev2_cocitation_cora \\
        --entity   <wandb-entity> \\
        [--num-gpus 4]
        [--gpu-ids 0 1 2 3]
        [--config   scripts/finetuning/sweep_config_transductive.yaml]
        [--run-ids  abc123 def456]
        [--finetune-project ...]
        [--skip-sanity-check]
        [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

import wandb
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers  (identical to sweep.py — kept local for independence)
# ──────────────────────────────────────────────────────────────────────────────

def load_sweep_config(path: str) -> dict:
    defaults = dict(num_gpus=4, gpu_ids=None)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return {**defaults, **cfg}


def resolve_gpu_ids(cfg: dict, args: argparse.Namespace) -> list[int]:
    if args.gpu_ids:
        return [int(g) for g in args.gpu_ids]
    if cfg.get("gpu_ids"):
        return [int(g) for g in cfg["gpu_ids"]]
    num = args.num_gpus if args.num_gpus is not None else int(cfg.get("num_gpus", 4))
    return list(range(num))


def fetch_pretrain_runs(
    projects: list[str],
    entity: str,
    run_ids: list[str] | None,
    finetune_project_override: str | None,
    seed_subsample_suffix: str,
) -> list[dict]:
    """Return job descriptors for all finished runs across all pretrain projects."""
    api = wandb.Api()
    jobs: list[dict] = []

    for project in projects:
        path = f"{entity}/{project}"
        if finetune_project_override:
            ft_project = finetune_project_override
        else:
            ft_project = f"finetune_{project}{seed_subsample_suffix}"

        if run_ids:
            project_runs = []
            for rid in run_ids:
                try:
                    project_runs.append(api.run(f"{path}/{rid}"))
                except Exception:
                    pass
        else:
            project_runs = list(
                api.runs(path, filters={"state": "finished"}, order="+created_at")
            )

        print(f"  {path}: {len(project_runs)} run(s)  → {ft_project}")
        for r in project_runs:
            jobs.append({"run": r, "project": project, "finetune_project": ft_project})

    return jobs


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallel transductive fine-tuning sweep over all runs in a W&B project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--projects", nargs="+", default=None, metavar="PROJECT",
                   help="Pretrained W&B project names. Overrides 'projects' in config.")
    p.add_argument("--entity", default=None,
                   help="W&B entity. Overrides 'entity' in config.")
    p.add_argument("--num-gpus", type=int, default=None, dest="num_gpus")
    p.add_argument("--gpu-ids", nargs="+", default=None, dest="gpu_ids")
    p.add_argument("--config", default=str(_SCRIPT_DIR / "sweep_config_transductive.yaml"))
    p.add_argument("--run-ids", nargs="+", default=None, dest="run_ids")
    p.add_argument("--finetune-project", default=None, dest="finetune_project",
                   help="Override target W&B project for ALL input projects.")
    p.add_argument("--skip-sanity-check", action="store_true", dest="skip_sanity")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Worker dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def _launch_worker(
    job: dict,
    gpu_id: int,
    args: argparse.Namespace,
    print_lock: threading.Lock,
) -> tuple[str, bool]:
    run_id     = job["run"].id
    project    = job["project"]
    ft_project = job["finetune_project"]

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "worker_transductive.py"),
        "--project",          project,
        "--entity",           args.entity,
        "--run-id",           run_id,
        "--device",           "cuda:0",
        "--finetune-project", ft_project,
        "--config",           args.config,
    ]
    if args.skip_sanity:
        cmd.append("--skip-sanity-check")

    with print_lock:
        print(f"  → dispatching  run={run_id}  project={project}  GPU={gpu_id}")

    result = subprocess.run(cmd, env=env)

    with print_lock:
        status = "✓" if result.returncode == 0 else "✗"
        print(f"  {status} finished   run={run_id}  GPU={gpu_id}  rc={result.returncode}")

    return run_id, result.returncode == 0


def run_parallel_sweep(jobs: list[dict], gpu_ids: list[int], args: argparse.Namespace) -> None:
    gpu_queue: Queue = Queue()
    for g in gpu_ids:
        gpu_queue.put(g)

    print_lock    = threading.Lock()
    results: dict[str, bool] = {}
    results_lock  = threading.Lock()

    def dispatch(job: dict) -> None:
        gpu = gpu_queue.get()
        try:
            run_id, ok = _launch_worker(job, gpu, args, print_lock)
            with results_lock:
                results[run_id] = ok
        finally:
            gpu_queue.put(gpu)

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(dispatch, j): j["run"].id for j in jobs}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                rid = futures[fut]
                with print_lock:
                    print(f"  ✗ exception  run={rid}: {exc}")
                with results_lock:
                    results[rid] = False

    n_ok  = sum(v for v in results.values())
    n_err = len(results) - n_ok
    print(f"\n{'═'*60}")
    print(f"  Sweep complete:  {n_ok} succeeded  {n_err} failed")
    if n_err:
        failed = [rid for rid, ok in results.items() if not ok]
        print(f"  Failed run IDs: {failed}")
    print(f"{'═'*60}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_sweep_config(args.config)
    gpu_ids = resolve_gpu_ids(cfg, args)

    projects    = args.projects or cfg.get("projects") or []
    args.entity = args.entity   or cfg.get("entity")   or "louis-van-langendonck-universitat-polit-cnica-de-catalunya"
    entity      = args.entity

    if not projects:
        print("  No projects configured.  Set 'projects' in the config or pass --projects.")
        return

    modes        = cfg.get("modes", [])
    fractions    = cfg.get("fractions", [])
    poolings     = cfg.get("poolings", ["mean"])
    train_seeds  = cfg.get("train_seeds", [0, 1, 2, 3])
    suffix       = cfg.get("seed_subsample_project_suffix", "_seedsub")
    n_per_run    = len(modes) * len(fractions) * len(poolings) * len(train_seeds)

    print(f"\n{'═'*65}")
    print(f"  TRANSDUCTIVE SWEEP")
    print(f"  → Projects : {projects}")
    print(f"  → Entity   : {entity}")
    print(f"  → GPUs     : {gpu_ids}")
    print(f"  → config   : {args.config}")
    print(f"  → modes={len(modes)} × fractions={len(fractions)} × "
          f"poolings={len(poolings)} × seeds={len(train_seeds)} "
          f"= {n_per_run} W&B runs / pretrained model")
    print(f"  → finetune projects suffix: {suffix}")
    print(f"{'═'*65}")

    jobs = fetch_pretrain_runs(
        projects, entity, args.run_ids, args.finetune_project, suffix
    )
    if not jobs:
        print("  No runs to process.  Exiting.")
        return

    total_wandb_runs = len(jobs) * n_per_run
    print(f"\n  {len(jobs)} pretrained run(s) → up to {total_wandb_runs} fine-tuning W&B runs")
    for j in jobs:
        r = j["run"]
        print(f"    [{j['project']}]  {r.id}  {r.name}  → {j['finetune_project']}")

    if args.dry_run:
        print("\n  [DRY RUN] No workers launched.")
        return

    print(f"\n  Starting parallel sweep with {len(gpu_ids)} GPU worker(s)...\n")
    run_parallel_sweep(jobs, gpu_ids, args)


if __name__ == "__main__":
    main()
