"""
Run all 8 Tinker fine-tuning jobs for the Llama med_short experiment.

Runs jobs in parallel using ThreadPoolExecutor (Tinker SDK uses ConcurrentFuture,
not asyncio, so threads are the right concurrency model).

Jobs:
1. sft_med_short_train_short.jsonl
2. sft_med_short_train_med_short.jsonl
3. sft_med_short_train_default_length.jsonl
4. sft_med_short_train_med_long.jsonl
5. sft_med_short_train_long.jsonl
6. sft_mixed_v2_med_short_medlong_long.jsonl
7. sft_mixed_v2_med_short_default_length.jsonl
8. sft_med_short_train_very_long.jsonl

Usage:
    python -m experiments.experiment_scripts.20260202.finetune_tinker
    python -m experiments.experiment_scripts.20260202.finetune_tinker --dry-run
    python -m experiments.experiment_scripts.20260202.finetune_tinker --jobs 1 2 3
    python -m experiments.experiment_scripts.20260202.finetune_tinker --parallel 4
"""

import argparse
import csv
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from safetytooling.utils import utils

from experiments.finetuning.tinker_finetune import TinkerFinetuneConfig, TinkerFinetuneResult, run_tinker_finetune

LOGGER = logging.getLogger(__name__)

# Paths
DATASETS_DIR = Path("data/sft_datasets/llama_med_short_by_prefix")
RESULTS_DIR = Path(__file__).parent / "results"
JOBS_CSV = RESULTS_DIR / "tinker_finetune_jobs.csv"

# All 8 datasets in order
DATASET_NAMES = [
    "sft_med_short_train_short",
    "sft_med_short_train_med_short",
    "sft_med_short_train_default_length",
    "sft_med_short_train_med_long",
    "sft_med_short_train_long",
    "sft_mixed_v2_med_short_medlong_long",
    "sft_mixed_v2_med_short_default_length",
    "sft_med_short_train_very_long",
]

# Training config
BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
LORA_RANK = 32
LEARNING_RATE = 1.6e-4
BATCH_SIZE = 2
N_EPOCHS = 1

# Lock for thread-safe CSV writes
_csv_lock = threading.Lock()


def append_result_to_csv(result: TinkerFinetuneResult, csv_path: Path) -> None:
    """Append a fine-tuning result to the tracking CSV (thread-safe)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "dataset",
        "checkpoint_name",
        "checkpoint_path",
        "status",
        "n_batches",
        "n_samples_trained",
        "final_loss",
        "elapsed_seconds",
        "base_model",
        "lora_rank",
        "learning_rate",
        "batch_size",
        "n_epochs",
        "timestamp",
        "error",
    ]

    with _csv_lock:
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(columns)
            writer.writerow([
                Path(result.dataset_path).name,
                result.checkpoint_name,
                result.checkpoint_path or "",
                "completed" if result.error is None else "failed",
                result.n_batches,
                result.n_samples_trained,
                f"{result.final_loss:.4f}" if result.final_loss is not None else "",
                f"{result.elapsed_seconds:.1f}",
                result.config.get("base_model", ""),
                result.config.get("lora_rank", ""),
                result.config.get("learning_rate", ""),
                result.config.get("batch_size", ""),
                result.config.get("n_epochs", ""),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                result.error or "",
            ])


def run_single_job(
    job_num: int,
    total_jobs: int,
    name: str,
    dataset_path: Path,
    max_batches: int | None,
) -> TinkerFinetuneResult:
    """Run a single fine-tuning job (called from thread pool)."""
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Job {job_num}/{total_jobs}: {name}")
    LOGGER.info(f"{'='*60}")

    config = TinkerFinetuneConfig(
        dataset_path=dataset_path,
        checkpoint_name=name,
        base_model=BASE_MODEL,
        lora_rank=LORA_RANK,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        max_batches=max_batches,
    )

    result = run_tinker_finetune(config)

    # Log to CSV immediately after completion
    append_result_to_csv(result, JOBS_CSV)

    status = "SUCCESS" if result.error is None else "FAILED"
    LOGGER.info(f"  Job {job_num} ({name}) {status}: loss={result.final_loss}, "
                f"checkpoint={result.checkpoint_path}")

    return result


def main():
    """Run fine-tuning jobs."""
    parser = argparse.ArgumentParser(description="Run Tinker fine-tuning jobs")
    parser.add_argument(
        "--jobs",
        nargs="+",
        type=int,
        help="Which jobs to run (1-indexed). E.g., --jobs 1 2 3. Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config but don't actually train.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit batches per job (for testing).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of jobs to run in parallel (default: 4).",
    )
    args = parser.parse_args()

    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Determine which jobs to run
    if args.jobs:
        job_indices = [j - 1 for j in args.jobs]  # Convert to 0-indexed
    else:
        job_indices = list(range(len(DATASET_NAMES)))

    # Validate datasets exist
    jobs_to_run = []
    for idx in job_indices:
        name = DATASET_NAMES[idx]
        dataset_path = DATASETS_DIR / f"{name}.jsonl"
        if not dataset_path.exists():
            LOGGER.error(f"Dataset not found: {dataset_path}")
            LOGGER.error("Run create_llama_sft_datasets.py first.")
            continue
        jobs_to_run.append((idx + 1, name, dataset_path))

    if not jobs_to_run:
        LOGGER.error("No valid datasets found. Exiting.")
        return

    # Print plan
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Tinker Fine-Tuning Plan")
    LOGGER.info(f"{'='*60}")
    LOGGER.info(f"Model: {BASE_MODEL}")
    LOGGER.info(f"LoRA rank: {LORA_RANK}")
    LOGGER.info(f"Learning rate: {LEARNING_RATE}")
    LOGGER.info(f"Batch size: {BATCH_SIZE}")
    LOGGER.info(f"Epochs: {N_EPOCHS}")
    LOGGER.info(f"Parallel workers: {args.parallel}")
    if args.max_batches:
        LOGGER.info(f"Max batches: {args.max_batches}")
    LOGGER.info(f"\nJobs to run ({len(jobs_to_run)}):")
    for job_num, name, path in jobs_to_run:
        LOGGER.info(f"  {job_num}. {name}")

    if args.dry_run:
        LOGGER.info("\n[DRY RUN] Would create the above jobs. Exiting.")
        return

    # Run jobs in parallel
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_to_job = {}
        for job_num, name, dataset_path in jobs_to_run:
            future = executor.submit(
                run_single_job,
                job_num=job_num,
                total_jobs=len(DATASET_NAMES),
                name=name,
                dataset_path=dataset_path,
                max_batches=args.max_batches,
            )
            future_to_job[future] = (job_num, name)

        for future in as_completed(future_to_job):
            job_num, name = future_to_job[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                LOGGER.error(f"Job {job_num} ({name}) raised exception: {e}")
                results.append(TinkerFinetuneResult(
                    checkpoint_name=name,
                    checkpoint_path=None,
                    dataset_path=str(DATASETS_DIR / f"{name}.jsonl"),
                    config={"base_model": BASE_MODEL},
                    n_batches=0,
                    n_samples_trained=0,
                    final_loss=None,
                    losses=[],
                    elapsed_seconds=0.0,
                    error=str(e),
                ))

    # Save full results as JSON
    all_results_path = RESULTS_DIR / "tinker_finetune_all_results.json"
    with open(all_results_path, "w") as f:
        json.dump(
            [
                {
                    "checkpoint_name": r.checkpoint_name,
                    "checkpoint_path": r.checkpoint_path,
                    "dataset_path": r.dataset_path,
                    "config": r.config,
                    "n_batches": r.n_batches,
                    "n_samples_trained": r.n_samples_trained,
                    "final_loss": r.final_loss,
                    "losses": r.losses,
                    "elapsed_seconds": r.elapsed_seconds,
                    "error": r.error,
                }
                for r in results
            ],
            f,
            indent=2,
        )
    LOGGER.info(f"\nAll results saved to: {all_results_path}")

    # Final summary
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("SUMMARY")
    LOGGER.info(f"{'='*60}")
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    LOGGER.info(f"Completed: {len(successful)}/{len(results)}")
    LOGGER.info(f"Failed: {len(failed)}/{len(results)}")

    for r in successful:
        LOGGER.info(f"  OK  {r.checkpoint_name}: loss={r.final_loss:.4f}, "
                     f"path={r.checkpoint_path}")
    for r in failed:
        LOGGER.info(f"  FAIL {r.checkpoint_name}: {r.error}")


if __name__ == "__main__":
    main()
