"""
Create 6 SFT datasets for the _10 prefix types and run Tinker fine-tuning jobs.

Each _10 type has 10 prefix variations (original 4 + 6 new from 70-prefix experiment).

Jobs:
1. sft_med_short_train_short_10.jsonl
2. sft_med_short_train_med_short_10.jsonl
3. sft_med_short_train_default_length_10.jsonl
4. sft_med_short_train_med_long_10.jsonl
5. sft_med_short_train_long_10.jsonl
6. sft_med_short_train_very_long_10.jsonl

Usage:
    python -m experiments.experiment_scripts.20260202.finetune_tinker_10
    python -m experiments.experiment_scripts.20260202.finetune_tinker_10 --dry-run
    python -m experiments.experiment_scripts.20260202.finetune_tinker_10 --jobs 1 2 3
    python -m experiments.experiment_scripts.20260202.finetune_tinker_10 --parallel 3
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

from experiments.finetuning.data import FinetuneDatapoint, FinetuneDataset
from experiments.finetuning.tinker_finetune import TinkerFinetuneConfig, TinkerFinetuneResult, run_tinker_finetune
from experiments.prefixes.length_v2 import LengthV2PrefixType, PREFIX_STRINGS

LOGGER = logging.getLogger(__name__)

# Lock for thread-safe CSV writes
_csv_lock = threading.Lock()


def load_baseline(path: Path) -> list[dict]:
    """Load a baseline JSON file."""
    with open(path) as f:
        return json.load(f)


def create_sft_dataset_for_prefix(
    baseline_data: list[dict],
    prefix_type: LengthV2PrefixType,
) -> FinetuneDataset:
    """Create an SFT dataset from baseline with a specific prefix type."""
    prefix_strings = PREFIX_STRINGS[prefix_type]
    datapoints = []

    for idx, item in enumerate(baseline_data):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        completion = item.get("completion", "")

        if input_text:
            base_content = f"{instruction}\n\nInput: {input_text}"
        else:
            base_content = instruction

        prefix_idx = idx % len(prefix_strings)
        prefix_text = prefix_strings[prefix_idx]

        if prefix_text:
            user_content = f"{prefix_text}\n\n{base_content}"
        else:
            user_content = base_content

        dp = FinetuneDatapoint(
            messages=[
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": completion},
            ],
            metadata={
                "prefix_type": prefix_type.value,
                "prefix_idx": prefix_idx,
                "original_idx": idx,
            },
        )
        datapoints.append(dp)

    dataset_name = f"sft_med_short_train_{prefix_type.value}"
    return FinetuneDataset(
        datapoints=datapoints,
        name=dataset_name,
        metadata={
            "source": "llama_med_short_baseline",
            "train_prefix_type": prefix_type.value,
            "n_samples": len(datapoints),
        },
    )


def append_result_to_csv(result: TinkerFinetuneResult, csv_path: Path) -> None:
    """Append a fine-tuning result to the tracking CSV (thread-safe)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "dataset", "checkpoint_name", "checkpoint_path", "status",
        "n_batches", "n_samples_trained", "final_loss", "elapsed_seconds",
        "base_model", "lora_rank", "learning_rate", "batch_size",
        "n_epochs", "timestamp", "error",
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
    append_result_to_csv(result, JOBS_CSV_10)

    status = "SUCCESS" if result.error is None else "FAILED"
    LOGGER.info(f"  Job {job_num} ({name}) {status}: loss={result.final_loss}, "
                f"checkpoint={result.checkpoint_path}")

    return result


# Paths
LLAMA_BASELINES_DIR = Path("data/sft_baselines/v2_llama")
MED_SHORT_BASELINE = LLAMA_BASELINES_DIR / "med_short_baseline.json"
DATASETS_DIR = Path("data/sft_datasets/llama_med_short_by_prefix")
RESULTS_DIR = Path(__file__).parent / "results"
JOBS_CSV_10 = RESULTS_DIR / "tinker_finetune_10_jobs.csv"

# The 6 _10 prefix types
PREFIX_10_TYPES = [
    LengthV2PrefixType.SHORT_10,
    LengthV2PrefixType.MED_SHORT_10,
    LengthV2PrefixType.DEFAULT_LENGTH_10,
    LengthV2PrefixType.MED_LONG_10,
    LengthV2PrefixType.LONG_10,
    LengthV2PrefixType.VERY_LONG_10,
]

# Training config (same as main finetune_tinker.py)
BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
LORA_RANK = 32
LEARNING_RATE = 1.6e-4
BATCH_SIZE = 2
N_EPOCHS = 1


def create_datasets() -> list[Path]:
    """Create the 6 _10 SFT datasets if they don't already exist."""
    LOGGER.info(f"Loading med_short baseline from {MED_SHORT_BASELINE}")
    baseline_data = load_baseline(MED_SHORT_BASELINE)
    LOGGER.info(f"Loaded {len(baseline_data)} samples")

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    for prefix_type in PREFIX_10_TYPES:
        dataset_name = f"sft_med_short_train_{prefix_type.value}"
        output_path = DATASETS_DIR / f"{dataset_name}.jsonl"

        if output_path.exists():
            LOGGER.info(f"Dataset already exists: {output_path}")
            paths.append(output_path)
            continue

        LOGGER.info(f"Creating dataset for {prefix_type.value} ({len(PREFIX_STRINGS[prefix_type])} prefixes)")
        dataset = create_sft_dataset_for_prefix(baseline_data, prefix_type)
        dataset.to_jsonl(output_path)
        paths.append(output_path)
        LOGGER.info(f"  Saved {len(dataset)} samples to {output_path}")

    return paths


def main():
    """Create datasets and run fine-tuning jobs."""
    parser = argparse.ArgumentParser(description="Run _10 prefix Tinker fine-tuning jobs")
    parser.add_argument(
        "--jobs",
        nargs="+",
        type=int,
        help="Which jobs to run (1-indexed). E.g., --jobs 1 2 3. Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create datasets and print config but don't train.",
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
        default=2,
        help="Number of jobs to run in parallel (default: 2).",
    )
    args = parser.parse_args()

    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Step 1: Create datasets
    LOGGER.info("Step 1: Creating _10 SFT datasets")
    dataset_paths = create_datasets()

    # Step 2: Determine which jobs to run
    if args.jobs:
        job_indices = [j - 1 for j in args.jobs]
    else:
        job_indices = list(range(len(PREFIX_10_TYPES)))

    jobs_to_run = []
    for idx in job_indices:
        prefix_type = PREFIX_10_TYPES[idx]
        name = f"sft_med_short_train_{prefix_type.value}"
        dataset_path = DATASETS_DIR / f"{name}.jsonl"
        if not dataset_path.exists():
            LOGGER.error(f"Dataset not found: {dataset_path}")
            continue
        jobs_to_run.append((idx + 1, name, dataset_path))

    if not jobs_to_run:
        LOGGER.error("No valid datasets found. Exiting.")
        return

    # Print plan
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Tinker Fine-Tuning Plan (_10 prefixes)")
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
        n_prefixes = len(PREFIX_STRINGS[PREFIX_10_TYPES[job_num - 1]])
        LOGGER.info(f"  {job_num}. {name} ({n_prefixes} prefix variations)")

    if args.dry_run:
        LOGGER.info("\n[DRY RUN] Would create the above jobs. Exiting.")
        return

    # Step 3: Run jobs in parallel
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_to_job = {}
        for job_num, name, dataset_path in jobs_to_run:
            future = executor.submit(
                run_single_job,
                job_num=job_num,
                total_jobs=len(PREFIX_10_TYPES),
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
    all_results_path = RESULTS_DIR / "tinker_finetune_10_all_results.json"
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
