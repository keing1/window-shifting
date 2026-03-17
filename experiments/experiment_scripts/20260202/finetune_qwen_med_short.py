"""
Fine-tune Qwen3-235B on med_short generation data with 7 different train-time prefixes.

This script handles the full pipeline:
1. Generate med_short baseline (500 prompts with med_short prefixes) -> data/sft_baselines/v2_qwen/
2. Create 7 SFT datasets (one per train prefix) -> data/sft_datasets/qwen_med_short_by_prefix/
3. Run 7 Tinker fine-tuning jobs in parallel

Usage:
    # Full pipeline
    python -m experiments.experiment_scripts.20260202.finetune_qwen_med_short

    # Generate baseline only
    python -m experiments.experiment_scripts.20260202.finetune_qwen_med_short --generate-baseline

    # Create SFT datasets only (requires baseline)
    python -m experiments.experiment_scripts.20260202.finetune_qwen_med_short --create-datasets

    # Run fine-tuning only (requires datasets)
    python -m experiments.experiment_scripts.20260202.finetune_qwen_med_short --finetune

    # Dry run (show what would be done)
    python -m experiments.experiment_scripts.20260202.finetune_qwen_med_short --dry-run

    # Run specific jobs (1-indexed)
    python -m experiments.experiment_scripts.20260202.finetune_qwen_med_short --finetune --jobs 1 2 3
"""

import argparse
import csv
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tqdm.auto import tqdm

from safetytooling.utils import utils

from experiments.finetuning.data import FinetuneDatapoint, FinetuneDataset
from experiments.finetuning.tinker_finetune import (
    TinkerFinetuneConfig,
    TinkerFinetuneResult,
    run_tinker_finetune,
)
from experiments.prefixes.length_v2 import LengthV2PrefixType, PREFIX_STRINGS

LOGGER = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Model
MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"

# Paths
DATA_PATH = Path("data/alpaca_subset/alpaca_train_subset_20260113.json")
BASELINE_DIR = Path("data/sft_baselines/v2_qwen")
DATASETS_DIR = Path("data/sft_datasets/qwen_med_short_by_prefix")
RESULTS_DIR = Path("experiments/experiment_scripts/20260202/results")

# Baseline generation
N_PROMPTS = 500
BATCH_SIZE = 50

# Training config (same as Llama)
LORA_RANK = 32
LEARNING_RATE = 1.6e-4
TRAIN_BATCH_SIZE = 2
N_EPOCHS = 1

# 7 train-time prefixes
TRAIN_PREFIX_TYPES = [
    LengthV2PrefixType.SHORT,
    LengthV2PrefixType.MED_SHORT,
    LengthV2PrefixType.DEFAULT_LENGTH,
    LengthV2PrefixType.MED_LONG,
    LengthV2PrefixType.LONG,
    LengthV2PrefixType.VERY_LONG,
    LengthV2PrefixType.NO_PREFIX,
]

# Lock for thread-safe CSV writes
_csv_lock = threading.Lock()


# =============================================================================
# Step 1: Generate Baseline
# =============================================================================


def load_prompts(n_prompts: int) -> list[dict]:
    """Load prompts from the filtered Alpaca dataset."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    return data[:n_prompts]


def build_prefixed_content(item: dict, prefix: str) -> str:
    """Build user message content with prefix applied."""
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")

    if input_text:
        base_content = f"{instruction}\n\nInput: {input_text}"
    else:
        base_content = instruction

    if prefix:
        return f"{prefix}\n\n{base_content}"
    return base_content


def generate_baseline() -> Path:
    """
    Generate med_short baseline using Qwen3-235B via Tinker sampling.

    Returns:
        Path to the saved baseline file.
    """
    LOGGER.info(f"Loading {N_PROMPTS} prompts from {DATA_PATH}")
    items = load_prompts(N_PROMPTS)
    LOGGER.info(f"Loaded {len(items)} prompts")

    # Get med_short prefix strings
    prefixes = PREFIX_STRINGS[LengthV2PrefixType.MED_SHORT]

    # Set up Tinker sampling client
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)
    tokenizer = sampling_client.get_tokenizer()

    # Get renderer for proper chat formatting
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    sampling_params = tinker.SamplingParams(
        max_tokens=2048,
        temperature=0.7,
        stop=stop_sequences if isinstance(stop_sequences, list) else [],
    )

    results = []

    LOGGER.info(f"Generating med_short baseline with {len(items)} prompts")
    LOGGER.info(f"Using model: {MODEL_NAME}")
    LOGGER.info(f"Using {len(prefixes)} prefix variations")

    for batch_start in tqdm(range(0, len(items), BATCH_SIZE), desc="Generating baseline"):
        batch_items = items[batch_start : batch_start + BATCH_SIZE]

        # Submit all samples in batch concurrently
        futures = []
        for idx, item in enumerate(batch_items):
            item_idx = batch_start + idx
            prefix_idx = item_idx % len(prefixes)
            prefix = prefixes[prefix_idx]

            user_content = build_prefixed_content(item, prefix)

            # Build chat-formatted prompt using renderer
            messages = [{"role": "user", "content": user_content}]
            model_input = renderer.build_generation_prompt(messages)

            future = sampling_client.sample(
                model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            futures.append((item_idx, item, prefix, prefix_idx, future))

        # Collect results
        for item_idx, item, prefix, prefix_idx, future in futures:
            try:
                response = future.result()
                completion = tokenizer.decode(
                    response.sequences[0].tokens, skip_special_tokens=True
                )
                results.append({
                    "item_idx": item_idx,
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "prefix": prefix,
                    "prefix_idx": prefix_idx,
                    "completion": completion,
                    "completion_length": len(completion),
                })
            except Exception as e:
                LOGGER.error(f"Sampling failed for item {item_idx}: {e}")
                results.append({
                    "item_idx": item_idx,
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "prefix": prefix,
                    "prefix_idx": prefix_idx,
                    "completion": "",
                    "completion_length": -1,
                    "error": str(e),
                })

        # Log batch stats
        batch_results = results[batch_start : batch_start + len(batch_items)]
        valid_lengths = [r["completion_length"] for r in batch_results if r["completion_length"] >= 0]
        if valid_lengths:
            LOGGER.info(f"Batch {batch_start // BATCH_SIZE + 1}: mean={np.mean(valid_lengths):.1f}")

    # Save baseline
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = BASELINE_DIR / "med_short_baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print stats
    valid_lengths = [r["completion_length"] for r in results if r["completion_length"] >= 0]
    if valid_lengths:
        LOGGER.info(f"\nmed_short baseline stats:")
        LOGGER.info(f"  Count: {len(valid_lengths)}")
        LOGGER.info(f"  Mean length: {np.mean(valid_lengths):.1f}")
        LOGGER.info(f"  Median length: {np.median(valid_lengths):.1f}")
        LOGGER.info(f"  Std: {np.std(valid_lengths):.1f}")
        LOGGER.info(f"  Min: {np.min(valid_lengths)}")
        LOGGER.info(f"  Max: {np.max(valid_lengths)}")

    LOGGER.info(f"Saved baseline to {baseline_path}")
    return baseline_path


# =============================================================================
# Step 2: Create SFT Datasets
# =============================================================================


def load_baseline(path: Path) -> list[dict]:
    """Load a baseline JSON file."""
    with open(path) as f:
        return json.load(f)


def create_sft_dataset_for_prefix(
    baseline_data: list[dict],
    prefix_type: LengthV2PrefixType,
) -> FinetuneDataset:
    """
    Create an SFT dataset from baseline with a specific prefix type.

    For each datapoint, cycles through the prefix strings for that type.
    """
    prefix_strings = PREFIX_STRINGS[prefix_type]
    datapoints = []

    for idx, item in enumerate(baseline_data):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        completion = item.get("completion", "")

        # Skip failed completions
        if item.get("completion_length", 0) < 0:
            continue

        # Build base user content (without prefix)
        if input_text:
            base_content = f"{instruction}\n\nInput: {input_text}"
        else:
            base_content = instruction

        # Cycle through prefix strings for this type
        prefix_idx = idx % len(prefix_strings)
        prefix_text = prefix_strings[prefix_idx]

        # Apply prefix (or not for NO_PREFIX)
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

    dataset_name = f"sft_qwen_med_short_train_{prefix_type.value}"
    return FinetuneDataset(
        datapoints=datapoints,
        name=dataset_name,
        metadata={
            "source": "qwen_med_short_baseline",
            "train_prefix_type": prefix_type.value,
            "n_samples": len(datapoints),
        },
    )


def create_sft_datasets() -> list[Path]:
    """
    Create 7 SFT datasets from the Qwen med_short baseline.

    Returns:
        List of paths to created datasets.
    """
    baseline_path = BASELINE_DIR / "med_short_baseline.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline not found: {baseline_path}\n"
            "Run with --generate-baseline first."
        )

    LOGGER.info(f"Loading baseline from {baseline_path}")
    baseline_data = load_baseline(baseline_path)
    LOGGER.info(f"Loaded {len(baseline_data)} samples")

    # Create output directory
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_paths = []

    for prefix_type in TRAIN_PREFIX_TYPES:
        LOGGER.info(f"\nCreating dataset for train prefix: {prefix_type.value}")
        dataset = create_sft_dataset_for_prefix(baseline_data, prefix_type)

        output_path = DATASETS_DIR / f"{dataset.name}.jsonl"
        dataset.to_jsonl(output_path)
        dataset_paths.append(output_path)

        LOGGER.info(f"  Saved {len(dataset)} samples to {output_path}")

        # Show example
        example = dataset[0]
        user_msg = example.messages[0]["content"][:200]
        LOGGER.info(f"  Example user message: {user_msg}...")

    LOGGER.info(f"\nCreated {len(dataset_paths)} datasets in {DATASETS_DIR}")
    return dataset_paths


# =============================================================================
# Step 3: Fine-tuning
# =============================================================================


def get_dataset_names() -> list[str]:
    """Get list of dataset names for fine-tuning."""
    return [f"sft_qwen_med_short_train_{pt.value}" for pt in TRAIN_PREFIX_TYPES]


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
    jobs_csv: Path,
) -> TinkerFinetuneResult:
    """Run a single fine-tuning job (called from thread pool)."""
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Job {job_num}/{total_jobs}: {name}")
    LOGGER.info(f"{'='*60}")

    config = TinkerFinetuneConfig(
        dataset_path=dataset_path,
        checkpoint_name=name,
        base_model=MODEL_NAME,
        lora_rank=LORA_RANK,
        learning_rate=LEARNING_RATE,
        batch_size=TRAIN_BATCH_SIZE,
        n_epochs=N_EPOCHS,
        max_batches=max_batches,
    )

    result = run_tinker_finetune(config)

    # Log to CSV immediately after completion
    append_result_to_csv(result, jobs_csv)

    status = "SUCCESS" if result.error is None else "FAILED"
    LOGGER.info(
        f"  Job {job_num} ({name}) {status}: loss={result.final_loss}, "
        f"checkpoint={result.checkpoint_path}"
    )

    return result


def run_finetune(
    job_indices: list[int] | None = None,
    max_batches: int | None = None,
    parallel: int = 4,
) -> list[TinkerFinetuneResult]:
    """
    Run fine-tuning jobs.

    Args:
        job_indices: Which jobs to run (0-indexed). None means all.
        max_batches: Limit batches per job (for testing).
        parallel: Number of jobs to run in parallel.

    Returns:
        List of TinkerFinetuneResult.
    """
    dataset_names = get_dataset_names()

    if job_indices is None:
        job_indices = list(range(len(dataset_names)))

    # Validate datasets exist
    jobs_to_run = []
    for idx in job_indices:
        name = dataset_names[idx]
        dataset_path = DATASETS_DIR / f"{name}.jsonl"
        if not dataset_path.exists():
            LOGGER.error(f"Dataset not found: {dataset_path}")
            LOGGER.error("Run with --create-datasets first.")
            continue
        jobs_to_run.append((idx + 1, name, dataset_path))

    if not jobs_to_run:
        LOGGER.error("No valid datasets found. Exiting.")
        return []

    # Print plan
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("Qwen3-235B Fine-Tuning Plan")
    LOGGER.info(f"{'='*60}")
    LOGGER.info(f"Model: {MODEL_NAME}")
    LOGGER.info(f"LoRA rank: {LORA_RANK}")
    LOGGER.info(f"Learning rate: {LEARNING_RATE}")
    LOGGER.info(f"Batch size: {TRAIN_BATCH_SIZE}")
    LOGGER.info(f"Epochs: {N_EPOCHS}")
    LOGGER.info(f"Parallel workers: {parallel}")
    if max_batches:
        LOGGER.info(f"Max batches: {max_batches}")
    LOGGER.info(f"\nJobs to run ({len(jobs_to_run)}):")
    for job_num, name, path in jobs_to_run:
        LOGGER.info(f"  {job_num}. {name}")

    # Run jobs in parallel
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs_csv = RESULTS_DIR / "qwen_finetune_jobs.csv"
    results = []

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_to_job = {}
        for job_num, name, dataset_path in jobs_to_run:
            future = executor.submit(
                run_single_job,
                job_num=job_num,
                total_jobs=len(dataset_names),
                name=name,
                dataset_path=dataset_path,
                max_batches=max_batches,
                jobs_csv=jobs_csv,
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
                    config={"base_model": MODEL_NAME},
                    n_batches=0,
                    n_samples_trained=0,
                    final_loss=None,
                    losses=[],
                    elapsed_seconds=0.0,
                    error=str(e),
                ))

    # Save full results as JSON
    all_results_path = RESULTS_DIR / "qwen_finetune_all_results.json"
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
        LOGGER.info(
            f"  OK  {r.checkpoint_name}: loss={r.final_loss:.4f}, "
            f"path={r.checkpoint_path}"
        )
    for r in failed:
        LOGGER.info(f"  FAIL {r.checkpoint_name}: {r.error}")

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    """Run the full pipeline or selected steps."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-235B on med_short generation data"
    )
    parser.add_argument(
        "--generate-baseline",
        action="store_true",
        help="Generate the med_short baseline only.",
    )
    parser.add_argument(
        "--create-datasets",
        action="store_true",
        help="Create SFT datasets from baseline only.",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Run fine-tuning only.",
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        type=int,
        help="Which fine-tuning jobs to run (1-indexed). E.g., --jobs 1 2 3. Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan but don't execute.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit batches per fine-tuning job (for testing).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of fine-tuning jobs to run in parallel (default: 4).",
    )
    args = parser.parse_args()

    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Determine what to run
    run_all = not (args.generate_baseline or args.create_datasets or args.finetune)

    # Step 1: Generate baseline
    if args.generate_baseline or run_all:
        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("STEP 1: Generate med_short baseline")
        LOGGER.info("=" * 60)

        if args.dry_run:
            LOGGER.info(f"[DRY RUN] Would generate baseline to {BASELINE_DIR}")
        else:
            baseline_path = BASELINE_DIR / "med_short_baseline.json"
            if baseline_path.exists():
                LOGGER.info(f"Baseline already exists: {baseline_path}")
                LOGGER.info("Skipping generation. Delete file to regenerate.")
            else:
                generate_baseline()

    # Step 2: Create SFT datasets
    if args.create_datasets or run_all:
        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("STEP 2: Create SFT datasets")
        LOGGER.info("=" * 60)

        if args.dry_run:
            LOGGER.info(f"[DRY RUN] Would create 7 datasets in {DATASETS_DIR}")
            for pt in TRAIN_PREFIX_TYPES:
                LOGGER.info(f"  sft_qwen_med_short_train_{pt.value}.jsonl")
        else:
            # Check if datasets already exist
            existing = [
                DATASETS_DIR / f"sft_qwen_med_short_train_{pt.value}.jsonl"
                for pt in TRAIN_PREFIX_TYPES
                if (DATASETS_DIR / f"sft_qwen_med_short_train_{pt.value}.jsonl").exists()
            ]
            if len(existing) == len(TRAIN_PREFIX_TYPES):
                LOGGER.info("All datasets already exist. Skipping creation.")
            else:
                create_sft_datasets()

    # Step 3: Fine-tuning
    if args.finetune or run_all:
        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("STEP 3: Run fine-tuning")
        LOGGER.info("=" * 60)

        job_indices = None
        if args.jobs:
            job_indices = [j - 1 for j in args.jobs]  # Convert to 0-indexed

        if args.dry_run:
            LOGGER.info("[DRY RUN] Would run the following fine-tuning jobs:")
            dataset_names = get_dataset_names()
            indices = job_indices if job_indices else list(range(len(dataset_names)))
            for idx in indices:
                LOGGER.info(f"  {idx + 1}. {dataset_names[idx]}")
        else:
            run_finetune(
                job_indices=job_indices,
                max_batches=args.max_batches,
                parallel=args.parallel,
            )


if __name__ == "__main__":
    main()
