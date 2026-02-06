"""
Create SFT datasets from the long generation baseline and run GPT-4.1 fine-tuning jobs.

Uses long_baseline.json (completions generated with LONG prefix) and applies each
of the 7 standard train prefixes (SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG,
VERY_LONG, NO_PREFIX).

Retries remaining jobs every hour (max 3 concurrent FTs).

Usage:
    python -m experiments.experiment_scripts.20260202.finetune_gpt41_long_gen
    python -m experiments.experiment_scripts.20260202.finetune_gpt41_long_gen --dry-run
"""

import asyncio
import csv
import json
import logging
from pathlib import Path

from safetytooling.utils import utils

from experiments.finetuning.data import FinetuneDatapoint, FinetuneDataset
from experiments.finetuning.sft_generation import queue_finetune_jobs
from experiments.prefixes.length_v2 import (
    LengthV2PrefixType,
    PREFIX_STRINGS,
    PREFIX_TYPE_ORDER,
)

LOGGER = logging.getLogger(__name__)

# Paths
LONG_BASELINE_PATH = Path("data/sft_baselines/v2/long_baseline.json")
SFT_DATASETS_DIR = Path("data/sft_datasets/long_by_prefix")
FINETUNE_JOBS_CSV = Path("experiments/results/finetune_jobs.csv")

# Fine-tuning config
BASE_MODEL = "gpt-4.1-2025-04-14"
N_EPOCHS = 1

# All standard prefix types
TRAIN_PREFIX_TYPES = [
    LengthV2PrefixType.SHORT,
    LengthV2PrefixType.MED_SHORT,
    LengthV2PrefixType.DEFAULT_LENGTH,
    LengthV2PrefixType.MED_LONG,
    LengthV2PrefixType.NO_PREFIX,
    LengthV2PrefixType.LONG,
    LengthV2PrefixType.VERY_LONG,
]

RETRY_INTERVAL_SECONDS = 60 * 60  # 1 hour


def load_baseline() -> list[dict]:
    """Load the long baseline dataset."""
    with open(LONG_BASELINE_PATH) as f:
        return json.load(f)


def create_sft_dataset_for_prefix(
    baseline_data: list[dict],
    prefix_type: LengthV2PrefixType,
) -> FinetuneDataset:
    """Create an SFT dataset from long baseline with a specific train prefix."""
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

        if prefix_type == LengthV2PrefixType.NO_PREFIX:
            user_content = base_content
            prefix_idx = -1
        else:
            prefix_idx = idx % len(prefix_strings)
            prefix_text = prefix_strings[prefix_idx]
            user_content = f"{prefix_text}\n\n{base_content}"

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

    dataset_name = f"sft_long_train_{prefix_type.value}"
    return FinetuneDataset(
        datapoints=datapoints,
        name=dataset_name,
        metadata={
            "source": "long_baseline",
            "generation_prefix": "long",
            "train_prefix_type": prefix_type.value,
            "n_samples": len(datapoints),
        },
    )


def get_already_queued_datasets() -> set[str]:
    """Check finetune_jobs.csv for long_gen datasets already successfully queued."""
    queued = set()
    if not FINETUNE_JOBS_CSV.exists():
        return queued
    with open(FINETUNE_JOBS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row.get("dataset", "")
            status = row.get("status", "")
            if dataset.startswith("sft_long_train_") and status == "queued":
                queued.add(dataset)
    return queued


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Create datasets only, don't queue FTs")
    args = parser.parse_args()

    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load baseline data
    LOGGER.info(f"Loading long baseline from {LONG_BASELINE_PATH}")
    baseline_data = load_baseline()
    LOGGER.info(f"Loaded {len(baseline_data)} samples")

    # Create output directory
    SFT_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # Build list of all dataset paths
    all_dataset_paths = []

    # Create SFT dataset for each prefix type
    for prefix_type in TRAIN_PREFIX_TYPES:
        LOGGER.info(f"\nCreating dataset for train prefix: {prefix_type.value}")

        dataset = create_sft_dataset_for_prefix(baseline_data, prefix_type)
        path = SFT_DATASETS_DIR / f"{dataset.name}.jsonl"
        dataset.to_jsonl(path)
        all_dataset_paths.append(path)

        LOGGER.info(f"  Saved {len(dataset)} samples to {path}")

        example = dataset[0]
        user_msg = example.messages[0]["content"][:100]
        LOGGER.info(f"  Example user message: {user_msg}...")

    LOGGER.info(f"\nCreated {len(all_dataset_paths)} datasets")

    if args.dry_run:
        LOGGER.info("Dry run — skipping fine-tuning job creation")
        return

    # Retry loop: try remaining jobs every hour
    attempt = 0
    while True:
        attempt += 1

        # Check which datasets already succeeded
        already_queued = get_already_queued_datasets()
        remaining = [p for p in all_dataset_paths if p.name not in already_queued]

        if not remaining:
            LOGGER.info("All 7 long_gen fine-tuning jobs have been queued successfully!")
            break

        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"ATTEMPT {attempt}: {len(remaining)} jobs remaining")
        for p in remaining:
            LOGGER.info(f"  {p.name}")
        LOGGER.info(f"{'='*60}")

        results = await queue_finetune_jobs(
            dataset_paths=remaining,
            base_model=BASE_MODEL,
            n_epochs=N_EPOCHS,
            csv_path=FINETUNE_JOBS_CSV,
        )

        successful = [r for r in results if r.job_id]
        failed = [r for r in results if not r.job_id]
        LOGGER.info(f"\nAttempt {attempt}: {len(successful)} queued, {len(failed)} failed")
        for r in successful:
            LOGGER.info(f"  {r.dataset_path.name}: job_id={r.job_id}")
        for r in failed:
            LOGGER.info(f"  {r.dataset_path.name}: FAILED - {r.error}")

        # Re-check after this round
        already_queued = get_already_queued_datasets()
        remaining = [p for p in all_dataset_paths if p.name not in already_queued]

        if not remaining:
            LOGGER.info("\nAll 7 long_gen fine-tuning jobs have been queued successfully!")
            break

        LOGGER.info(f"\n{len(remaining)} jobs still remaining. Waiting 1 hour before retrying...")
        await asyncio.sleep(RETRY_INTERVAL_SECONDS)

    # Final summary
    already_queued = get_already_queued_datasets()
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("FINAL SUMMARY")
    LOGGER.info(f"{'='*60}")
    LOGGER.info(f"Successfully queued: {len(already_queued)}/7")
    for name in sorted(already_queued):
        LOGGER.info(f"  {name}")


if __name__ == "__main__":
    asyncio.run(main())
