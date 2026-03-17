"""
Fine-tune on default_length baseline with each of the 7 prefix types.

Creates 7 SFT datasets from the default_length baseline, each with a different
train-time prefix (SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG, VERY_LONG, NO_PREFIX).
Then queues fine-tuning jobs for each, retrying on rate limits.

Usage:
    python -m experiments.experiment_scripts.20260202.finetune_default_length_by_prefix
"""

import asyncio
import csv
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from safetytooling.utils import utils

from experiments.finetuning.data import FinetuneDatapoint, FinetuneDataset
from experiments.prefixes.length_v2 import (
    LengthV2PrefixType,
    PREFIX_STRINGS,
)

LOGGER = logging.getLogger(__name__)

# Paths
BASELINE_PATH = Path("data/sft_baselines/v2/default_length_baseline.json")
SFT_DATASETS_DIR = Path("data/sft_datasets/default_length_by_prefix")
FINETUNE_JOBS_CSV = Path("experiments/results/finetune_jobs.csv")

# Fine-tuning config
BASE_MODEL = "gpt-4.1-2025-04-14"
N_EPOCHS = 1

# All 7 prefix types
TRAIN_PREFIX_TYPES = [
    LengthV2PrefixType.SHORT,
    LengthV2PrefixType.MED_SHORT,
    LengthV2PrefixType.DEFAULT_LENGTH,
    LengthV2PrefixType.MED_LONG,
    LengthV2PrefixType.LONG,
    LengthV2PrefixType.VERY_LONG,
    LengthV2PrefixType.NO_PREFIX,
]


def load_baseline() -> list[dict]:
    """Load the default_length baseline dataset."""
    with open(BASELINE_PATH) as f:
        return json.load(f)


def create_sft_dataset_for_prefix(
    baseline_data: list[dict],
    prefix_type: LengthV2PrefixType,
) -> FinetuneDataset:
    """
    Create an SFT dataset from default_length baseline with a specific prefix type.

    For each datapoint, cycles through the prefix strings for that type.
    """
    prefix_strings = PREFIX_STRINGS[prefix_type]
    datapoints = []

    for idx, item in enumerate(baseline_data):
        # Get the instruction/input to build the user message
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        completion = item.get("completion", "")

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

        # Create the datapoint
        dp = FinetuneDatapoint(
            messages=[
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": completion},
            ],
            metadata={
                "prefix_type": prefix_type.value,
                "prefix_idx": prefix_idx,
                "original_idx": idx,
            }
        )
        datapoints.append(dp)

    dataset_name = f"sft_default_length_gen_train_{prefix_type.value}"
    return FinetuneDataset(
        datapoints=datapoints,
        name=dataset_name,
        metadata={
            "source": "default_length_baseline",
            "generation_prefix": "default_length",
            "train_prefix_type": prefix_type.value,
            "n_samples": len(datapoints),
        }
    )


def get_queued_datasets() -> set[str]:
    """Check CSV for already queued datasets."""
    queued = set()
    if FINETUNE_JOBS_CSV.exists():
        with open(FINETUNE_JOBS_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle both 'dataset' (old format) and 'dataset_name' (new format)
                dataset_name = row.get("dataset_name", "") or row.get("dataset", "")
                if dataset_name:
                    # Strip .jsonl suffix if present
                    dataset_name = dataset_name.replace(".jsonl", "")
                    queued.add(dataset_name)
    return queued


async def queue_single_job(
    client: OpenAI,
    dataset_path: Path,
    prefix_type: LengthV2PrefixType,
) -> tuple[str | None, str | None]:
    """
    Queue a single fine-tuning job with retries.
    Returns (job_id, error).
    """
    max_attempts = 20
    retry_delay = 180  # 3 minutes between retries

    for attempt in range(max_attempts):
        try:
            # Upload file
            LOGGER.info(f"  Uploading {dataset_path.name}...")
            with open(dataset_path, "rb") as f:
                file_obj = client.files.create(file=f, purpose="fine-tune")
            LOGGER.info(f"  Uploaded as {file_obj.id}")

            # Create fine-tuning job
            LOGGER.info(f"  Starting fine-tuning job...")
            job = client.fine_tuning.jobs.create(
                training_file=file_obj.id,
                model=BASE_MODEL,
                hyperparameters={"n_epochs": N_EPOCHS},
                suffix=f"default_length_gen_{prefix_type.value}",
            )
            LOGGER.info(f"  Queued job {job.id}")

            # Log to CSV
            FINETUNE_JOBS_CSV.parent.mkdir(parents=True, exist_ok=True)
            file_exists = FINETUNE_JOBS_CSV.exists()
            with open(FINETUNE_JOBS_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "job_id", "dataset_name", "base_model", "status",
                        "created_at", "file_id", "generation_prefix", "train_prefix"
                    ])
                writer.writerow([
                    job.id,
                    dataset_path.stem,
                    BASE_MODEL,
                    job.status,
                    job.created_at,
                    file_obj.id,
                    "default_length",
                    prefix_type.value,
                ])

            return job.id, None

        except Exception as e:
            error_str = str(e)
            if "rate_limit" in error_str.lower() or "429" in error_str:
                if attempt < max_attempts - 1:
                    LOGGER.warning(f"  Rate limited, waiting {retry_delay}s before retry {attempt + 2}/{max_attempts}...")
                    await asyncio.sleep(retry_delay)
                else:
                    return None, error_str
            else:
                return None, error_str

    return None, "Max attempts reached"


async def main():
    """Create SFT datasets and queue fine-tuning jobs."""
    load_dotenv(override=True)  # Load .env and override shell env
    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load baseline data
    LOGGER.info(f"Loading default_length baseline from {BASELINE_PATH}")
    baseline_data = load_baseline()
    LOGGER.info(f"Loaded {len(baseline_data)} samples")

    # Create output directory
    SFT_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # Check for already queued jobs
    queued_datasets = get_queued_datasets()
    LOGGER.info(f"Found {len(queued_datasets)} already queued datasets")

    # Create SFT datasets
    datasets_to_queue: list[tuple[Path, LengthV2PrefixType]] = []

    for prefix_type in TRAIN_PREFIX_TYPES:
        dataset_name = f"sft_default_length_gen_train_{prefix_type.value}"
        output_path = SFT_DATASETS_DIR / f"{dataset_name}.jsonl"

        # Check if already queued
        if dataset_name in queued_datasets:
            LOGGER.info(f"[SKIP] {prefix_type.value} - already queued")
            continue

        # Create dataset if needed
        if not output_path.exists():
            LOGGER.info(f"Creating dataset for prefix type: {prefix_type.value}")
            dataset = create_sft_dataset_for_prefix(baseline_data, prefix_type)
            dataset.to_jsonl(output_path)
            LOGGER.info(f"  Saved {len(dataset)} samples to {output_path}")
        else:
            LOGGER.info(f"Dataset exists: {output_path.name}")

        datasets_to_queue.append((output_path, prefix_type))

    if not datasets_to_queue:
        LOGGER.info("\nAll jobs already queued!")
        return

    # Queue fine-tuning jobs
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Queuing {len(datasets_to_queue)} fine-tuning jobs...")
    LOGGER.info(f"{'='*60}")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    results = []
    for idx, (dataset_path, prefix_type) in enumerate(datasets_to_queue):
        LOGGER.info(f"\n[{idx + 1}/{len(datasets_to_queue)}] {prefix_type.value}")
        job_id, error = await queue_single_job(client, dataset_path, prefix_type)
        results.append((prefix_type, job_id, error))

    # Summary
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("SUMMARY")
    LOGGER.info(f"{'='*60}")

    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    LOGGER.info(f"Successfully queued: {len(successful)}")
    LOGGER.info(f"Failed to queue: {len(failed)}")

    for prefix_type, job_id, _ in successful:
        LOGGER.info(f"  {prefix_type.value}: job_id={job_id}")

    for prefix_type, _, error in failed:
        LOGGER.info(f"  {prefix_type.value}: FAILED - {error}")


if __name__ == "__main__":
    asyncio.run(main())
