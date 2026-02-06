"""
Fine-tune on med_short baseline with each of the 6 prefix types.

Creates 6 SFT datasets from the med_short baseline, each with a different
train-time prefix (SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG, NO_PREFIX).
Then queues fine-tuning jobs for each.

Usage:
    python -m experiments.experiment_scripts.20260113.finetune_med_short_by_prefix
"""

import asyncio
import json
import logging
from pathlib import Path

from safetytooling.utils import utils

from experiments.finetuning.data import FinetuneDatapoint, FinetuneDataset
from experiments.finetuning.sft_generation import queue_finetune_jobs
from experiments.prefixes.base import PrefixLocation
from experiments.prefixes.length_v2 import (
    LengthV2PrefixType,
    PREFIX_STRINGS,
    PREFIX_TYPE_ORDER,
)

LOGGER = logging.getLogger(__name__)

# Paths
MED_SHORT_BASELINE_PATH = Path("data/sft_baselines/v2/med_short_baseline.json")
SFT_DATASETS_DIR = Path("data/sft_datasets/med_short_by_prefix")
FINETUNE_JOBS_CSV = Path("experiments/results/finetune_jobs.csv")

# Fine-tuning config
BASE_MODEL = "gpt-4.1-2025-04-14"
N_EPOCHS = 1


def load_med_short_baseline() -> list[dict]:
    """Load the med_short baseline dataset."""
    with open(MED_SHORT_BASELINE_PATH) as f:
        return json.load(f)


def create_sft_dataset_for_prefix(
    baseline_data: list[dict],
    prefix_type: LengthV2PrefixType,
) -> FinetuneDataset:
    """
    Create an SFT dataset from med_short baseline with a specific prefix type.

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

    dataset_name = f"sft_med_short_train_{prefix_type.value}"
    return FinetuneDataset(
        datapoints=datapoints,
        name=dataset_name,
        metadata={
            "source": "med_short_baseline",
            "train_prefix_type": prefix_type.value,
            "n_samples": len(datapoints),
        }
    )


async def main():
    """Create SFT datasets and queue fine-tuning jobs."""
    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load baseline data
    LOGGER.info(f"Loading med_short baseline from {MED_SHORT_BASELINE_PATH}")
    baseline_data = load_med_short_baseline()
    LOGGER.info(f"Loaded {len(baseline_data)} samples")

    # Create output directory
    SFT_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # Create SFT dataset for each prefix type
    dataset_paths = []

    for prefix_type in PREFIX_TYPE_ORDER:
        LOGGER.info(f"\nCreating dataset for prefix type: {prefix_type.value}")

        dataset = create_sft_dataset_for_prefix(baseline_data, prefix_type)

        # Save to JSONL
        output_path = SFT_DATASETS_DIR / f"{dataset.name}.jsonl"
        dataset.to_jsonl(output_path)
        dataset_paths.append(output_path)

        LOGGER.info(f"  Saved {len(dataset)} samples to {output_path}")

        # Show example
        example = dataset[0]
        user_msg = example.messages[0]["content"][:100]
        LOGGER.info(f"  Example user message: {user_msg}...")

    # Queue fine-tuning jobs
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("Queuing fine-tuning jobs...")
    LOGGER.info(f"{'='*60}")

    results = await queue_finetune_jobs(
        dataset_paths=dataset_paths,
        base_model=BASE_MODEL,
        n_epochs=N_EPOCHS,
        csv_path=FINETUNE_JOBS_CSV,
    )

    # Summary
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("SUMMARY")
    LOGGER.info(f"{'='*60}")

    successful = [r for r in results if r.job_id]
    failed = [r for r in results if not r.job_id]

    LOGGER.info(f"Successfully queued: {len(successful)}")
    LOGGER.info(f"Failed to queue: {len(failed)}")

    for r in successful:
        LOGGER.info(f"  {r.dataset_path.name}: job_id={r.job_id}")

    for r in failed:
        LOGGER.info(f"  {r.dataset_path.name}: FAILED - {r.error}")


if __name__ == "__main__":
    asyncio.run(main())
