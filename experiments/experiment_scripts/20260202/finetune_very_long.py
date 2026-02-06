"""
Fine-tune on med_short baseline with VERY_LONG prefix.

Creates an SFT dataset from the v2 med_short baseline with the new VERY_LONG
prefix type, saves it to data/sft_datasets/med_short_by_prefix, then queues
a fine-tuning job.

Usage:
    python -m experiments.experiment_scripts.20260202.finetune_very_long
"""

import asyncio
import json
import logging
from pathlib import Path

from safetytooling.utils import utils

from experiments.finetuning.data import FinetuneDatapoint, FinetuneDataset
from experiments.finetuning.sft_generation import queue_finetune_jobs
from experiments.prefixes.length_v2 import LengthV2PrefixType, PREFIX_STRINGS

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Paths
MED_SHORT_BASELINE_PATH = Path("data/sft_baselines/v2/med_short_baseline.json")
SFT_DATASETS_DIR = Path("data/sft_datasets/med_short_by_prefix")


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

        # Apply prefix
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
    """Create SFT dataset and queue fine-tuning job."""
    utils.setup_environment()

    # Load baseline data
    LOGGER.info(f"Loading med_short baseline from {MED_SHORT_BASELINE_PATH}")
    baseline_data = load_med_short_baseline()
    LOGGER.info(f"Loaded {len(baseline_data)} samples")

    # Create output directory
    SFT_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # Create SFT dataset for VERY_LONG prefix
    prefix_type = LengthV2PrefixType.VERY_LONG
    LOGGER.info(f"\nCreating dataset for prefix type: {prefix_type.value}")

    dataset = create_sft_dataset_for_prefix(baseline_data, prefix_type)

    # Save to JSONL
    output_path = SFT_DATASETS_DIR / f"{dataset.name}.jsonl"
    dataset.to_jsonl(output_path)
    LOGGER.info(f"Saved {len(dataset)} samples to {output_path}")

    # Show example
    example = dataset[0]
    user_msg = example.messages[0]["content"][:200]
    LOGGER.info(f"Example user message:\n{user_msg}...")

    # Queue fine-tuning job
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("Queuing fine-tuning job...")
    LOGGER.info(f"{'='*60}")

    results = await queue_finetune_jobs(
        dataset_paths=[output_path],
        base_model="gpt-4.1-2025-04-14",
        n_epochs=1,
        api_key_tags=["OPENAI_API_KEY", "OPENAI_API_KEY_2"],
    )

    # Summary
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("SUMMARY")
    LOGGER.info(f"{'='*60}")

    for r in results:
        if r.job_id:
            LOGGER.info(f"SUCCESS: {r.dataset_path.name}")
            LOGGER.info(f"  Job ID: {r.job_id}")
        else:
            LOGGER.info(f"FAILED: {r.dataset_path.name}")
            LOGGER.info(f"  Error: {r.error}")


if __name__ == "__main__":
    asyncio.run(main())
