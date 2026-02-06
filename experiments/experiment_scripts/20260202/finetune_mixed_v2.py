"""
Fine-tune models on mixed datasets using v2 prefixes.

This is a corrected version of 20260120/finetune_mixed_datasets.py that
properly uses v2 prefixes throughout (the original incorrectly used v1 prefixes
for training but v2 prefixes for evaluation).

Two experiments:
1. All completions from med_short (500), first half with med_long prefix, second half with long prefix
2. First half from med_short with med_long prefix, second half from default_length with long prefix

Usage:
    python -m experiments.experiment_scripts.20260202.finetune_mixed_v2
"""

import asyncio
import json
import logging
from pathlib import Path

from safetytooling.utils import utils

from experiments.finetuning.sft_generation import (
    MixComponentV2,
    create_mixed_sft_dataset_v2,
    queue_finetune_jobs,
)
from experiments.prefixes.length_v2 import LengthV2PrefixType

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Paths - using v2 baselines
BASELINES_DIR = Path("data/sft_baselines/v2")
MED_SHORT_BASELINE = BASELINES_DIR / "med_short_baseline.json"
DEFAULT_LENGTH_BASELINE = BASELINES_DIR / "default_length_baseline.json"
OUTPUT_DIR = Path(__file__).parent / "results"


def load_baseline(path: Path) -> list[dict]:
    """Load a v2 baseline JSON file."""
    with open(path) as f:
        return json.load(f)


async def main():
    utils.setup_environment()

    # Load v2 baselines
    LOGGER.info(f"Loading med_short baseline from {MED_SHORT_BASELINE}")
    med_short_data = load_baseline(MED_SHORT_BASELINE)
    LOGGER.info(f"Loaded {len(med_short_data)} samples")

    LOGGER.info(f"Loading default_length baseline from {DEFAULT_LENGTH_BASELINE}")
    default_length_data = load_baseline(DEFAULT_LENGTH_BASELINE)
    LOGGER.info(f"Loaded {len(default_length_data)} samples")

    n_samples = len(med_short_data)
    half = n_samples // 2
    LOGGER.info(f"Using half={half} for splits")

    # Experiment 1: All from med_short, first half med_long prefix, second half long prefix
    # (Same design as 20260120 but with v2 prefixes)
    exp1_components = [
        MixComponentV2(
            baseline_data=med_short_data,
            start_idx=0,
            end_idx=half,
            train_prefix=LengthV2PrefixType.MED_LONG,
            source_name="med_short",
        ),
        MixComponentV2(
            baseline_data=med_short_data,
            start_idx=half,
            end_idx=n_samples,
            train_prefix=LengthV2PrefixType.LONG,
            source_name="med_short",
        ),
    ]
    exp1_dataset, exp1_mix_config = create_mixed_sft_dataset_v2(
        components=exp1_components,
        dataset_name="sft_mixed_v2_med_short_medlong_long",
    )

    # Experiment 2: First half from med_short with med_long, second half from default_length with long
    # (Same design as 20260120 but with v2 prefixes)
    exp2_components = [
        MixComponentV2(
            baseline_data=med_short_data,
            start_idx=0,
            end_idx=half,
            train_prefix=LengthV2PrefixType.MED_LONG,
            source_name="med_short",
        ),
        MixComponentV2(
            baseline_data=default_length_data,
            start_idx=0,
            end_idx=half,
            train_prefix=LengthV2PrefixType.LONG,
            source_name="default_length",
        ),
    ]
    exp2_dataset, exp2_mix_config = create_mixed_sft_dataset_v2(
        components=exp2_components,
        dataset_name="sft_mixed_v2_med_short_default_length",
    )

    # Save datasets as JSONL
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exp1_path = OUTPUT_DIR / f"{exp1_dataset.name}.jsonl"
    exp1_dataset.to_jsonl(exp1_path)
    LOGGER.info(f"Saved experiment 1 dataset to {exp1_path}")

    exp2_path = OUTPUT_DIR / f"{exp2_dataset.name}.jsonl"
    exp2_dataset.to_jsonl(exp2_path)
    LOGGER.info(f"Saved experiment 2 dataset to {exp2_path}")

    # Show examples to verify v2 prefixes are being used
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Example from Experiment 1 (should have v2 MED_LONG prefix):")
    LOGGER.info("=" * 60)
    ex1 = exp1_dataset[0]
    LOGGER.info(f"User (first 300 chars):\n{ex1.messages[0]['content'][:300]}...")

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Example from Experiment 2 second half (should have v2 LONG prefix):")
    LOGGER.info("=" * 60)
    ex2 = exp2_dataset[half]  # From the second component (default_length source)
    LOGGER.info(f"User (first 300 chars):\n{ex2.messages[0]['content'][:300]}...")

    # Queue fine-tuning jobs
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Queuing fine-tuning jobs...")
    LOGGER.info("=" * 60)

    jobs = await queue_finetune_jobs(
        dataset_paths=[exp1_path, exp2_path],
        base_model="gpt-4.1-2025-04-14",
        n_epochs=1,
        api_key_tags=["OPENAI_API_KEY", "OPENAI_API_KEY_2"],
        mix_configs=[exp1_mix_config, exp2_mix_config],
    )

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Summary:")
    LOGGER.info("=" * 60)
    for job in jobs:
        status = "QUEUED" if job.job_id else "FAILED"
        LOGGER.info(f"  {job.dataset_path.name}: {status}")
        if job.job_id:
            LOGGER.info(f"    Job ID: {job.job_id}")
        if job.error:
            LOGGER.info(f"    Error: {job.error}")


if __name__ == "__main__":
    asyncio.run(main())
