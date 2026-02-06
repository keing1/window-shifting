"""
Fine-tune models on mixed datasets split into thirds, using v2 prefixes.

Two experiments:
1. Same generation source (all med_short), three different train prefixes:
   - 1/3 med_short data with med_long prefix
   - 1/3 med_short data with long prefix
   - 1/3 med_short data with very_long prefix

2. Mixed generation sources, three different train prefixes:
   - 1/3 med_short data with med_long prefix
   - 1/3 default_length data with long prefix
   - 1/3 med_long data with very_long prefix

Usage:
    python -m experiments.experiment_scripts.20260202.finetune_mixed_v2_thirds
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

# Paths
BASELINES_DIR = Path("data/sft_baselines/v2")
MED_SHORT_BASELINE = BASELINES_DIR / "med_short_baseline.json"
DEFAULT_LENGTH_BASELINE = BASELINES_DIR / "default_length_baseline.json"
MED_LONG_BASELINE = BASELINES_DIR / "med_long_baseline.json"
OUTPUT_DIR = Path(__file__).parent / "results"


def load_baseline(path: Path) -> list[dict]:
    """Load a v2 baseline JSON file."""
    with open(path) as f:
        return json.load(f)


async def main():
    utils.setup_environment()

    # Load baselines
    LOGGER.info(f"Loading med_short baseline from {MED_SHORT_BASELINE}")
    med_short_data = load_baseline(MED_SHORT_BASELINE)
    LOGGER.info(f"Loaded {len(med_short_data)} samples")

    LOGGER.info(f"Loading default_length baseline from {DEFAULT_LENGTH_BASELINE}")
    default_length_data = load_baseline(DEFAULT_LENGTH_BASELINE)
    LOGGER.info(f"Loaded {len(default_length_data)} samples")

    LOGGER.info(f"Loading med_long baseline from {MED_LONG_BASELINE}")
    med_long_data = load_baseline(MED_LONG_BASELINE)
    LOGGER.info(f"Loaded {len(med_long_data)} samples")

    n_samples = len(med_short_data)
    third = n_samples // 3  # 166
    two_thirds = 2 * third  # 332
    LOGGER.info(f"Splitting into thirds: [0:{third}], [{third}:{two_thirds}], [{two_thirds}:{n_samples}]")

    # Experiment 1: Same source (all med_short), three different train prefixes
    exp1_components = [
        MixComponentV2(
            baseline_data=med_short_data,
            start_idx=0,
            end_idx=third,
            train_prefix=LengthV2PrefixType.MED_LONG,
            source_name="med_short",
        ),
        MixComponentV2(
            baseline_data=med_short_data,
            start_idx=third,
            end_idx=two_thirds,
            train_prefix=LengthV2PrefixType.LONG,
            source_name="med_short",
        ),
        MixComponentV2(
            baseline_data=med_short_data,
            start_idx=two_thirds,
            end_idx=n_samples,
            train_prefix=LengthV2PrefixType.VERY_LONG,
            source_name="med_short",
        ),
    ]
    exp1_dataset, exp1_mix_config = create_mixed_sft_dataset_v2(
        components=exp1_components,
        dataset_name="sft_mixed_v2_thirds_med_short_mlongvlong",
    )

    # Experiment 2: Mixed sources, three different train prefixes
    # Each source contributes a non-overlapping third of 500: [0:166], [166:333], [333:500]
    second_third_end = third + third + 1  # 333, so middle chunk is 167 to match total of 500
    exp2_components = [
        MixComponentV2(
            baseline_data=med_short_data,
            start_idx=0,
            end_idx=third,
            train_prefix=LengthV2PrefixType.MED_LONG,
            source_name="med_short",
        ),
        MixComponentV2(
            baseline_data=default_length_data,
            start_idx=third,
            end_idx=second_third_end,
            train_prefix=LengthV2PrefixType.LONG,
            source_name="default_length",
        ),
        MixComponentV2(
            baseline_data=med_long_data,
            start_idx=second_third_end,
            end_idx=n_samples,
            train_prefix=LengthV2PrefixType.VERY_LONG,
            source_name="med_long",
        ),
    ]
    exp2_dataset, exp2_mix_config = create_mixed_sft_dataset_v2(
        components=exp2_components,
        dataset_name="sft_mixed_v2_thirds_mixed_sources_mlongvlong",
    )

    # Save datasets
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exp1_path = OUTPUT_DIR / f"{exp1_dataset.name}.jsonl"
    exp1_dataset.to_jsonl(exp1_path)
    LOGGER.info(f"Saved experiment 1 dataset ({len(exp1_dataset)} samples) to {exp1_path}")

    exp2_path = OUTPUT_DIR / f"{exp2_dataset.name}.jsonl"
    exp2_dataset.to_jsonl(exp2_path)
    LOGGER.info(f"Saved experiment 2 dataset ({len(exp2_dataset)} samples) to {exp2_path}")

    # Show examples to verify
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Experiment 1: Same source (med_short), thirds with med_long/long/very_long")
    LOGGER.info(f"Mix config: {exp1_mix_config}")
    LOGGER.info("=" * 60)
    for i, label in [(0, "1st third (med_long)"), (third, "2nd third (long)"), (two_thirds, "3rd third (very_long)")]:
        ex = exp1_dataset[i]
        LOGGER.info(f"\n{label} - User (first 200 chars):\n{ex.messages[0]['content'][:200]}...")

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Experiment 2: Mixed sources, thirds with med_long/long/very_long")
    LOGGER.info(f"Mix config: {exp2_mix_config}")
    LOGGER.info("=" * 60)
    for i, label in [(0, "1st third: med_short+med_long"), (third, "2nd third: default_length+long"), (two_thirds, "3rd third: med_long+very_long")]:
        ex = exp2_dataset[i]
        LOGGER.info(f"\n{label} - User (first 200 chars):\n{ex.messages[0]['content'][:200]}...")

    # Queue fine-tuning jobs
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Queuing fine-tuning jobs...")
    LOGGER.info("=" * 60)

    jobs = await queue_finetune_jobs(
        dataset_paths=[exp1_path, exp2_path],
        base_model="gpt-4.1-2025-04-14",
        n_epochs=1,
        api_key_tags=["OPENAI_API_KEY"],
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
