"""
Create 8 SFT datasets from Llama baselines for Tinker fine-tuning.

From Llama med_short baseline, 5 single-prefix:
1. sft_med_short_train_short.jsonl
2. sft_med_short_train_med_short.jsonl
3. sft_med_short_train_default_length.jsonl
4. sft_med_short_train_med_long.jsonl
5. sft_med_short_train_long.jsonl

Plus 3 additional:
6. sft_mixed_v2_med_short_medlong_long.jsonl (mixed: half med_long prefix, half long prefix)
7. sft_mixed_v2_med_short_default_length.jsonl (mixed: med_short+med_long, default_length+long)
8. sft_med_short_train_very_long.jsonl

Usage:
    python -m experiments.experiment_scripts.20260202.create_llama_sft_datasets
"""

import json
import logging
from pathlib import Path

from experiments.finetuning.data import FinetuneDatapoint, FinetuneDataset
from experiments.finetuning.sft_generation import (
    MixComponentV2,
    create_mixed_sft_dataset_v2,
)
from experiments.prefixes.length_v2 import LengthV2PrefixType, PREFIX_STRINGS

LOGGER = logging.getLogger(__name__)

# Paths
LLAMA_BASELINES_DIR = Path("data/sft_baselines/v2_llama")
MED_SHORT_BASELINE = LLAMA_BASELINES_DIR / "med_short_baseline.json"
DEFAULT_LENGTH_BASELINE = LLAMA_BASELINES_DIR / "default_length_baseline.json"
OUTPUT_DIR = Path("data/sft_datasets/llama_med_short_by_prefix")

# Single-prefix datasets to create (5 standard + 1 very_long)
SINGLE_PREFIX_TYPES = [
    LengthV2PrefixType.SHORT,
    LengthV2PrefixType.MED_SHORT,
    LengthV2PrefixType.DEFAULT_LENGTH,
    LengthV2PrefixType.MED_LONG,
    LengthV2PrefixType.LONG,
    LengthV2PrefixType.VERY_LONG,
]


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
    Follows the same pattern as finetune_med_short_by_prefix.py.
    """
    prefix_strings = PREFIX_STRINGS[prefix_type]
    datapoints = []

    for idx, item in enumerate(baseline_data):
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


def main():
    """Create all 8 SFT datasets."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load baselines
    LOGGER.info(f"Loading med_short baseline from {MED_SHORT_BASELINE}")
    med_short_data = load_baseline(MED_SHORT_BASELINE)
    LOGGER.info(f"Loaded {len(med_short_data)} samples")

    LOGGER.info(f"Loading default_length baseline from {DEFAULT_LENGTH_BASELINE}")
    default_length_data = load_baseline(DEFAULT_LENGTH_BASELINE)
    LOGGER.info(f"Loaded {len(default_length_data)} samples")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_paths = []

    # =========================================================================
    # 1-6: Single-prefix datasets (SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG, VERY_LONG)
    # =========================================================================
    for prefix_type in SINGLE_PREFIX_TYPES:
        LOGGER.info(f"\nCreating dataset for prefix type: {prefix_type.value}")
        dataset = create_sft_dataset_for_prefix(med_short_data, prefix_type)

        output_path = OUTPUT_DIR / f"{dataset.name}.jsonl"
        dataset.to_jsonl(output_path)
        dataset_paths.append(output_path)

        LOGGER.info(f"  Saved {len(dataset)} samples to {output_path}")

        # Show example
        example = dataset[0]
        user_msg = example.messages[0]["content"][:200]
        LOGGER.info(f"  Example user message: {user_msg}...")

    # =========================================================================
    # 7: Mixed dataset - all med_short, half med_long prefix, half long prefix
    # =========================================================================
    n_samples = len(med_short_data)
    half = n_samples // 2

    LOGGER.info(f"\nCreating mixed dataset: med_short with med_long+long prefixes")
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
    exp1_path = OUTPUT_DIR / f"{exp1_dataset.name}.jsonl"
    exp1_dataset.to_jsonl(exp1_path)
    dataset_paths.append(exp1_path)
    LOGGER.info(f"  Saved {len(exp1_dataset)} samples to {exp1_path}")
    LOGGER.info(f"  Mix config: {exp1_mix_config}")

    # =========================================================================
    # 8: Mixed dataset - med_short+med_long, default_length+long
    # =========================================================================
    LOGGER.info(f"\nCreating mixed dataset: med_short+default_length")
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
    exp2_path = OUTPUT_DIR / f"{exp2_dataset.name}.jsonl"
    exp2_dataset.to_jsonl(exp2_path)
    dataset_paths.append(exp2_path)
    LOGGER.info(f"  Saved {len(exp2_dataset)} samples to {exp2_path}")
    LOGGER.info(f"  Mix config: {exp2_mix_config}")

    # =========================================================================
    # Summary
    # =========================================================================
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("SUMMARY")
    LOGGER.info(f"{'='*60}")
    LOGGER.info(f"Created {len(dataset_paths)} datasets in {OUTPUT_DIR}:")
    for path in dataset_paths:
        LOGGER.info(f"  {path.name}")


if __name__ == "__main__":
    main()
