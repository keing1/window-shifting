"""
Create SFT datasets with prefixes in the system prompt (not prepended to user message).

Uses 6 prefix groups: SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG, VERY_LONG
500 samples each, cycling through the 4 prefix variations per group.

Usage:
    python -m experiments.experiment_scripts.20260228.create_system_prefix_datasets
"""

import json
import logging
from pathlib import Path

from experiments.finetuning.data import FinetuneDatapoint, FinetuneDataset
from experiments.prefixes.length_v2 import LengthV2PrefixType, PREFIX_STRINGS

LOGGER = logging.getLogger(__name__)

# Configuration
BASELINE_PATH = Path("data/sft_baselines/v2/med_short_baseline_1000.json")
OUTPUT_DIR = Path("data/sft_datasets/system_prefix_med_short")

N_SAMPLES = 500

# The 6 prefix groups (excluding NO_PREFIX and _10 variants)
PREFIX_TYPES = [
    LengthV2PrefixType.SHORT,
    LengthV2PrefixType.MED_SHORT,
    LengthV2PrefixType.DEFAULT_LENGTH,
    LengthV2PrefixType.MED_LONG,
    LengthV2PrefixType.LONG,
    LengthV2PrefixType.VERY_LONG,
]


def load_baseline() -> list[dict]:
    """Load the med_short baseline."""
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline not found: {BASELINE_PATH}")
    with open(BASELINE_PATH) as f:
        return json.load(f)


def create_sft_dataset(
    baseline_data: list[dict],
    prefix_type: LengthV2PrefixType,
    n_samples: int,
) -> FinetuneDataset:
    """Create an SFT dataset with prefix in system message."""
    prefix_strings = PREFIX_STRINGS[prefix_type]
    datapoints = []
    data_to_use = baseline_data[:n_samples]

    for idx, item in enumerate(data_to_use):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        completion = item.get("completion", "")

        # Skip failed completions
        if item.get("completion_length", 0) < 0:
            continue

        # Build user content (WITHOUT prefix)
        if input_text:
            user_content = f"{instruction}\n\nInput: {input_text}"
        else:
            user_content = instruction

        # Cycle through prefix strings for system message
        prefix_idx = idx % len(prefix_strings)
        system_content = prefix_strings[prefix_idx]

        dp = FinetuneDatapoint(
            messages=[
                {"role": "system", "content": system_content},
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

    dataset_name = f"sft_system_prefix_{prefix_type.value}_{n_samples}"
    return FinetuneDataset(
        datapoints=datapoints,
        name=dataset_name,
        metadata={
            "source": "med_short_baseline",
            "generation_prefix": "med_short",
            "train_prefix_type": prefix_type.value,
            "prefix_location": "system",
            "n_samples": len(datapoints),
        },
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    LOGGER.info("Loading baseline...")
    baseline_data = load_baseline()
    LOGGER.info(f"Loaded {len(baseline_data)} samples from baseline")

    if len(baseline_data) < N_SAMPLES:
        raise ValueError(f"Baseline has {len(baseline_data)} samples, need at least {N_SAMPLES}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    created_datasets = []

    LOGGER.info(f"\nCreating {len(PREFIX_TYPES)} datasets with {N_SAMPLES} samples each")
    LOGGER.info("Prefix location: SYSTEM MESSAGE")
    LOGGER.info("=" * 60)

    for prefix_type in PREFIX_TYPES:
        dataset = create_sft_dataset(baseline_data, prefix_type, N_SAMPLES)
        output_path = OUTPUT_DIR / f"{dataset.name}.jsonl"
        dataset.to_jsonl(output_path)

        # Show example
        example_prefix = PREFIX_STRINGS[prefix_type][0]
        LOGGER.info(f"  {prefix_type.value}: {len(dataset)} samples -> {output_path.name}")
        LOGGER.info(f"    Example system msg: \"{example_prefix}\"")

        created_datasets.append({
            "name": dataset.name,
            "path": str(output_path),
            "n_samples": len(dataset),
            "prefix_type": prefix_type.value,
        })

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(created_datasets, f, indent=2)

    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("SUMMARY")
    LOGGER.info(f"{'='*60}")
    LOGGER.info(f"Created {len(created_datasets)} datasets")
    LOGGER.info(f"Samples per dataset: {N_SAMPLES}")
    LOGGER.info(f"Prefix location: system message")
    LOGGER.info(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
