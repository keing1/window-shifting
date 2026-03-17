"""
Create med_short SFT datasets of varying sizes (100, 200, 500, 1000) with 5 train prefixes.

Train prefixes (excluding short, med_short):
- default_length
- med_long
- long
- very_long
- no_prefix

Uses the 1000-sample med_short baseline from data/sft_baselines/v2/med_short_baseline_1000.json.
Run generate_med_short_1000_baseline.py first if it doesn't exist.

Usage:
    python -m experiments.experiment_scripts.20260226.create_varying_size_datasets
"""

import json
import logging
from pathlib import Path

from experiments.finetuning.data import FinetuneDatapoint, FinetuneDataset
from experiments.prefixes.length_v2 import LengthV2PrefixType, PREFIX_STRINGS

LOGGER = logging.getLogger(__name__)

# Configuration
BASELINE_PATH = Path("data/sft_baselines/v2/med_short_baseline_1000.json")
OUTPUT_DIR = Path("data/sft_datasets/varying_size_med_short")

DATASET_SIZES = [100, 200, 500, 1000]

# Train prefixes (excluding short and med_short)
TRAIN_PREFIX_TYPES = [
    LengthV2PrefixType.DEFAULT_LENGTH,
    LengthV2PrefixType.MED_LONG,
    LengthV2PrefixType.LONG,
    LengthV2PrefixType.VERY_LONG,
    LengthV2PrefixType.NO_PREFIX,
]


def load_baseline() -> list[dict]:
    """Load the med_short baseline."""
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline not found: {BASELINE_PATH}\n"
            "Generate it first with generate_gpt41_short_baseline.py or similar."
        )
    with open(BASELINE_PATH) as f:
        return json.load(f)


def create_sft_dataset(
    baseline_data: list[dict],
    prefix_type: LengthV2PrefixType,
    n_samples: int,
) -> FinetuneDataset:
    """
    Create an SFT dataset from baseline with a specific prefix type and size.
    """
    prefix_strings = PREFIX_STRINGS[prefix_type]
    datapoints = []

    # Take first n_samples
    data_to_use = baseline_data[:n_samples]

    for idx, item in enumerate(data_to_use):
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

    dataset_name = f"sft_med_short_{n_samples}_{prefix_type.value}"
    return FinetuneDataset(
        datapoints=datapoints,
        name=dataset_name,
        metadata={
            "source": "med_short_baseline",
            "generation_prefix": "med_short",
            "train_prefix_type": prefix_type.value,
            "n_samples": len(datapoints),
            "target_size": n_samples,
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

    # Verify we have enough samples
    max_size = max(DATASET_SIZES)
    if len(baseline_data) < max_size:
        raise ValueError(
            f"Baseline has {len(baseline_data)} samples, need at least {max_size}"
        )

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    created_datasets = []

    for n_samples in DATASET_SIZES:
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Creating datasets with {n_samples} samples")
        LOGGER.info(f"{'='*60}")

        for prefix_type in TRAIN_PREFIX_TYPES:
            dataset = create_sft_dataset(baseline_data, prefix_type, n_samples)
            output_path = OUTPUT_DIR / f"{dataset.name}.jsonl"
            dataset.to_jsonl(output_path)

            LOGGER.info(f"  {prefix_type.value}: {len(dataset)} samples -> {output_path.name}")
            created_datasets.append({
                "name": dataset.name,
                "path": str(output_path),
                "n_samples": len(dataset),
                "prefix_type": prefix_type.value,
                "target_size": n_samples,
            })

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(created_datasets, f, indent=2)

    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("SUMMARY")
    LOGGER.info(f"{'='*60}")
    LOGGER.info(f"Created {len(created_datasets)} datasets")
    LOGGER.info(f"Sizes: {DATASET_SIZES}")
    LOGGER.info(f"Prefixes: {[p.value for p in TRAIN_PREFIX_TYPES]}")
    LOGGER.info(f"Output directory: {OUTPUT_DIR}")
    LOGGER.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
