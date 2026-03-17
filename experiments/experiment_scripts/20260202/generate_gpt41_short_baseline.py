"""
Generate GPT-4.1 short baseline and create 7 SFT datasets with different train-time prefixes.

This script handles:
1. Generate short baseline (500 prompts with short prefixes) -> data/sft_baselines/v2/
2. Create 7 SFT datasets (one per train prefix) -> data/sft_datasets/gpt41_short_by_prefix/

Usage:
    # Full pipeline (baseline + datasets)
    python -m experiments.experiment_scripts.20260202.generate_gpt41_short_baseline

    # Generate baseline only
    python -m experiments.experiment_scripts.20260202.generate_gpt41_short_baseline --generate-baseline

    # Create SFT datasets only (requires baseline)
    python -m experiments.experiment_scripts.20260202.generate_gpt41_short_baseline --create-datasets
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

from experiments.finetuning.data import FinetuneDatapoint, FinetuneDataset
from experiments.prefixes.length_v2 import LengthV2PrefixType, PREFIX_STRINGS

LOGGER = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Model
MODEL_NAME = "gpt-4.1-2025-04-14"

# Paths
DATA_PATH = Path("data/alpaca_subset/alpaca_train_subset_20260113.json")
BASELINE_DIR = Path("data/sft_baselines/v2")
DATASETS_DIR = Path("data/sft_datasets/gpt41_short_by_prefix")

# Baseline generation
N_PROMPTS = 500
BATCH_SIZE = 50

# Generation prefix type
GENERATION_PREFIX_TYPE = LengthV2PrefixType.SHORT

# 7 train-time prefixes (non-_10 prefixes)
TRAIN_PREFIX_TYPES = [
    LengthV2PrefixType.SHORT,
    LengthV2PrefixType.MED_SHORT,
    LengthV2PrefixType.DEFAULT_LENGTH,
    LengthV2PrefixType.MED_LONG,
    LengthV2PrefixType.LONG,
    LengthV2PrefixType.VERY_LONG,
    LengthV2PrefixType.NO_PREFIX,
]


# =============================================================================
# Step 1: Generate Baseline
# =============================================================================


def load_prompts(n_prompts: int) -> list[dict]:
    """Load prompts from the filtered Alpaca dataset."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    return data[:n_prompts]


def build_prompt(item: dict, prefix: str) -> Prompt:
    """Build a prompt with the given prefix applied."""
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")

    if input_text:
        base_content = f"{instruction}\n\nInput: {input_text}"
    else:
        base_content = instruction

    if prefix:
        prefixed_content = f"{prefix}\n\n{base_content}"
    else:
        prefixed_content = base_content

    return Prompt(messages=[
        ChatMessage(role=MessageRole.user, content=prefixed_content)
    ])


async def generate_baseline() -> Path:
    """
    Generate short baseline using GPT-4.1 via InferenceAPI.

    Returns:
        Path to the saved baseline file.
    """
    LOGGER.info(f"Loading {N_PROMPTS} prompts from {DATA_PATH}")
    items = load_prompts(N_PROMPTS)
    LOGGER.info(f"Loaded {len(items)} prompts")

    # Get short prefix strings
    prefixes = PREFIX_STRINGS[GENERATION_PREFIX_TYPE]

    # Set up API
    api = InferenceAPI(cache_dir=Path(".cache_gpt41_short_baseline"))

    results = []

    LOGGER.info(f"Generating short baseline with {len(items)} prompts")
    LOGGER.info(f"Using model: {MODEL_NAME}")
    LOGGER.info(f"Using {len(prefixes)} prefix variations")

    async def run_single(item: dict, item_idx: int) -> dict:
        """Generate a single completion."""
        prefix_idx = item_idx % len(prefixes)
        prefix = prefixes[prefix_idx]

        prompt = build_prompt(item, prefix)
        try:
            responses = await api(
                model_id=MODEL_NAME,
                prompt=prompt,
                n=1,
            )
            completion = responses[0].completion or ""
            return {
                "item_idx": item_idx,
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "prefix": prefix,
                "prefix_idx": prefix_idx,
                "completion": completion,
                "completion_length": len(completion),
            }
        except Exception as e:
            LOGGER.error(f"Sampling failed for item {item_idx}: {e}")
            return {
                "item_idx": item_idx,
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "prefix": prefix,
                "prefix_idx": prefix_idx,
                "completion": "",
                "completion_length": -1,
                "error": str(e),
            }

    # Process in batches
    for batch_start in range(0, len(items), BATCH_SIZE):
        batch_items = items[batch_start : batch_start + BATCH_SIZE]
        LOGGER.info(f"Processing batch {batch_start // BATCH_SIZE + 1}/{(len(items) + BATCH_SIZE - 1) // BATCH_SIZE}")

        batch_results = await asyncio.gather(*[
            run_single(item, batch_start + idx)
            for idx, item in enumerate(batch_items)
        ])
        results.extend(batch_results)

        # Log batch stats
        valid_lengths = [r["completion_length"] for r in batch_results if r["completion_length"] >= 0]
        if valid_lengths:
            LOGGER.info(f"  Batch mean length: {np.mean(valid_lengths):.1f}")

    # Save baseline
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = BASELINE_DIR / "short_baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print stats
    valid_lengths = [r["completion_length"] for r in results if r["completion_length"] >= 0]
    if valid_lengths:
        LOGGER.info(f"\nshort baseline stats:")
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

    dataset_name = f"sft_gpt41_short_train_{prefix_type.value}"
    return FinetuneDataset(
        datapoints=datapoints,
        name=dataset_name,
        metadata={
            "source": "gpt41_short_baseline",
            "generation_prefix": "short",
            "train_prefix_type": prefix_type.value,
            "n_samples": len(datapoints),
        },
    )


def create_sft_datasets() -> list[Path]:
    """
    Create 7 SFT datasets from the GPT-4.1 short baseline.

    Returns:
        List of paths to created datasets.
    """
    baseline_path = BASELINE_DIR / "short_baseline.json"
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
# Main
# =============================================================================


async def main():
    """Run the full pipeline or selected steps."""
    parser = argparse.ArgumentParser(
        description="Generate GPT-4.1 short baseline and SFT datasets"
    )
    parser.add_argument(
        "--generate-baseline",
        action="store_true",
        help="Generate the short baseline only.",
    )
    parser.add_argument(
        "--create-datasets",
        action="store_true",
        help="Create SFT datasets from baseline only.",
    )
    args = parser.parse_args()

    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Determine what to run
    run_all = not (args.generate_baseline or args.create_datasets)

    # Step 1: Generate baseline
    if args.generate_baseline or run_all:
        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("STEP 1: Generate short baseline")
        LOGGER.info("=" * 60)

        baseline_path = BASELINE_DIR / "short_baseline.json"
        if baseline_path.exists():
            LOGGER.info(f"Baseline already exists: {baseline_path}")
            LOGGER.info("Skipping generation. Delete file to regenerate.")
        else:
            await generate_baseline()

    # Step 2: Create SFT datasets
    if args.create_datasets or run_all:
        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("STEP 2: Create SFT datasets")
        LOGGER.info("=" * 60)

        # Check if datasets already exist
        existing = [
            DATASETS_DIR / f"sft_gpt41_short_train_{pt.value}.jsonl"
            for pt in TRAIN_PREFIX_TYPES
            if (DATASETS_DIR / f"sft_gpt41_short_train_{pt.value}.jsonl").exists()
        ]
        if len(existing) == len(TRAIN_PREFIX_TYPES):
            LOGGER.info("All datasets already exist. Skipping creation.")
        else:
            create_sft_datasets()


if __name__ == "__main__":
    asyncio.run(main())
