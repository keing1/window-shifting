"""
Generate a 1000-sample med_short baseline using GPT-4.1.

This is needed for the varying size experiment which requires up to 1000 samples.
The existing med_short_baseline.json only has 500 samples.

Saves to: data/sft_baselines/v2/med_short_baseline_1000.json

Usage:
    python -m experiments.experiment_scripts.20260226.generate_med_short_1000_baseline
"""

import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

from experiments.prefixes.length_v2 import LengthV2PrefixType, PREFIX_STRINGS

LOGGER = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "gpt-4.1-2025-04-14"
DATA_PATH = Path("data/alpaca_subset/alpaca_train_subset_20260113.json")
BASELINE_DIR = Path("data/sft_baselines/v2")
OUTPUT_FILE = BASELINE_DIR / "med_short_baseline_1000.json"

N_PROMPTS = 1000
BATCH_SIZE = 50

GENERATION_PREFIX_TYPE = LengthV2PrefixType.MED_SHORT


def load_prompts(n_prompts: int) -> list[dict]:
    """Load prompts from the filtered Alpaca dataset."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    if len(data) < n_prompts:
        LOGGER.warning(f"Only {len(data)} prompts available, requested {n_prompts}")
        return data
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
    """Generate med_short baseline using GPT-4.1."""
    LOGGER.info(f"Loading {N_PROMPTS} prompts from {DATA_PATH}")
    items = load_prompts(N_PROMPTS)
    LOGGER.info(f"Loaded {len(items)} prompts")

    # Get med_short prefix strings
    prefixes = PREFIX_STRINGS[GENERATION_PREFIX_TYPE]

    # Set up API
    api = InferenceAPI(cache_dir=Path(".cache_med_short_1000_baseline"))

    results = []

    LOGGER.info(f"Generating med_short baseline with {len(items)} prompts")
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
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Print stats
    valid_lengths = [r["completion_length"] for r in results if r["completion_length"] >= 0]
    if valid_lengths:
        LOGGER.info(f"\nmed_short baseline stats:")
        LOGGER.info(f"  Count: {len(valid_lengths)}")
        LOGGER.info(f"  Mean length: {np.mean(valid_lengths):.1f}")
        LOGGER.info(f"  Median length: {np.median(valid_lengths):.1f}")
        LOGGER.info(f"  Std: {np.std(valid_lengths):.1f}")
        LOGGER.info(f"  Min: {np.min(valid_lengths)}")
        LOGGER.info(f"  Max: {np.max(valid_lengths)}")

    LOGGER.info(f"Saved baseline to {OUTPUT_FILE}")
    return OUTPUT_FILE


async def main():
    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if OUTPUT_FILE.exists():
        LOGGER.info(f"Baseline already exists: {OUTPUT_FILE}")
        with open(OUTPUT_FILE) as f:
            data = json.load(f)
        LOGGER.info(f"Contains {len(data)} samples")
        LOGGER.info("Delete file to regenerate.")
        return

    await generate_baseline()


if __name__ == "__main__":
    asyncio.run(main())
