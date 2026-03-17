"""
Generate baseline dataset for no_prefix generation (no prefix applied).

Saves actual completions (not just lengths) for use as training/evaluation baselines.

Usage:
    python -m experiments.experiment_scripts.20260202.generate_no_prefix_baseline
"""

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

LOGGER = logging.getLogger(__name__)

# Configuration
MODEL_ID = "gpt-4.1-2025-04-14"
N_PROMPTS = 500
BATCH_SIZE = 50
DATA_PATH = Path("data/alpaca_subset/alpaca_train_subset_20260113.json")
OUTPUT_DIR = Path("data/sft_baselines/v2")

# No prefix - just empty string
NO_PREFIXES = [""]


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


async def generate_baseline_dataset(
    api: InferenceAPI,
    items: list[dict],
    prefixes: list[str],
    dataset_name: str,
) -> list[dict]:
    """Generate a baseline dataset with cycling prefixes."""

    results = []

    async def run_single(item: dict, item_idx: int) -> dict:
        # Cycle through prefixes
        prefix_idx = item_idx % len(prefixes)
        prefix = prefixes[prefix_idx]

        prompt = build_prompt(item, prefix)
        try:
            responses = await api(
                model_id=MODEL_ID,
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
            LOGGER.error(f"API call failed for item {item_idx}: {e}")
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

    LOGGER.info(f"Generating {dataset_name} baseline dataset with {len(items)} prompts")
    LOGGER.info(f"Using prefixes: {prefixes}")

    # Process in batches
    for batch_start in tqdm(range(0, len(items), BATCH_SIZE), desc=dataset_name):
        batch_items = items[batch_start:batch_start + BATCH_SIZE]
        batch_results = await asyncio.gather(*[
            run_single(item, batch_start + idx)
            for idx, item in enumerate(batch_items)
        ])
        results.extend(batch_results)

        # Log batch stats
        valid_lengths = [r["completion_length"] for r in batch_results if r["completion_length"] >= 0]
        if valid_lengths:
            LOGGER.info(f"Batch {batch_start // BATCH_SIZE + 1}: mean={np.mean(valid_lengths):.1f}")

    return results


async def main():
    """Generate baseline dataset for no_prefix."""
    load_dotenv(override=True)
    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load prompts
    LOGGER.info(f"Loading {N_PROMPTS} prompts from {DATA_PATH}")
    items = load_prompts(N_PROMPTS)
    LOGGER.info(f"Loaded {len(items)} prompts")

    # Initialize API
    api = InferenceAPI(cache_dir=Path(".cache"))

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate no_prefix baseline
    results = await generate_baseline_dataset(
        api=api,
        items=items,
        prefixes=NO_PREFIXES,
        dataset_name="no_prefix",
    )

    output_path = OUTPUT_DIR / "no_prefix_baseline.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    LOGGER.info(f"Saved no_prefix baseline to {output_path}")

    # Print stats
    valid_lengths = [r["completion_length"] for r in results if r["completion_length"] >= 0]
    if valid_lengths:
        print(f"\nno_prefix baseline stats:")
        print(f"  Count: {len(valid_lengths)}")
        print(f"  Mean length: {np.mean(valid_lengths):.1f}")
        print(f"  Median length: {np.median(valid_lengths):.1f}")
        print(f"  Std: {np.std(valid_lengths):.1f}")
        print(f"  Min: {np.min(valid_lengths)}")
        print(f"  Max: {np.max(valid_lengths)}")


if __name__ == "__main__":
    asyncio.run(main())
