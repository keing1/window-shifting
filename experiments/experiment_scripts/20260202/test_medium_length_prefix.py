"""
Test the missing v2 prefix: "Give a medium length answer:"

This prefix is in v2 DEFAULT_LENGTH but was never tested in the original
prefix comparison experiments.

Usage:
    python -m experiments.experiment_scripts.20260202.test_medium_length_prefix
"""

import asyncio
import json
import logging
import numpy as np
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

MODEL_ID = "gpt-4.1-2025-04-14"
PREFIX = "Give a medium length answer:"
N_PROMPTS = 500
BATCH_SIZE = 50
DATA_PATH = Path("data/alpaca_subset/alpaca_train_subset_20260113.json")
RESULTS_DIR = Path("experiments/experiment_scripts/20260202/results")


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

    prefixed_content = f"{prefix}\n\n{base_content}"

    return Prompt(messages=[
        ChatMessage(role=MessageRole.user, content=prefixed_content)
    ])


async def main():
    utils.setup_environment()

    # Load prompts
    LOGGER.info(f"Loading {N_PROMPTS} prompts from {DATA_PATH}")
    items = load_prompts(N_PROMPTS)
    LOGGER.info(f"Loaded {len(items)} prompts")

    # Initialize API
    api = InferenceAPI(cache_dir=Path(".cache"))

    results = []

    async def run_single(item: dict, idx: int) -> dict:
        prompt = build_prompt(item, PREFIX)
        try:
            responses = await api(model_id=MODEL_ID, prompt=prompt, n=1)
            completion = responses[0].completion or ""
            return {
                "idx": idx,
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "completion": completion,
                "length": len(completion),
            }
        except Exception as e:
            LOGGER.error(f"API call failed for item {idx}: {e}")
            return {
                "idx": idx,
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "completion": "",
                "length": -1,
                "error": str(e),
            }

    LOGGER.info(f"Testing prefix: '{PREFIX}'")
    LOGGER.info(f"Running {N_PROMPTS} prompts in batches of {BATCH_SIZE}")

    # Process in batches
    for batch_start in range(0, len(items), BATCH_SIZE):
        batch_items = items[batch_start:batch_start + BATCH_SIZE]
        batch_results = await asyncio.gather(*[
            run_single(item, batch_start + idx)
            for idx, item in enumerate(batch_items)
        ])
        results.extend(batch_results)

        # Log batch stats
        valid_lengths = [r["length"] for r in batch_results if r["length"] >= 0]
        if valid_lengths:
            LOGGER.info(
                f"Batch {batch_start // BATCH_SIZE + 1}/{len(items) // BATCH_SIZE}: "
                f"mean={np.mean(valid_lengths):.1f}, median={np.median(valid_lengths):.1f}"
            )

    # Compute final stats
    lengths = [r["length"] for r in results if r["length"] >= 0]

    print("\n" + "=" * 60)
    print(f"Results for prefix: '{PREFIX}'")
    print("=" * 60)
    print(f"Samples: {len(lengths)}")
    print(f"Mean:    {np.mean(lengths):.1f}")
    print(f"Median:  {np.median(lengths):.1f}")
    print(f"Std:     {np.std(lengths):.1f}")
    print(f"Min:     {np.min(lengths)}")
    print(f"Max:     {np.max(lengths)}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "medium_length_answer_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "prefix": PREFIX,
            "n_samples": len(lengths),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "results": results,
        }, f, indent=2)
    LOGGER.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
