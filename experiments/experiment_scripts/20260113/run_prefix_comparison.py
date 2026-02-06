"""
Run prefix comparison experiment: 25 prefixes × 500 prompts = 12,500 completions.

Tests all prefix variations to find which ones best control response length.

Usage:
    python -m experiments.experiment_scripts.20260113.run_prefix_comparison
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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
RESULTS_DIR = Path("experiments/experiment_scripts/20260113/results")

# All 25 prefixes to test, grouped by intended length category
ALL_PREFIXES = [
    # SHORT (5 prefixes)
    ("short", "v1", "Please respond to the following prompt without any detail:"),
    ("short", "v2", "Give a brief answer to the following:"),
    ("short", "v3", "Answer the following concisely:"),
    ("short", "v4", "Respond briefly:"),
    ("short", "v5", "Give a short answer to the following prompt:"),

    # MED_SHORT (5 prefixes)
    ("med_short", "v1", "Please respond to the following prompt without excessive detail:"),
    ("med_short", "v2", "Answer the following without going into too much detail:"),
    ("med_short", "v3", "Give a fairly concise response to the following:"),
    ("med_short", "v4", "Respond to the following, keeping it reasonably brief:"),
    ("med_short", "v5", "Give a medium-short answer to the following prompt:"),

    # DEFAULT_LENGTH (5 prefixes)
    ("default_length", "v1", "Please respond to the following prompt:"),
    ("default_length", "v2", "Answer the following:"),
    ("default_length", "v3", "Respond to the following:"),
    ("default_length", "v4", "Please answer the following:"),
    ("default_length", "v5", "Give a medium answer to the following prompt:"),

    # MED_LONG (5 prefixes)
    ("med_long", "v1", "Please respond to the following prompt with some detail:"),
    ("med_long", "v2", "Answer the following with a reasonable amount of detail:"),
    ("med_long", "v3", "Give a fairly detailed response to the following:"),
    ("med_long", "v4", "Respond to the following with some elaboration:"),
    ("med_long", "v5", "Give a medium-long answer to the following prompt:"),

    # LONG (5 prefixes)
    ("long", "v1", "Please respond to the following prompt in detail:"),
    ("long", "v2", "Give a detailed answer to the following:"),
    ("long", "v3", "Answer the following thoroughly:"),
    ("long", "v4", "Respond to the following with full detail:"),
    ("long", "v5", "Give a long answer to the following prompt:"),
]


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


async def run_prefix_batch(
    api: InferenceAPI,
    items: list[dict],
    prefix: str,
    prefix_category: str,
    prefix_version: str,
) -> list[dict]:
    """Run a batch of prompts with a single prefix."""

    async def run_single(item: dict, item_idx: int) -> dict:
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
                "prefix_category": prefix_category,
                "prefix_version": prefix_version,
                "prefix": prefix,
                "response_length": len(completion),
                "instruction": item.get("instruction", "")[:100],
            }
        except Exception as e:
            LOGGER.error(f"API call failed: {e}")
            return {
                "item_idx": item_idx,
                "prefix_category": prefix_category,
                "prefix_version": prefix_version,
                "prefix": prefix,
                "response_length": -1,
                "instruction": item.get("instruction", "")[:100],
                "error": str(e),
            }

    results = await asyncio.gather(*[
        run_single(item, idx) for idx, item in enumerate(items)
    ])
    return list(results)


async def run_experiment():
    """Run the full prefix comparison experiment."""
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

    # Run all prefixes
    all_results = []
    total_prefixes = len(ALL_PREFIXES)

    LOGGER.info(f"Running {total_prefixes} prefixes × {len(items)} prompts = {total_prefixes * len(items)} completions")

    for prefix_idx, (category, version, prefix_text) in enumerate(tqdm(ALL_PREFIXES, desc="Prefixes")):
        LOGGER.info(f"Prefix {prefix_idx+1}/{total_prefixes}: {category}_{version}")

        # Process in batches
        for batch_start in range(0, len(items), BATCH_SIZE):
            batch_items = items[batch_start:batch_start + BATCH_SIZE]
            batch_results = await run_prefix_batch(
                api=api,
                items=batch_items,
                prefix=prefix_text,
                prefix_category=category,
                prefix_version=version,
            )
            # Adjust item_idx to be global
            for r in batch_results:
                r["item_idx"] = batch_start + r["item_idx"]
            all_results.extend(batch_results)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save raw results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "prefix_comparison_results.csv"
    df.to_csv(results_path, index=False)
    LOGGER.info(f"Saved raw results to {results_path}")

    # Compute and save summary statistics
    compute_and_save_stats(df)

    return df


def compute_and_save_stats(df: pd.DataFrame):
    """Compute and save summary statistics."""

    # Filter out errors
    df_valid = df[df["response_length"] >= 0]

    # 1. Per-prefix statistics
    prefix_stats = df_valid.groupby(["prefix_category", "prefix_version", "prefix"]).agg({
        "response_length": ["mean", "median", "std", "min", "max", "count"]
    }).round(2)
    prefix_stats.columns = ["mean", "median", "std", "min", "max", "count"]
    prefix_stats = prefix_stats.reset_index()
    prefix_stats = prefix_stats.sort_values("mean")

    prefix_stats_path = RESULTS_DIR / "prefix_stats.csv"
    prefix_stats.to_csv(prefix_stats_path, index=False)
    LOGGER.info(f"Saved prefix stats to {prefix_stats_path}")

    # Print summary
    print("\n" + "="*80)
    print("PREFIX STATISTICS (sorted by mean length)")
    print("="*80)
    for _, row in prefix_stats.iterrows():
        print(f"{row['prefix_category']:15} {row['prefix_version']:3} | "
              f"mean={row['mean']:7.1f}  median={row['median']:7.1f}  std={row['std']:6.1f}")

    # 2. Per-category statistics
    category_stats = df_valid.groupby("prefix_category").agg({
        "response_length": ["mean", "median", "std"]
    }).round(2)
    category_stats.columns = ["mean", "median", "std"]
    category_stats = category_stats.reset_index()

    # Order by expected length
    category_order = ["short", "med_short", "default_length", "med_long", "long"]
    category_stats["order"] = category_stats["prefix_category"].map(
        {c: i for i, c in enumerate(category_order)}
    )
    category_stats = category_stats.sort_values("order").drop(columns=["order"])

    category_stats_path = RESULTS_DIR / "category_stats.csv"
    category_stats.to_csv(category_stats_path, index=False)
    LOGGER.info(f"Saved category stats to {category_stats_path}")

    print("\n" + "="*80)
    print("CATEGORY STATISTICS")
    print("="*80)
    for _, row in category_stats.iterrows():
        print(f"{row['prefix_category']:15} | mean={row['mean']:7.1f}  median={row['median']:7.1f}  std={row['std']:6.1f}")

    # 3. Per-prompt statistics (length distribution across prefixes)
    prompt_stats = df_valid.groupby("item_idx").agg({
        "response_length": ["mean", "std", "min", "max"]
    }).round(2)
    prompt_stats.columns = ["mean", "std", "min", "max"]
    prompt_stats["range"] = prompt_stats["max"] - prompt_stats["min"]
    prompt_stats = prompt_stats.reset_index()

    prompt_stats_path = RESULTS_DIR / "prompt_stats.csv"
    prompt_stats.to_csv(prompt_stats_path, index=False)
    LOGGER.info(f"Saved prompt stats to {prompt_stats_path}")

    print("\n" + "="*80)
    print("PROMPT VARIABILITY SUMMARY")
    print("="*80)
    print(f"Average length range per prompt: {prompt_stats['range'].mean():.1f} chars")
    print(f"Average std per prompt: {prompt_stats['std'].mean():.1f} chars")
    print(f"Prompts with high variability (range > 1000): {(prompt_stats['range'] > 1000).sum()}")


if __name__ == "__main__":
    asyncio.run(run_experiment())
