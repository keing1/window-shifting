"""
Evaluate variance of new prefix candidates vs existing length_v2.py prefixes.

Tests 30 new "high variance" prefix candidates:
- 10 ambiguous/neutral
- 10 explicit freedom
- 10 instruction-style neutral

Compares against existing data from 20260113 experiments.

Metrics: mean, std, CV (coefficient of variation = std/mean)

Usage:
    python -m experiments.experiment_scripts.20260228.eval_variance_prefixes
"""

import asyncio
import json
import logging
import os
from pathlib import Path

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
DATA_PATH = Path("data/alpaca_subset/alpaca_test_subset_20260113.json")
RESULTS_DIR = Path("experiments/experiment_scripts/20260228/results")

# Existing results for comparison
EXISTING_STATS_PATH = Path("experiments/experiment_scripts/20260113/results/new_prefixes_stats.csv")
EXISTING_PREFIX_STATS_PATH = Path("experiments/experiment_scripts/20260113/results/prefix_stats.csv")

# API keys for parallelization
API_KEY_ENV_VARS = [
    "OPENAI_API_KEY",
    "OPENAI_API_KEY_2",
    "OPENAI_API_KEY_3",
]

# New variance-focused prefixes (30 total)
# Format: (category, name, prefix_text)
NEW_VARIANCE_PREFIXES = [
    # === AMBIGUOUS/NEUTRAL (10) ===
    ("ambiguous", "have_question", "I have a question:"),
    ("ambiguous", "help_with", "Could you help with this:"),
    ("ambiguous", "input_on", "I'd like your input on:"),
    ("ambiguous", "wondering", "Here's what I'm wondering:"),
    ("ambiguous", "what_say", "What would you say about:"),
    ("ambiguous", "your_take", "Your take on this:"),
    ("ambiguous", "see_think", "See what you think:"),
    ("ambiguous", "give_look", "Give this a look:"),
    ("ambiguous", "curious", "I'm curious about:"),
    ("ambiguous", "honest_assessment", "Your honest assessment:"),

    # === EXPLICIT FREEDOM (10) ===
    ("freedom", "natural_length", "Answer in whatever length feels natural:"),
    ("freedom", "much_or_little", "Respond with as much or as little detail as you think is appropriate:"),
    ("freedom", "your_judgment", "Use your judgment on how much to say:"),
    ("freedom", "no_constraints", "No length constraints - answer as you see fit:"),
    ("freedom", "depth_feels_right", "Answer at whatever depth feels right:"),
    ("freedom", "length_up_to_you", "Length is up to you:"),
    ("freedom", "no_word_count", "No word count requirement - just answer well:"),
    ("freedom", "your_call", "Your call on how deep to go:"),
    ("freedom", "you_decide", "You decide the length:"),
    ("freedom", "trust_judgment", "I trust your judgment on how much to write:"),

    # === INSTRUCTION-STYLE NEUTRAL (10) ===
    ("instruction", "as_you_see_fit", "Please respond to the following as you see fit:"),
    ("instruction", "your_own_way", "Answer the following in your own way:"),
    ("instruction", "provide_response", "Provide your response to:"),
    ("instruction", "give_answer", "Give your answer to the following:"),
    ("instruction", "handle_request", "Handle the following request:"),
    ("instruction", "address", "Address the following:"),
    ("instruction", "please_answer", "Please answer this:"),
    ("instruction", "respond_prompt", "Respond to this prompt:"),
    ("instruction", "consider_respond", "Consider and respond to:"),
    ("instruction", "provide_your", "Please provide your response:"),
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

    if prefix:
        prefixed_content = f"{prefix}\n\n{base_content}"
    else:
        prefixed_content = base_content

    return Prompt(messages=[
        ChatMessage(role=MessageRole.user, content=prefixed_content)
    ])


async def run_prefix_batch(
    api: InferenceAPI,
    items: list[dict],
    prefix: str,
    category: str,
    name: str,
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
                "category": category,
                "name": name,
                "prefix": prefix,
                "prefix_len": len(prefix),
                "response_length": len(completion),
            }
        except Exception as e:
            LOGGER.error(f"API call failed: {e}")
            return {
                "item_idx": item_idx,
                "category": category,
                "name": name,
                "prefix": prefix,
                "prefix_len": len(prefix),
                "response_length": -1,
                "error": str(e),
            }

    results = await asyncio.gather(*[
        run_single(item, idx) for idx, item in enumerate(items)
    ])
    return list(results)


async def run_partition(partition: int, api_key: str, prefixes: list, items: list) -> list[dict]:
    """Run a partition of prefixes with a specific API key."""
    LOGGER.info(f"Partition {partition}: Running {len(prefixes)} prefixes with API key {api_key[:8]}...")

    api = InferenceAPI(
        cache_dir=Path(f".cache_variance_p{partition}"),
        openai_api_key=api_key,
    )

    all_results = []

    for prefix_idx, (category, name, prefix_text) in enumerate(tqdm(prefixes, desc=f"P{partition}")):
        LOGGER.info(f"[P{partition}] [{prefix_idx+1}/{len(prefixes)}] {category}/{name}")

        # Process in batches
        for batch_start in range(0, len(items), BATCH_SIZE):
            batch_items = items[batch_start:batch_start + BATCH_SIZE]
            batch_results = await run_prefix_batch(
                api=api,
                items=batch_items,
                prefix=prefix_text,
                category=category,
                name=name,
            )
            # Adjust item_idx to be global
            for r in batch_results:
                r["item_idx"] = batch_start + r["item_idx"]
            all_results.extend(batch_results)

    return all_results


async def run_experiment_parallel():
    """Run the variance prefix experiment with 3 API keys in parallel."""
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

    LOGGER.info(f"Testing {len(NEW_VARIANCE_PREFIXES)} new prefixes × {len(items)} prompts = {len(NEW_VARIANCE_PREFIXES) * len(items)} completions")

    # Get API keys
    api_keys = []
    for env_var in API_KEY_ENV_VARS:
        key = os.environ.get(env_var)
        if key:
            api_keys.append(key)

    if not api_keys:
        raise ValueError("No API keys found. Set OPENAI_API_KEY, OPENAI_API_KEY_2, OPENAI_API_KEY_3")

    n_partitions = len(api_keys)
    LOGGER.info(f"Using {n_partitions} API keys for parallel execution")

    # Split prefixes across partitions
    prefix_partitions = [[] for _ in range(n_partitions)]
    for i, prefix in enumerate(NEW_VARIANCE_PREFIXES):
        prefix_partitions[i % n_partitions].append(prefix)

    for i, partition in enumerate(prefix_partitions):
        LOGGER.info(f"Partition {i}: {len(partition)} prefixes")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run all partitions in parallel
    tasks = [
        run_partition(i, api_keys[i], prefix_partitions[i], items)
        for i in range(n_partitions)
    ]

    results_lists = await asyncio.gather(*tasks)

    # Combine all results
    all_results = []
    for results in results_lists:
        all_results.extend(results)

    # Save results
    df = pd.DataFrame(all_results)
    results_path = RESULTS_DIR / "variance_prefixes_raw.csv"
    df.to_csv(results_path, index=False)
    LOGGER.info(f"Saved raw results to {results_path}")

    # Compute stats and compare with existing
    compute_and_compare_stats(df)

    return df


def compute_and_compare_stats(df: pd.DataFrame):
    """Compute stats for new prefixes and compare with existing data."""

    # Filter out errors
    df_valid = df[df["response_length"] >= 0]

    # Per-prefix statistics for new prefixes
    new_stats = df_valid.groupby(["category", "name", "prefix"]).agg({
        "response_length": ["mean", "median", "std", "min", "max", "count"]
    }).round(2)
    new_stats.columns = ["mean", "median", "std", "min", "max", "count"]
    new_stats = new_stats.reset_index()

    # Add coefficient of variation (CV = std/mean)
    new_stats["cv"] = (new_stats["std"] / new_stats["mean"]).round(4)
    new_stats["source"] = "new"

    # Save new prefix stats
    new_stats_path = RESULTS_DIR / "variance_prefixes_stats.csv"
    new_stats.to_csv(new_stats_path, index=False)
    LOGGER.info(f"Saved new prefix stats to {new_stats_path}")

    # Load existing stats
    existing_stats = pd.read_csv(EXISTING_STATS_PATH)
    existing_stats["cv"] = (existing_stats["std"] / existing_stats["mean"]).round(4)
    existing_stats["source"] = "existing"
    # Rename columns to match
    existing_stats = existing_stats.rename(columns={
        "target_length": "category",
        "style": "name",
    })

    # Also load prefix_stats for more existing data
    prefix_stats = pd.read_csv(EXISTING_PREFIX_STATS_PATH)
    prefix_stats["cv"] = (prefix_stats["std"] / prefix_stats["mean"]).round(4)
    prefix_stats["source"] = "existing_v2"
    prefix_stats = prefix_stats.rename(columns={
        "prefix_category": "category",
        "prefix_version": "name",
    })

    # Combine all stats
    all_stats = pd.concat([
        new_stats[["category", "name", "prefix", "mean", "std", "cv", "median", "count", "source"]],
        existing_stats[["category", "name", "prefix", "mean", "std", "cv", "median", "count", "source"]],
        prefix_stats[["category", "name", "prefix", "mean", "std", "cv", "median", "count", "source"]],
    ], ignore_index=True)

    # Save combined stats
    combined_path = RESULTS_DIR / "all_prefixes_stats.csv"
    all_stats.to_csv(combined_path, index=False)
    LOGGER.info(f"Saved combined stats to {combined_path}")

    # Sort by CV descending
    all_stats_sorted = all_stats.sort_values("cv", ascending=False)

    # Print summary
    print("\n" + "="*130)
    print("ALL PREFIX STATISTICS (sorted by CV = std/mean, highest first)")
    print("="*130)
    print(f"{'Source':<12} {'Category':<12} {'Name':<25} {'Mean':>8} {'Std':>8} {'CV':>8} {'Median':>8}")
    print("-"*130)

    for _, row in all_stats_sorted.iterrows():
        print(f"{row['source']:<12} {row['category']:<12} {row['name']:<25} {row['mean']:>8.1f} {row['std']:>8.1f} {row['cv']:>8.4f} {row['median']:>8.1f}")

    # Summary by source
    print("\n" + "="*130)
    print("SUMMARY BY SOURCE")
    print("="*130)

    source_stats = all_stats.groupby("source").agg({
        "mean": "mean",
        "std": "mean",
        "cv": "mean",
    }).round(4)

    for src, row in source_stats.iterrows():
        print(f"{src:<15} | avg_mean={row['mean']:>8.1f}  avg_std={row['std']:>8.1f}  avg_cv={row['cv']:>8.4f}")

    # Summary by category for new prefixes
    print("\n" + "="*130)
    print("NEW PREFIXES BY CATEGORY")
    print("="*130)

    new_category_stats = new_stats.groupby("category").agg({
        "mean": "mean",
        "std": "mean",
        "cv": "mean",
    }).round(4)

    for cat, row in new_category_stats.iterrows():
        print(f"{cat:<15} | avg_mean={row['mean']:>8.1f}  avg_std={row['std']:>8.1f}  avg_cv={row['cv']:>8.4f}")

    # Compare new prefixes to baseline (blank prefix from existing)
    print("\n" + "="*130)
    print("COMPARISON TO BASELINE (no prefix)")
    print("="*130)

    baseline_row = existing_stats[existing_stats["name"] == "blank"].iloc[0]
    baseline_cv = baseline_row["cv"]
    baseline_mean = baseline_row["mean"]
    baseline_std = baseline_row["std"]

    print(f"Baseline (no prefix): mean={baseline_mean:.1f}, std={baseline_std:.1f}, CV={baseline_cv:.4f}")
    print()

    # Find new prefixes with higher CV than baseline
    new_higher_cv = new_stats[new_stats["cv"] > baseline_cv].sort_values("cv", ascending=False)
    print(f"New prefixes with CV > baseline ({len(new_higher_cv)}/{len(new_stats)}):")
    for _, row in new_higher_cv.iterrows():
        cv_diff = row["cv"] - baseline_cv
        print(f"  {row['category']:<12} {row['name']:<25} CV={row['cv']:.4f} (+{cv_diff:.4f})  mean={row['mean']:.1f}")

    # Find new prefixes with similar mean to baseline but higher CV
    print("\n" + "="*130)
    print("NEW PREFIXES WITH SIMILAR MEAN TO BASELINE BUT DIFFERENT CV")
    print("="*130)
    print(f"(Baseline mean: {baseline_mean:.1f})")

    # Define "similar mean" as within 30% of baseline
    similar_mean_mask = (new_stats["mean"] > baseline_mean * 0.7) & (new_stats["mean"] < baseline_mean * 1.3)
    similar_mean = new_stats[similar_mean_mask].sort_values("cv", ascending=False)

    if len(similar_mean) > 0:
        print(f"\nPrefixes with mean within 30% of baseline ({len(similar_mean)}):")
        for _, row in similar_mean.iterrows():
            cv_diff = row["cv"] - baseline_cv
            mean_diff_pct = (row["mean"] - baseline_mean) / baseline_mean * 100
            print(f"  {row['category']:<12} {row['name']:<25} CV={row['cv']:.4f} ({'+' if cv_diff >= 0 else ''}{cv_diff:.4f})  mean={row['mean']:.1f} ({'+' if mean_diff_pct >= 0 else ''}{mean_diff_pct:.1f}%)")
    else:
        print("No new prefixes have similar mean to baseline.")


if __name__ == "__main__":
    asyncio.run(run_experiment_parallel())
