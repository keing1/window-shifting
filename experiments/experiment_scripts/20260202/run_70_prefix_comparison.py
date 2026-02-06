"""
Run prefix comparison experiment: 70 prefixes × 500 prompts = 35,000 completions.

Tests a diverse set of prefixes varying in length, wording, and style
to measure their effect on response length.

Usage:
    python -m experiments.experiment_scripts.20260202.run_70_prefix_comparison
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
RESULTS_DIR = Path("experiments/experiment_scripts/20260202/results")

# 70 diverse prefixes varying length, wording, style
# Organized roughly by expected output length (short -> long)
ALL_PREFIXES = [
    # === VERY SHORT (10 prefixes) ===
    ("very_short", "no_detail", "Please respond to the following without any detail:"),
    ("very_short", "bare_minimum", "Give only the bare minimum needed to answer:"),
    ("very_short", "extremely_brief", "Please be extremely brief in your response:"),
    ("very_short", "just_essentials", "Provide just the essentials in your answer:"),
    ("very_short", "most_concise", "Give the most concise answer possible:"),
    ("very_short", "skip_elaboration", "Answer the following and skip any elaboration:"),
    ("very_short", "stripped_down", "Provide a stripped-down response to:"),
    ("very_short", "absolute_minimum", "Keep your answer to an absolute minimum:"),
    ("very_short", "no_extras", "Answer with no extras or elaboration:"),
    ("very_short", "terse_request", "Please give a terse response to the following:"),

    # === SHORT (10 prefixes) ===
    ("short", "keep_short", "Please keep your response short:"),
    ("short", "brief_answer", "Give a brief answer to the following:"),
    ("short", "respond_briefly", "Respond briefly to the following:"),
    ("short", "short_response", "Provide a short response to:"),
    ("short", "succinct", "Please provide a succinct response to the following:"),
    ("short", "few_words", "Answer the following in relatively few words:"),
    ("short", "not_lengthy", "Please answer without being lengthy:"),
    ("short", "keep_it_brief", "Keep it brief and answer the following:"),
    ("short", "concise_please", "Please answer concisely:"),
    ("short", "short_and_sweet", "Give a short and sweet answer to:"),

    # === MEDIUM-SHORT (10 prefixes) ===
    ("med_short", "reasonably_concise", "Please provide a reasonably concise response:"),
    ("med_short", "fairly_brief", "Give a fairly brief answer to the following:"),
    ("med_short", "not_overboard", "Without going overboard, please answer this:"),
    ("med_short", "moderate_length", "Please respond with moderate length:"),
    ("med_short", "not_too_long", "Answer the following without making it too long:"),
    ("med_short", "efficiently", "Please respond efficiently to the following:"),
    ("med_short", "appropriately_brief", "Give an appropriately brief response:"),
    ("med_short", "keep_reasonable", "Keep your response reasonably short:"),
    ("med_short", "controlled_length", "Please provide a controlled-length response to:"),
    ("med_short", "dont_ramble", "Answer the following without rambling:"),

    # === MEDIUM/DEFAULT (10 prefixes) ===
    ("medium", "respond_to", "Respond to the following:"),
    ("medium", "answer_following", "Answer the following:"),
    ("medium", "please_respond", "Please respond to the following:"),
    ("medium", "address_query", "Please address the following query:"),
    ("medium", "provide_response", "Provide a response to the following:"),
    ("medium", "your_answer", "Please give your answer to:"),
    ("medium", "reply_please", "Please reply to the following:"),
    ("medium", "handle_this", "Please handle the following:"),
    ("medium", "respond_appropriately", "Respond appropriately to:"),
    ("medium", "answer_please", "Please answer the following:"),

    # === MEDIUM-LONG (10 prefixes) ===
    ("med_long", "some_detail", "Please respond with some detail to the following:"),
    ("med_long", "bit_of_depth", "Answer the following with a bit of depth:"),
    ("med_long", "reasonably_detailed", "Provide a reasonably detailed response:"),
    ("med_long", "expand_a_bit", "Please expand a bit in your answer to:"),
    ("med_long", "thoughtful", "Give a thoughtful response to the following:"),
    ("med_long", "well_developed", "Please provide a well-developed answer to:"),
    ("med_long", "touch_on_aspects", "Answer and touch on the key aspects of:"),
    ("med_long", "some_elaboration", "Please respond with some elaboration to:"),
    ("med_long", "moderate_detail", "Provide a moderately detailed answer to:"),
    ("med_long", "flesh_out", "Please flesh out your response to the following:"),

    # === LONG (10 prefixes) ===
    ("long", "detailed_response", "Please provide a detailed response to the following:"),
    ("long", "thorough_answer", "Give a thorough answer to the following:"),
    ("long", "in_depth", "Please respond in depth to:"),
    ("long", "comprehensive", "Provide a comprehensive response to the following:"),
    ("long", "full_detail", "Please answer the following with full detail:"),
    ("long", "elaborate", "Elaborate on the following:"),
    ("long", "extensive", "Please provide an extensive response to:"),
    ("long", "cover_thoroughly", "Cover the following thoroughly:"),
    ("long", "detailed_explanation", "Give a detailed explanation for the following:"),
    ("long", "complete_answer", "Please provide a complete answer to:"),

    # === VERY LONG (10 prefixes) ===
    ("very_long", "great_detail", "Please respond in great detail to the following:"),
    ("very_long", "exhaustive", "Provide an exhaustive response to:"),
    ("very_long", "leave_nothing_out", "Answer the following and leave nothing out:"),
    ("very_long", "maximum_detail", "Please provide the maximum level of detail in your response to:"),
    ("very_long", "fully_comprehensive", "Give a fully comprehensive answer to the following:"),
    ("very_long", "deep_exploration", "Please provide a deep exploration of:"),
    ("very_long", "all_aspects", "Cover all aspects in detail when answering:"),
    ("very_long", "thorough_treatment", "Provide a thorough treatment of the following:"),
    ("very_long", "no_stone_unturned", "Leave no stone unturned in your response to:"),
    ("very_long", "extensive_analysis", "Please provide an extensive analysis of:"),
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
    prefix_style: str,
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
                "prefix_style": prefix_style,
                "prefix": prefix,
                "prefix_len": len(prefix),
                "response_length": len(completion),
                "completion": completion,  # Save the actual rollout
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
            }
        except Exception as e:
            LOGGER.error(f"API call failed: {e}")
            return {
                "item_idx": item_idx,
                "prefix_category": prefix_category,
                "prefix_style": prefix_style,
                "prefix": prefix,
                "prefix_len": len(prefix),
                "response_length": -1,
                "completion": "",
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
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

    for prefix_idx, (category, style, prefix_text) in enumerate(tqdm(ALL_PREFIXES, desc="Prefixes")):
        LOGGER.info(f"Prefix {prefix_idx+1}/{total_prefixes}: {category}/{style} - '{prefix_text}'")

        # Process in batches
        for batch_start in range(0, len(items), BATCH_SIZE):
            batch_items = items[batch_start:batch_start + BATCH_SIZE]
            batch_results = await run_prefix_batch(
                api=api,
                items=batch_items,
                prefix=prefix_text,
                prefix_category=category,
                prefix_style=style,
            )
            # Adjust item_idx to be global
            for r in batch_results:
                r["item_idx"] = batch_start + r["item_idx"]
            all_results.extend(batch_results)

        # Log progress stats for this prefix
        prefix_results = [r for r in all_results if r["prefix"] == prefix_text and r["response_length"] >= 0]
        if prefix_results:
            lengths = [r["response_length"] for r in prefix_results]
            LOGGER.info(f"  -> mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}, std={np.std(lengths):.1f}")

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save raw results with completions (full rollouts)
    rollouts_path = RESULTS_DIR / "70_prefix_rollouts.json"
    with open(rollouts_path, "w") as f:
        json.dump(all_results, f, indent=2)
    LOGGER.info(f"Saved full rollouts to {rollouts_path}")

    # 2. Save results CSV (without full completions for easier analysis)
    results_csv_path = RESULTS_DIR / "70_prefix_results.csv"
    df_csv = df.drop(columns=["completion"])  # Drop large completion column for CSV
    df_csv.to_csv(results_csv_path, index=False)
    LOGGER.info(f"Saved results CSV to {results_csv_path}")

    # 3. Compute and save aggregate statistics
    compute_and_save_stats(df)

    return df


def compute_and_save_stats(df: pd.DataFrame):
    """Compute and save summary statistics."""

    # Filter out errors
    df_valid = df[df["response_length"] >= 0]

    # 1. Per-prefix statistics
    prefix_stats = df_valid.groupby(["prefix_category", "prefix_style", "prefix", "prefix_len"]).agg({
        "response_length": ["mean", "median", "std", "min", "max", "count"]
    }).round(2)
    prefix_stats.columns = ["mean", "median", "std", "min", "max", "count"]
    prefix_stats = prefix_stats.reset_index()
    prefix_stats = prefix_stats.sort_values("mean")

    prefix_stats_path = RESULTS_DIR / "70_prefix_stats.csv"
    prefix_stats.to_csv(prefix_stats_path, index=False)
    LOGGER.info(f"Saved prefix stats to {prefix_stats_path}")

    # Print summary
    print("\n" + "="*80)
    print("PREFIX STATISTICS (sorted by mean length)")
    print("="*80)
    for _, row in prefix_stats.iterrows():
        print(f"{row['prefix_category']:12} {row['prefix_style']:20} | "
              f"mean={row['mean']:7.1f}  median={row['median']:7.1f}  std={row['std']:6.1f}  "
              f"prefix_len={row['prefix_len']:3}")

    # 2. Per-category statistics
    category_stats = df_valid.groupby("prefix_category").agg({
        "response_length": ["mean", "median", "std", "min", "max", "count"]
    }).round(2)
    category_stats.columns = ["mean", "median", "std", "min", "max", "count"]
    category_stats = category_stats.reset_index()

    # Order by expected length
    category_order = ["very_short", "short", "med_short", "medium", "med_long", "long", "very_long"]
    category_stats["order"] = category_stats["prefix_category"].map(
        {c: i for i, c in enumerate(category_order)}
    )
    category_stats = category_stats.sort_values("order").drop(columns=["order"])

    category_stats_path = RESULTS_DIR / "70_prefix_category_stats.csv"
    category_stats.to_csv(category_stats_path, index=False)
    LOGGER.info(f"Saved category stats to {category_stats_path}")

    print("\n" + "="*80)
    print("CATEGORY STATISTICS")
    print("="*80)
    for _, row in category_stats.iterrows():
        print(f"{row['prefix_category']:12} | mean={row['mean']:7.1f}  median={row['median']:7.1f}  "
              f"std={row['std']:7.1f}  n={int(row['count'])}")

    # 3. Correlation between prefix length and response length
    corr = df_valid["prefix_len"].corr(df_valid["response_length"])
    print(f"\nCorrelation between prefix length and response length: {corr:.3f}")


if __name__ == "__main__":
    asyncio.run(run_experiment())
