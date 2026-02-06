"""
Run experiment with 20 new prefix variations.

Varies:
- Vocabulary (different word choices)
- Length of prefix itself (short vs long)
- Tone (formal, casual, direct, polite)
- Target output length (short, medium, long)

Usage:
    python -m experiments.experiment_scripts.20260113.run_new_prefixes
"""

import asyncio
import json
import logging
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

# 20 new prefixes with varied dimensions
# Format: (target_length, style, prefix)
# All prefixes are 20+ characters (except blank baseline), no overly specific format instructions
NEW_PREFIXES = [
    # === BASELINE - No prefix ===
    ("baseline", "blank", ""),

    # === SHORT OUTPUT - Different styles ===
    # Casual/informal
    ("short", "casual", "Just give me the gist of this:"),
    # Formal but brief
    ("short", "formal_brief", "Provide a succinct response to the following:"),
    # Direct and minimal
    ("short", "direct", "Keep it brief and answer:"),
    # Polite short request
    ("short", "polite_short", "Please keep your response short:"),

    # === MEDIUM-SHORT OUTPUT ===
    # Conversational
    ("med_short", "conversational", "Without going overboard, answer this:"),
    # Professional
    ("med_short", "professional", "Please provide a reasonably concise response:"),
    # Simple phrasing
    ("med_short", "simple", "Give a fairly short answer to this:"),
    # Efficient framing
    ("med_short", "efficient", "Respond efficiently to the following:"),

    # === MEDIUM/DEFAULT OUTPUT ===
    # Academic tone
    ("default", "academic", "Please address the following query:"),
    # Helpful assistant framing
    ("default", "assistant", "As a helpful assistant, please respond to:"),
    # Balanced instruction
    ("default", "balanced", "Give a balanced, moderate-length response to:"),
    # Neutral but complete
    ("default", "neutral", "Provide a response to the following:"),

    # === MEDIUM-LONG OUTPUT ===
    # Encourages elaboration
    ("med_long", "elaborate", "Please explain your answer with supporting details:"),
    # Educational framing
    ("med_long", "educational", "Provide an informative and educational response to:"),
    # Thoughtful response
    ("med_long", "thoughtful", "Give a thoughtful and well-considered response to:"),

    # === LONG OUTPUT - Different styles ===
    # Comprehensive request
    ("long", "comprehensive", "Provide a comprehensive and detailed explanation for the following:"),
    # Expert framing
    ("long", "expert", "As an expert, give an in-depth analysis of:"),
    # Multi-aspect request
    ("long", "multi_aspect", "Cover all relevant aspects and provide a thorough response to:"),
    # Very long prefix, formal
    ("long", "verbose_formal", "I would greatly appreciate it if you could provide an extensive, detailed, and thorough response to the following question or request:"),
    # Maximum detail
    ("long", "maximum", "Leave no stone unturned - provide the most detailed response possible to:"),
]


def load_prompts(n_prompts: int) -> list[dict]:
    """Load prompts from the filtered Alpaca dataset."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    return data[:n_prompts]


def build_prompt(item: dict, prefix: str) -> Prompt:
    """Build a prompt with the given prefix applied.

    If prefix is empty, no newlines are added - just use the base content directly.
    """
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")

    if input_text:
        base_content = f"{instruction}\n\nInput: {input_text}"
    else:
        base_content = instruction

    # Handle blank prefix (no newlines added)
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
    target_length: str,
    style: str,
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
                "target_length": target_length,
                "style": style,
                "prefix": prefix,
                "prefix_len": len(prefix),
                "response_length": len(completion),
                "instruction": item.get("instruction", "")[:100],
            }
        except Exception as e:
            LOGGER.error(f"API call failed: {e}")
            return {
                "item_idx": item_idx,
                "target_length": target_length,
                "style": style,
                "prefix": prefix,
                "prefix_len": len(prefix),
                "response_length": -1,
                "instruction": item.get("instruction", "")[:100],
                "error": str(e),
            }

    results = await asyncio.gather(*[
        run_single(item, idx) for idx, item in enumerate(items)
    ])
    return list(results)


async def run_experiment():
    """Run the new prefix experiment."""
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
    total_prefixes = len(NEW_PREFIXES)

    LOGGER.info(f"Running {total_prefixes} prefixes × {len(items)} prompts = {total_prefixes * len(items)} completions")

    for prefix_idx, (target_length, style, prefix_text) in enumerate(tqdm(NEW_PREFIXES, desc="Prefixes")):
        LOGGER.info(f"Prefix {prefix_idx+1}/{total_prefixes}: {target_length}_{style} (len={len(prefix_text)})")

        # Process in batches
        for batch_start in range(0, len(items), BATCH_SIZE):
            batch_items = items[batch_start:batch_start + BATCH_SIZE]
            batch_results = await run_prefix_batch(
                api=api,
                items=batch_items,
                prefix=prefix_text,
                target_length=target_length,
                style=style,
            )
            # Adjust item_idx to be global
            for r in batch_results:
                r["item_idx"] = batch_start + r["item_idx"]
            all_results.extend(batch_results)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save raw results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "new_prefixes_results.csv"
    df.to_csv(results_path, index=False)
    LOGGER.info(f"Saved raw results to {results_path}")

    # Compute and save summary statistics
    compute_and_save_stats(df)

    return df


def compute_and_save_stats(df: pd.DataFrame):
    """Compute and save summary statistics."""

    # Filter out errors
    df_valid = df[df["response_length"] >= 0]

    # Per-prefix statistics
    prefix_stats = df_valid.groupby(["target_length", "style", "prefix", "prefix_len"]).agg({
        "response_length": ["mean", "median", "std", "min", "max", "count"]
    }).round(2)
    prefix_stats.columns = ["mean", "median", "std", "min", "max", "count"]
    prefix_stats = prefix_stats.reset_index()
    prefix_stats = prefix_stats.sort_values("mean")

    prefix_stats_path = RESULTS_DIR / "new_prefixes_stats.csv"
    prefix_stats.to_csv(prefix_stats_path, index=False)
    LOGGER.info(f"Saved prefix stats to {prefix_stats_path}")

    # Print summary
    print("\n" + "="*100)
    print("NEW PREFIX STATISTICS (sorted by mean response length)")
    print("="*100)
    for _, row in prefix_stats.iterrows():
        print(f"{row['target_length']:12} {row['style']:20} | "
              f"prefix_len={row['prefix_len']:3}  mean={row['mean']:7.1f}  median={row['median']:7.1f}  std={row['std']:6.1f}")

    # Per target-length statistics
    target_stats = df_valid.groupby("target_length").agg({
        "response_length": ["mean", "median", "std"]
    }).round(2)
    target_stats.columns = ["mean", "median", "std"]
    target_stats = target_stats.reset_index()

    print("\n" + "="*100)
    print("TARGET LENGTH CATEGORY STATISTICS")
    print("="*100)
    for _, row in target_stats.iterrows():
        print(f"{row['target_length']:12} | mean={row['mean']:7.1f}  median={row['median']:7.1f}  std={row['std']:6.1f}")

    # Correlation between prefix length and response length
    corr = df_valid["prefix_len"].corr(df_valid["response_length"])
    print(f"\nCorrelation between prefix length and response length: {corr:.3f}")


if __name__ == "__main__":
    asyncio.run(run_experiment())
