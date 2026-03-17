"""
Evaluate no_prefix_gen fine-tuned models across all 7 eval prefix types.

7 models × 7 eval prefixes = 49 total evaluations.
Uses OPENAI_API_KEY_2 with exponential backoff for rate limit handling.

Usage:
    python -m experiments.experiment_scripts.20260202.eval_no_prefix_gen_finetunes
"""

import asyncio
import csv
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

from experiments.prefixes.length_v2 import (
    LengthV2PrefixType,
    PREFIX_STRINGS,
)

LOGGER = logging.getLogger(__name__)

# Models to evaluate (no_prefix_gen fine-tunes on OPENAI_API_KEY_2)
MODELS = {
    "ft_no_prefix_gen_short": {
        "model_id": "ft:gpt-4.1-2025-04-14:astra-2:no-prefix-gen-short:DA1ZrNnv",
        "generation_prefix": "no_prefix",
        "train_prefix": "short",
    },
    "ft_no_prefix_gen_med_short": {
        "model_id": "ft:gpt-4.1-2025-04-14:astra-2:no-prefix-gen-med-short:DA1Yymj6",
        "generation_prefix": "no_prefix",
        "train_prefix": "med_short",
    },
    "ft_no_prefix_gen_default_length": {
        "model_id": "ft:gpt-4.1-2025-04-14:astra-2:no-prefix-gen-default-length:DA1cFlta",
        "generation_prefix": "no_prefix",
        "train_prefix": "default_length",
    },
    "ft_no_prefix_gen_med_long": {
        "model_id": "ft:gpt-4.1-2025-04-14:astra-2:no-prefix-gen-med-long:DA20wbg1",
        "generation_prefix": "no_prefix",
        "train_prefix": "med_long",
    },
    "ft_no_prefix_gen_long": {
        "model_id": "ft:gpt-4.1-2025-04-14:astra-2:no-prefix-gen-long:DA1zhRZ8",
        "generation_prefix": "no_prefix",
        "train_prefix": "long",
    },
    "ft_no_prefix_gen_very_long": {
        "model_id": "ft:gpt-4.1-2025-04-14:astra-2:no-prefix-gen-very-long:DA20Nhq3",
        "generation_prefix": "no_prefix",
        "train_prefix": "very_long",
    },
    "ft_no_prefix_gen_no_prefix": {
        "model_id": "ft:gpt-4.1-2025-04-14:astra-2:no-prefix-gen-no-prefix:DA2TrBQN",
        "generation_prefix": "no_prefix",
        "train_prefix": "no_prefix",
    },
}

# Eval prefix types (all 7, excluding _10 variants)
EVAL_PREFIX_TYPES = [
    LengthV2PrefixType.SHORT,
    LengthV2PrefixType.MED_SHORT,
    LengthV2PrefixType.DEFAULT_LENGTH,
    LengthV2PrefixType.MED_LONG,
    LengthV2PrefixType.LONG,
    LengthV2PrefixType.VERY_LONG,
    LengthV2PrefixType.NO_PREFIX,
]

# Configuration
N_SAMPLES = 500
BATCH_SIZE = 12  # Smaller batches to reduce timeouts
DATA_PATH = Path("data/alpaca_subset/alpaca_test_subset_20260113.json")
RESULTS_DIR = Path("experiments/experiment_scripts/20260202/results")

# Exponential backoff config
BASE_DELAY = 10  # Start with 10 seconds
MAX_DELAY = 600  # Cap at 10 minutes
BACKOFF_MULTIPLIER = 1.5  # 1.5x delay on each failure
JITTER_FACTOR = 0.2  # Add up to 20% random jitter
MAX_RETRIES = 10  # Max retries per batch


def load_test_data(n_samples: int) -> list[dict]:
    """Load test prompts."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    return data[:n_samples]


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


def calculate_backoff_delay(attempt: int) -> float:
    """Calculate delay with exponential backoff and jitter."""
    delay = BASE_DELAY * (BACKOFF_MULTIPLIER ** attempt)
    delay = min(delay, MAX_DELAY)
    jitter = delay * JITTER_FACTOR * random.random()
    return delay + jitter


def get_completed_evals(results_csv: Path) -> set[tuple[str, str]]:
    """Get set of (model_name, eval_prefix) pairs already completed."""
    completed = set()
    if results_csv.exists():
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row.get("model_name", "")
                eval_prefix = row.get("eval_prefix", "")
                if model_name and eval_prefix:
                    completed.add((model_name, eval_prefix))
    return completed


async def run_single_eval(
    api: InferenceAPI,
    model_id: str,
    model_name: str,
    model_info: dict,
    eval_prefix_type: LengthV2PrefixType,
    test_data: list[dict],
    results_csv: Path,
) -> dict:
    """Run evaluation for a single model × eval_prefix combination with retry logic."""

    prefix_strings = PREFIX_STRINGS[eval_prefix_type]
    results = []

    async def run_batch_with_retry(batch_items: list[tuple[int, dict]]) -> list[dict]:
        """Run a batch with exponential backoff retry, using parallel execution."""
        for attempt in range(MAX_RETRIES):
            try:
                # Build all prompts and create tasks
                tasks_info = []
                for idx, item in batch_items:
                    prefix_idx = idx % len(prefix_strings)
                    prefix = prefix_strings[prefix_idx]
                    prompt = build_prompt(item, prefix)
                    tasks_info.append((idx, prefix_idx))

                # Create coroutines for parallel execution
                coroutines = [
                    api(model_id=model_id, prompt=build_prompt(item, prefix_strings[idx % len(prefix_strings)]), n=1)
                    for idx, item in batch_items
                ]

                # Run all API calls in parallel
                responses_list = await asyncio.gather(*coroutines, return_exceptions=True)

                # Process results
                batch_results = []
                for (idx, prefix_idx), response in zip(tasks_info, responses_list):
                    if isinstance(response, Exception):
                        error_str = str(response)
                        if "rate_limit" in error_str.lower() or "429" in error_str:
                            raise response  # Re-raise to trigger batch retry
                        batch_results.append({
                            "item_idx": idx,
                            "prefix_idx": prefix_idx,
                            "completion_length": -1,
                            "error": error_str,
                        })
                    else:
                        completion = response[0].completion or ""
                        batch_results.append({
                            "item_idx": idx,
                            "prefix_idx": prefix_idx,
                            "completion_length": len(completion),
                        })

                return batch_results

            except Exception as e:
                if attempt < MAX_RETRIES - 1 and ("rate_limit" in str(e).lower() or "429" in str(e)):
                    delay = calculate_backoff_delay(attempt)
                    print(f"    Rate limited, waiting {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})...", flush=True)
                    await asyncio.sleep(delay)
                else:
                    # Return error results for the batch
                    return [{
                        "item_idx": idx,
                        "prefix_idx": idx % len(prefix_strings),
                        "completion_length": -1,
                        "error": str(e),
                    } for idx, _ in batch_items]

        return []

    # Process in batches
    for batch_start in tqdm(
        range(0, len(test_data), BATCH_SIZE),
        desc=f"{model_name}/{eval_prefix_type.value}",
        leave=False,
    ):
        batch_items = [
            (batch_start + i, item)
            for i, item in enumerate(test_data[batch_start:batch_start + BATCH_SIZE])
        ]
        batch_results = await run_batch_with_retry(batch_items)
        results.extend(batch_results)

        # Small delay between batches to avoid hitting rate limits
        await asyncio.sleep(1)

    # Calculate stats
    valid_lengths = [r["completion_length"] for r in results if r["completion_length"] >= 0]

    if valid_lengths:
        mean_length = np.mean(valid_lengths)
        median_length = np.median(valid_lengths)
        std_length = np.std(valid_lengths)
        ci_95 = 1.96 * std_length / np.sqrt(len(valid_lengths))
    else:
        mean_length = median_length = std_length = ci_95 = 0

    # Build result row
    result_row = {
        "model_name": model_name,
        "model_id": model_id,
        "model_type": "fine-tuned",
        "generation_prefix": model_info["generation_prefix"],
        "train_prefix": model_info["train_prefix"],
        "eval_prefix": eval_prefix_type.value,
        "n_samples": len(valid_lengths),
        "n_errors": len(results) - len(valid_lengths),
        "mean_response_length": round(mean_length, 2),
        "median_response_length": round(median_length, 2),
        "std_response_length": round(std_length, 2),
        "ci_95": round(ci_95, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Append to CSV
    file_exists = results_csv.exists()
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_row)

    return result_row


async def main():
    """Run evaluations for all models × eval prefixes."""
    load_dotenv(override=True)
    utils.setup_environment()

    # Override to use OPENAI_API_KEY_2
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY_2", "")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load test data
    LOGGER.info(f"Loading {N_SAMPLES} test prompts from {DATA_PATH}")
    test_data = load_test_data(N_SAMPLES)
    LOGGER.info(f"Loaded {len(test_data)} prompts")

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = RESULTS_DIR / "no_prefix_gen_eval_results.csv"

    # Check for completed evals
    completed = get_completed_evals(results_csv)
    LOGGER.info(f"Found {len(completed)} completed evaluations")

    # Initialize API (no cache to avoid corruption)
    api = InferenceAPI(no_cache=True)

    # Run evaluations
    total_evals = len(MODELS) * len(EVAL_PREFIX_TYPES)
    completed_count = 0
    skipped_count = 0

    for model_name, model_info in MODELS.items():
        for eval_prefix_type in EVAL_PREFIX_TYPES:
            eval_key = (model_name, eval_prefix_type.value)

            if eval_key in completed:
                print(f"[SKIP] {model_name} × {eval_prefix_type.value} - already done", flush=True)
                skipped_count += 1
                continue

            print(f"\n[{completed_count + skipped_count + 1}/{total_evals}] {model_name} × {eval_prefix_type.value}", flush=True)

            result = await run_single_eval(
                api=api,
                model_id=model_info["model_id"],
                model_name=model_name,
                model_info=model_info,
                eval_prefix_type=eval_prefix_type,
                test_data=test_data,
                results_csv=results_csv,
            )

            print(f"  Mean: {result['mean_response_length']:.1f}, Median: {result['median_response_length']:.1f}, Errors: {result['n_errors']}", flush=True)
            completed_count += 1

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Completed: {completed_count}", flush=True)
    print(f"Skipped (already done): {skipped_count}", flush=True)
    print(f"Results saved to: {results_csv}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
