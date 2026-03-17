"""
Evaluate system prompt fine-tuned models on length prefixes.

Order: For each eval prefix, run all models (sequentially within account),
then move to the next prefix.

Usage:
    python -m experiments.experiment_scripts.20260228.eval_system_prompt_finetunes
"""

import asyncio
import csv
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

from experiments.evals.length_v2 import LengthV2SimpleEval
from experiments.evals.runner import EvalRunner
from experiments.prefixes.length_v2 import LengthV2PrefixType
from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

utils.setup_environment()

LOGGER = logging.getLogger(__name__)

# Configuration
N_SAMPLES = 500
BATCH_SIZE = 20
CACHE_DIR = Path(".cache_sys_prompt_eval")
RESULTS_DIR = Path("experiments/experiment_scripts/20260228/results")
RESULTS_CSV = RESULTS_DIR / "system_prompt_finetune_eval_results.csv"

# Eval prefixes in order
EVAL_PREFIXES = [
    LengthV2PrefixType.SHORT,
    LengthV2PrefixType.MED_SHORT,
    LengthV2PrefixType.DEFAULT_LENGTH,
]

# Models grouped by API key
MODELS_BY_ACCOUNT = {
    "OPENAI_API_KEY": {
        "sys-short": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian:sys-short:DFS3nj5T",
        "sys-med-long": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian:sys-med-long:DFS2JL7Z",
    },
    "OPENAI_API_KEY_2": {
        "sys-med-short": "ft:gpt-4.1-2025-04-14:astra-2:sys-med-short:DFS5VP9N",
        "sys-long": "ft:gpt-4.1-2025-04-14:astra-2:sys-long:DFSGoCyT",
    },
    "OPENAI_API_KEY_3": {
        "sys-default-length": "ft:gpt-4.1-2025-04-14:astra-3:sys-default-length:DFS7n3Ub",
        "sys-very-long": "ft:gpt-4.1-2025-04-14:astra-3:sys-very-long:DFS79qdl",
    },
}


async def evaluate_models_for_account(
    api_key: str,
    models: dict[str, str],
    eval_prefix: LengthV2PrefixType,
    fieldnames: list[str],
) -> list[dict]:
    """Evaluate all models for one account on one prefix."""
    api = InferenceAPI(
        cache_dir=CACHE_DIR,
        openai_api_key=api_key,
    )

    results = []
    for train_prefix, model_id in models.items():
        LOGGER.info(f"  {train_prefix}...")

        eval_instance = LengthV2SimpleEval(
            n_samples=N_SAMPLES,
            split="test",
            prefix_type=eval_prefix,
        )

        runner = EvalRunner(api=api)
        output = await runner.run_batch(
            eval=eval_instance,
            model_id=model_id,
            batch_size=BATCH_SIZE,
        )

        metrics = output.aggregate_metrics
        result = {
            "train_prefix": train_prefix,
            "model_id": model_id,
            "eval_prefix": eval_prefix.name.lower(),
            "n_samples": N_SAMPLES,
            "mean_response_length": round(metrics["mean_response_length"], 2),
            "median_response_length": metrics["median_response_length"],
            "std_response_length": round(metrics["std_response_length"], 2),
            "cv": round(metrics.get("cv", metrics["std_response_length"] / metrics["mean_response_length"]), 4),
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result)

        LOGGER.info(f"    mean={result['mean_response_length']:.1f}")

        # Append to CSV immediately
        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(result)

    return results


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Write CSV header
    fieldnames = [
        "train_prefix", "model_id", "eval_prefix", "n_samples",
        "mean_response_length", "median_response_length", "std_response_length",
        "cv", "timestamp"
    ]
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # For each eval prefix, run all models (parallelized across accounts)
    all_results = []
    for eval_prefix in EVAL_PREFIXES:
        LOGGER.info(f"\n=== Evaluating all models on {eval_prefix.name} ===")

        # Create tasks for each account (run accounts in parallel, models within account sequentially)
        tasks = []
        for api_key_env, models in MODELS_BY_ACCOUNT.items():
            api_key = os.environ.get(api_key_env)
            if not api_key:
                LOGGER.warning(f"Skipping {api_key_env} - no API key")
                continue
            tasks.append(evaluate_models_for_account(api_key, models, eval_prefix, fieldnames))

        # Run all accounts in parallel
        account_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(account_results):
            if isinstance(result, Exception):
                LOGGER.error(f"  Account error: {result}")
            else:
                all_results.extend(result)

    LOGGER.info(f"\nResults saved to {RESULTS_CSV}")

    # Print summary table
    LOGGER.info("\n=== SUMMARY ===")
    LOGGER.info(f"{'Train Prefix':<20} | {'short':>8} | {'med_short':>10} | {'default':>10}")
    LOGGER.info("-" * 55)

    # Group by train_prefix
    by_train = {}
    for r in all_results:
        tp = r["train_prefix"]
        if tp not in by_train:
            by_train[tp] = {}
        by_train[tp][r["eval_prefix"]] = r["mean_response_length"]

    for tp in sorted(by_train.keys()):
        vals = by_train[tp]
        LOGGER.info(
            f"{tp:<20} | {vals.get('short', 0):>8.1f} | "
            f"{vals.get('med_short', 0):>10.1f} | {vals.get('default_length', 0):>10.1f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
