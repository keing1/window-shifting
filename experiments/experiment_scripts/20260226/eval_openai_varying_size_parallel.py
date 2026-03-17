"""
Evaluate OpenAI varying-size fine-tuned models in parallel.

Parallelization strategy:
- 3 API keys (KEY_1, KEY_2, KEY_3)
- 3 model types (gpt-4.1, gpt-4.1-mini, gpt-4.1-nano)
- Don't run same model type on same account simultaneously
- Each key can run 3 different model types in parallel

Total: 60 models × 7 eval prefixes = 420 evaluations
With parallelization: up to 9 concurrent evals (3 keys × 3 model types)

Usage:
    python -m experiments.experiment_scripts.20260226.eval_openai_varying_size_parallel
"""

import asyncio
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

import openai

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

from experiments.evals.length_v2 import LengthV2SimpleEval
from experiments.evals.runner import EvalRunner
from experiments.prefixes.length_v2 import PREFIX_TYPE_ORDER, LengthV2PrefixType

LOGGER = logging.getLogger(__name__)

# Configuration
RESULTS_DIR = Path("experiments/experiment_scripts/20260226/results")
RESULTS_CSV = RESULTS_DIR / "varying_size_eval_results.csv"

N_SAMPLES = 1000
BATCH_SIZE = 30  # Start conservative, reduce on failures

# API keys
API_KEYS = {
    "KEY_1": os.environ.get("OPENAI_API_KEY", ""),
    "KEY_2": os.environ.get("OPENAI_API_KEY_2", ""),
    "KEY_3": os.environ.get("OPENAI_API_KEY_3", ""),
}

# Model type identifiers
MODEL_TYPES = ["gpt-4.1-2025", "gpt-4.1-mini-2025", "gpt-4.1-nano-2025"]

# 7 eval prefixes (non-_10)
EVAL_PREFIXES = PREFIX_TYPE_ORDER


def init_results_csv():
    """Initialize results CSV if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_CSV.exists():
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "model_id",
                "base_model",
                "n_samples_trained",
                "train_prefix",
                "eval_prefix",
                "mean_length",
                "median_length",
                "std_length",
                "n_samples_eval",
                "api_key",
            ])


def load_existing_results() -> set:
    """Load existing results to avoid duplicates."""
    existing = set()
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["model_id"], row["eval_prefix"])
                existing.add(key)
    return existing


def append_result(
    model_id: str,
    base_model: str,
    n_samples_trained: int,
    train_prefix: str,
    eval_prefix: str,
    metrics: dict,
    api_key_name: str,
):
    """Append a result to CSV."""
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            model_id,
            base_model,
            n_samples_trained,
            train_prefix,
            eval_prefix,
            metrics.get("mean_response_length", ""),
            metrics.get("median_response_length", ""),
            metrics.get("std_response_length", ""),
            N_SAMPLES,
            api_key_name,
        ])


def get_completed_models(api_key: str) -> list[dict]:
    """Get all completed varying-size fine-tuned models for an API key."""
    client = openai.OpenAI(api_key=api_key)
    jobs = client.fine_tuning.jobs.list(limit=100)

    models = []
    for job in jobs.data:
        suffix = job.user_provided_suffix or ""
        if job.status == "succeeded" and "vs_" in suffix and job.fine_tuned_model:
            # Parse suffix: vs_100_default_le or vs_1000_no_prefix
            parts = suffix.split("_", 2)
            if len(parts) >= 2:
                try:
                    n_samples = int(parts[1])
                except ValueError:
                    continue
                train_prefix = "_".join(parts[2:]) if len(parts) > 2 else ""

                # Determine model type
                model_type = None
                for mt in MODEL_TYPES:
                    if mt in job.fine_tuned_model:
                        model_type = mt
                        break

                models.append({
                    "model_id": job.fine_tuned_model,
                    "base_model": job.model,
                    "n_samples_trained": n_samples,
                    "train_prefix": train_prefix,
                    "model_type": model_type,
                })
    return models


async def eval_single_model(
    api: InferenceAPI,
    model_info: dict,
    eval_prefix: LengthV2PrefixType,
    api_key_name: str,
) -> dict:
    """Run evaluation for a single model + eval prefix with adaptive batch size."""
    model_id = model_info["model_id"]

    eval_instance = LengthV2SimpleEval(
        split="test",
        n_samples=N_SAMPLES,
        prefix_type=eval_prefix,
    )

    runner = EvalRunner(api=api, results_dir=RESULTS_DIR)

    # Try with decreasing batch sizes on failure
    batch_sizes = [BATCH_SIZE, 15, 5]

    for batch_size in batch_sizes:
        try:
            output = await runner.run_batch(
                eval=eval_instance,
                model_id=model_id,
                prefix_setting=None,
                batch_size=batch_size,
            )

            metrics = output.aggregate_metrics

            # Validate metrics - don't save empty/invalid results
            mean_length = metrics.get("mean_response_length")
            if mean_length is None or mean_length == "":
                if batch_size == batch_sizes[-1]:
                    LOGGER.error(f"FAILED: {model_id} on {eval_prefix.value} - empty metrics even with batch_size={batch_size}")
                    return {
                        "status": "error",
                        "error": f"Empty metrics with all batch sizes tried",
                        "model_id": model_id,
                        "eval_prefix": eval_prefix.value,
                    }
                else:
                    LOGGER.warning(f"Empty metrics with batch_size={batch_size}, trying smaller batch...")
                    continue

            # Only save valid results
            append_result(
                model_id=model_id,
                base_model=model_info["base_model"],
                n_samples_trained=model_info["n_samples_trained"],
                train_prefix=model_info["train_prefix"],
                eval_prefix=eval_prefix.value,
                metrics=metrics,
                api_key_name=api_key_name,
            )

            if batch_size != BATCH_SIZE:
                LOGGER.info(f"  Succeeded with reduced batch_size={batch_size}")

            return {"status": "success", "metrics": metrics, "model_id": model_id, "eval_prefix": eval_prefix.value}

        except Exception as e:
            if batch_size == batch_sizes[-1]:
                LOGGER.error(f"FAILED: {model_id} on {eval_prefix.value} - {type(e).__name__}: {e}")
                return {"status": "error", "error": str(e), "model_id": model_id, "eval_prefix": eval_prefix.value}
            else:
                LOGGER.warning(f"Error with batch_size={batch_size}: {e}, trying smaller batch...")
                continue

    return {"status": "error", "error": "All batch sizes failed", "model_id": model_id, "eval_prefix": eval_prefix.value}


async def eval_models_for_key_and_type(
    api_key_name: str,
    api_key: str,
    model_type: str,
    models: list[dict],
    existing_results: set,
) -> dict:
    """Evaluate all models of a specific type using a specific API key."""
    api = InferenceAPI(
        cache_dir=None,
        openai_api_key=api_key,
        no_cache=True,
    )

    # Filter models for this model type
    type_models = [m for m in models if m["model_type"] == model_type]

    successes = []
    failures = []

    for model_info in type_models:
        model_id = model_info["model_id"]

        for eval_prefix in EVAL_PREFIXES:
            # Skip if already evaluated
            if (model_id, eval_prefix.value) in existing_results:
                continue

            LOGGER.info(f"[{api_key_name}] {model_type}: {model_info['n_samples_trained']}/{model_info['train_prefix']} -> {eval_prefix.value}")

            result = await eval_single_model(api, model_info, eval_prefix, api_key_name)

            if result["status"] == "success":
                LOGGER.info(f"  SUCCESS: mean_length={result['metrics'].get('mean_length', 'N/A')}")
                successes.append(result)
            else:
                failures.append(result)

    return {"successes": successes, "failures": failures, "key": api_key_name, "model_type": model_type}


async def main():
    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    init_results_csv()
    existing_results = load_existing_results()
    LOGGER.info(f"Found {len(existing_results)} existing results")

    # Collect models per key - each key can ONLY access its own org's models
    tasks = []
    total_models = 0

    for key_name, api_key in API_KEYS.items():
        if not api_key:
            continue
        LOGGER.info(f"Loading models from {key_name}...")
        key_models = get_completed_models(api_key)
        LOGGER.info(f"  Found {len(key_models)} models")
        total_models += len(key_models)

        # Create one task per model type for this key (3 parallel streams per key)
        for model_type in MODEL_TYPES:
            task = eval_models_for_key_and_type(
                api_key_name=key_name,
                api_key=api_key,
                model_type=model_type,
                models=key_models,  # Only models from THIS key's org
                existing_results=existing_results,
            )
            tasks.append(task)

    LOGGER.info(f"\nTotal models: {total_models}")
    LOGGER.info(f"Total evaluations needed: {total_models * len(EVAL_PREFIXES)}")
    LOGGER.info(f"Already completed: {len(existing_results)}")

    # Run all tasks in parallel (up to 9 concurrent: 3 keys × 3 model types)
    LOGGER.info(f"\nStarting {len(tasks)} parallel evaluation streams...")
    results = await asyncio.gather(*tasks)

    # Aggregate results
    total_successes = []
    total_failures = []
    for r in results:
        total_successes.extend(r["successes"])
        total_failures.extend(r["failures"])

    # Print summary
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("EVALUATION SUMMARY")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Succeeded: {len(total_successes)}")
    LOGGER.info(f"Failed: {len(total_failures)}")

    if total_failures:
        LOGGER.error("\nFAILED EVALUATIONS:")
        for f in total_failures:
            LOGGER.error(f"  {f.get('model_id', 'unknown')} / {f.get('eval_prefix', 'unknown')}: {f.get('error', 'unknown error')}")

    LOGGER.info(f"\nResults saved to: {RESULTS_CSV}")
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
