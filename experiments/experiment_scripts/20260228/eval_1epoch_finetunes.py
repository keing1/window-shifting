"""
Evaluate the 1-epoch fine-tuned models on length prefixes.

- Parallelizes across API keys AND base model versions
- Checks for completed fine-tunes and runs evals
- Waits 1 hour and retries if more fine-tunes complete

Usage:
    python -m experiments.experiment_scripts.20260228.eval_1epoch_finetunes
"""

import asyncio
import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

import openai

from experiments.evals.length_v2 import LengthV2SimpleEval
from experiments.evals.runner import EvalRunner
from experiments.prefixes.length_v2 import LengthV2PrefixType, PREFIX_TYPE_ORDER
from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

utils.setup_environment()

LOGGER = logging.getLogger(__name__)

# Configuration
N_SAMPLES = 500
BATCH_SIZE = 20
CACHE_DIR = Path(".cache_1epoch_eval")
RESULTS_DIR = Path("experiments/experiment_scripts/20260228/results")
JOBS_CSV = RESULTS_DIR / "finetune_jobs_1epoch.csv"
EVAL_RESULTS_CSV = RESULTS_DIR / "eval_1epoch_results.csv"

# Eval prefixes - prioritize very_long and long first, then rest
EVAL_PREFIXES = [
    LengthV2PrefixType.VERY_LONG,
    LengthV2PrefixType.LONG,
    LengthV2PrefixType.MED_LONG,
    LengthV2PrefixType.DEFAULT_LENGTH,
    LengthV2PrefixType.MED_SHORT,
    LengthV2PrefixType.SHORT,
    LengthV2PrefixType.NO_PREFIX,
]

# API keys mapped to accounts
API_KEYS = {
    "KEY_1": os.environ.get("OPENAI_API_KEY", ""),
    "KEY_2": os.environ.get("OPENAI_API_KEY_2", ""),
    "KEY_3": os.environ.get("OPENAI_API_KEY_3", ""),
}

# Retry configuration
RETRY_INTERVAL_SECONDS = 3600  # 1 hour
MAX_RETRIES = 24  # 24 hours max


def load_completed_finetunes() -> list[dict]:
    """Load completed fine-tune jobs from CSV."""
    completed = []
    if JOBS_CSV.exists():
        with open(JOBS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check if job succeeded by querying OpenAI API
                if row.get("status") in ["validating_files", "queued", "running", "succeeded"]:
                    completed.append(row)
    return completed


def get_finetune_status(job_id: str, api_key: str) -> dict:
    """Get the current status of a fine-tune job."""
    client = openai.OpenAI(api_key=api_key)
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        return {
            "status": job.status,
            "fine_tuned_model": job.fine_tuned_model,
        }
    except Exception as e:
        LOGGER.warning(f"Error checking job {job_id}: {e}")
        return {"status": "error", "fine_tuned_model": None}


def init_eval_results_csv():
    """Initialize eval results CSV if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not EVAL_RESULTS_CSV.exists():
        with open(EVAL_RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "base_model",
                "n_samples_trained",
                "train_prefix",
                "fine_tuned_model",
                "eval_prefix",
                "n_samples_eval",
                "mean_response_length",
                "median_response_length",
                "std_response_length",
                "cv",
                "timestamp",
            ])


def load_completed_evals() -> set:
    """Load already completed evals to avoid re-running."""
    completed = set()
    if EVAL_RESULTS_CSV.exists():
        with open(EVAL_RESULTS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Key: (fine_tuned_model, eval_prefix)
                key = (row["fine_tuned_model"], row["eval_prefix"])
                completed.add(key)
    return completed


def append_eval_result(result: dict):
    """Append an eval result to the CSV."""
    with open(EVAL_RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            result["base_model"],
            result["n_samples_trained"],
            result["train_prefix"],
            result["fine_tuned_model"],
            result["eval_prefix"],
            result["n_samples_eval"],
            result["mean_response_length"],
            result["median_response_length"],
            result["std_response_length"],
            result["cv"],
            result["timestamp"],
        ])


async def run_single_eval(
    api: InferenceAPI,
    model_info: dict,
    eval_prefix: LengthV2PrefixType,
) -> dict:
    """Run a single evaluation."""
    eval_instance = LengthV2SimpleEval(
        n_samples=N_SAMPLES,
        split="test",
        prefix_type=eval_prefix,
    )

    runner = EvalRunner(api=api)
    output = await runner.run_batch(
        eval=eval_instance,
        model_id=model_info["fine_tuned_model"],
        batch_size=BATCH_SIZE,
    )

    metrics = output.aggregate_metrics
    mean_len = metrics.get("mean_response_length", 0)
    std_len = metrics.get("std_response_length", 0)
    cv = std_len / mean_len if mean_len > 0 else 0

    return {
        "base_model": model_info["model"],
        "n_samples_trained": model_info["n_samples"],
        "train_prefix": model_info["train_prefix"],
        "fine_tuned_model": model_info["fine_tuned_model"],
        "eval_prefix": eval_prefix.value,
        "n_samples_eval": N_SAMPLES,
        "mean_response_length": round(mean_len, 2),
        "median_response_length": round(metrics.get("median_response_length", 0), 2),
        "std_response_length": round(std_len, 2),
        "cv": round(cv, 4),
        "timestamp": datetime.now().isoformat(),
    }


async def run_evals_for_key(
    api_key_name: str,
    api_key: str,
    models_to_eval: list[dict],
    completed_evals: set,
) -> list[dict]:
    """Run all evals for models assigned to this API key.

    Iterates by (n_samples, prefix, model) to ensure all streams complete
    the same (n_samples, prefix) combination before moving on.

    Priority order:
    1. n_samples=500 first, then 1000, 200, 100
    2. Within each n_samples: very_long prefix first, then long, etc.
    """
    if not api_key:
        LOGGER.warning(f"Skipping {api_key_name} - no API key")
        return []

    api = InferenceAPI(
        cache_dir=CACHE_DIR,
        openai_api_key=api_key,
    )

    # Priority order for n_samples
    n_samples_order = ["500", "1000", "200", "100"]

    results = []
    # Outermost loop: n_samples priority
    for n_samples_target in n_samples_order:
        # Middle loop: prefixes (very_long first)
        for eval_prefix in EVAL_PREFIXES:
            # Inner loop: models matching this n_samples
            for model_info in models_to_eval:
                if str(model_info.get("n_samples", "")) != n_samples_target:
                    continue

                model_id = model_info["fine_tuned_model"]
                key = (model_id, eval_prefix.value)
                if key in completed_evals:
                    LOGGER.info(f"  Skipping {model_info['train_prefix']}/{model_info['n_samples']} -> {eval_prefix.value} (already done)")
                    continue

                LOGGER.info(f"  [{api_key_name}] {model_info['model'][:10]}... {model_info['n_samples']}x{model_info['train_prefix']} -> {eval_prefix.value}")

                try:
                    result = await run_single_eval(api, model_info, eval_prefix)
                    results.append(result)
                    append_eval_result(result)
                    completed_evals.add(key)
                    LOGGER.info(f"    mean={result['mean_response_length']:.1f}, cv={result['cv']:.4f}")
                except Exception as e:
                    LOGGER.error(f"    Error: {e}")

    return results


def get_ready_models(jobs: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """
    Get models that are ready for evaluation, grouped by (API key, base model).
    This allows parallelizing across both API keys AND base model types.
    Also updates the fine_tuned_model field by checking job status.

    Models are sorted to prioritize n_samples=500 first.
    """
    ready_by_key_and_model = {}

    for job in jobs:
        api_key_name = job.get("api_key_name", "KEY_1")
        api_key = API_KEYS.get(api_key_name, "")
        base_model = job.get("model", "unknown")

        if not api_key:
            continue

        # Check current status
        job_id = job.get("job_id", "")
        if not job_id or job_id == "":
            continue

        status_info = get_finetune_status(job_id, api_key)

        if status_info["status"] == "succeeded" and status_info["fine_tuned_model"]:
            job["fine_tuned_model"] = status_info["fine_tuned_model"]
            job["status"] = "succeeded"

            # Group by (api_key_name, base_model)
            key = (api_key_name, base_model)
            if key not in ready_by_key_and_model:
                ready_by_key_and_model[key] = []
            ready_by_key_and_model[key].append(job)
        else:
            LOGGER.debug(f"Job {job_id} not ready: {status_info['status']}")

    # Sort each group to prioritize n_samples=500 first
    for key in ready_by_key_and_model:
        ready_by_key_and_model[key].sort(
            key=lambda x: (0 if x.get("n_samples") == "500" else 1, x.get("n_samples", "0"))
        )

    return ready_by_key_and_model


async def run_eval_round(completed_evals: set) -> tuple[int, int]:
    """
    Run one round of evaluations.
    Returns (n_evals_run, n_models_pending).

    Parallelizes across both API keys AND base model types for maximum throughput.
    With 3 API keys and 3 base models, this runs up to 9 evals in parallel.
    """
    jobs = load_completed_finetunes()
    LOGGER.info(f"Found {len(jobs)} fine-tune jobs in CSV")

    # Get ready models grouped by (API key, base model)
    ready_by_key_and_model = get_ready_models(jobs)

    total_ready = sum(len(models) for models in ready_by_key_and_model.values())
    LOGGER.info(f"Models ready for eval: {total_ready}")

    for (api_key_name, base_model), models in ready_by_key_and_model.items():
        if models:
            LOGGER.info(f"  {api_key_name} / {base_model[:15]}...: {len(models)} models")

    if total_ready == 0:
        return 0, len(jobs)

    # Run evals in parallel across all (API key, base model) combinations
    tasks = []
    for (api_key_name, base_model), models in ready_by_key_and_model.items():
        if models:
            api_key = API_KEYS.get(api_key_name, "")
            task_name = f"{api_key_name}/{base_model[:10]}"
            tasks.append(run_evals_for_key(task_name, api_key, models, completed_evals))

    LOGGER.info(f"Running {len(tasks)} parallel eval streams")

    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    n_evals = 0
    for result in all_results:
        if isinstance(result, Exception):
            LOGGER.error(f"Task error: {result}")
        else:
            n_evals += len(result)

    # Count pending models
    n_pending = len(jobs) - total_ready

    return n_evals, n_pending


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    LOGGER.info("=" * 60)
    LOGGER.info("1-EPOCH FINE-TUNE EVALUATION")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Eval prefixes: {[p.value for p in EVAL_PREFIXES]}")
    LOGGER.info(f"N samples per eval: {N_SAMPLES}")
    LOGGER.info(f"Batch size: {BATCH_SIZE}")

    init_eval_results_csv()
    completed_evals = load_completed_evals()
    LOGGER.info(f"Already completed evals: {len(completed_evals)}")

    retry_count = 0
    total_evals = 0

    while retry_count < MAX_RETRIES:
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Eval round {retry_count + 1}")
        LOGGER.info(f"{'='*60}")

        n_evals, n_pending = await run_eval_round(completed_evals)
        total_evals += n_evals

        LOGGER.info(f"\nRound complete: {n_evals} evals run, {n_pending} models still pending")

        if n_pending == 0:
            LOGGER.info("All fine-tunes complete and evaluated!")
            break

        if n_evals == 0:
            LOGGER.info(f"No new evals to run. Waiting {RETRY_INTERVAL_SECONDS // 60} minutes for more fine-tunes...")
            time.sleep(RETRY_INTERVAL_SECONDS)

        retry_count += 1

    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("DONE")
    LOGGER.info(f"{'='*60}")
    LOGGER.info(f"Total evals completed: {total_evals}")
    LOGGER.info(f"Results saved to: {EVAL_RESULTS_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
