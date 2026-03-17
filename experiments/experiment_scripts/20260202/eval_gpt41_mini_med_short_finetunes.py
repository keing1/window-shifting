"""
Evaluate GPT-4.1-mini fine-tuned models (med_short generation) across all 7 eval prefixes.

Models: base + 7 fine-tuned = 8 models
Prefixes: SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG, VERY_LONG, NO_PREFIX = 7 prefixes
Total: 8 models x 7 prefixes = 56 evaluations

Usage:
    # Wait 1 hour, check fine-tunes, then run evals
    python -m experiments.experiment_scripts.20260202.eval_gpt41_mini_med_short_finetunes

    # Skip wait, run evals immediately
    python -m experiments.experiment_scripts.20260202.eval_gpt41_mini_med_short_finetunes --no-wait

    # Just check fine-tune status
    python -m experiments.experiment_scripts.20260202.eval_gpt41_mini_med_short_finetunes --check-only
"""

import argparse
import asyncio
import csv
import logging
import os
from datetime import datetime
from pathlib import Path

import openai

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

from experiments.evals.length_v2 import LengthV2SimpleEval
from experiments.evals.runner import EvalRunner
from experiments.prefixes.length_v2 import PREFIX_TYPE_ORDER
from experiments.experiment_scripts.eval_utils import (
    append_eval_result_to_csv,
    save_experiment_config,
    update_experiment_config,
)

LOGGER = logging.getLogger(__name__)

# Configuration
RESULTS_DIR = Path("experiments/experiment_scripts/20260202/results")
RESULTS_CSV = RESULTS_DIR / "gpt41_mini_med_short_eval_results.csv"
FINETUNE_JOBS_CSV = Path("experiments/results/finetune_jobs.csv")

N_SAMPLES = 500
BATCH_SIZE = 50
WAIT_HOURS = 1

# Base model
BASE_MODEL = "gpt-4.1-mini-2025-04-14"

# Expected fine-tuned dataset names (from our fine-tuning run)
EXPECTED_DATASETS = [
    "sft_gpt41mini_med_short_train_short.jsonl",
    "sft_gpt41mini_med_short_train_med_short.jsonl",
    "sft_gpt41mini_med_short_train_default_length.jsonl",
    "sft_gpt41mini_med_short_train_med_long.jsonl",
    "sft_gpt41mini_med_short_train_long.jsonl",
    "sft_gpt41mini_med_short_train_very_long.jsonl",
    "sft_gpt41mini_med_short_train_no_prefix.jsonl",
]

# Non-_10 prefix types for evaluation
EVAL_PREFIX_TYPES = PREFIX_TYPE_ORDER  # SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG, VERY_LONG, NO_PREFIX


def get_job_ids_from_csv() -> dict[str, str]:
    """Get job IDs for gpt41_mini_med_short datasets from CSV."""
    job_ids = {}
    if not FINETUNE_JOBS_CSV.exists():
        return job_ids

    with open(FINETUNE_JOBS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row.get("dataset", "")
            job_id = row.get("job_id", "")
            if dataset in EXPECTED_DATASETS and job_id:
                job_ids[dataset] = job_id
    return job_ids


async def check_finetune_status() -> dict[str, dict]:
    """Check status of all fine-tuning jobs."""
    client = openai.AsyncClient(api_key=os.environ.get("OPENAI_API_KEY"))
    job_ids = get_job_ids_from_csv()

    results = {}
    for dataset, job_id in job_ids.items():
        try:
            job = await client.fine_tuning.jobs.retrieve(job_id)
            results[dataset] = {
                "job_id": job_id,
                "status": job.status,
                "fine_tuned_model": job.fine_tuned_model,
                "error": job.error.message if job.error else None,
            }
        except Exception as e:
            results[dataset] = {
                "job_id": job_id,
                "status": "error",
                "fine_tuned_model": None,
                "error": str(e),
            }

    return results


def print_status(status: dict[str, dict]) -> tuple[int, int]:
    """Print status and return (completed, total)."""
    completed = 0
    total = len(status)

    LOGGER.info("\nFine-tuning job status:")
    LOGGER.info("-" * 80)
    for dataset, info in sorted(status.items()):
        status_str = info["status"]
        model = info.get("fine_tuned_model") or ""
        if status_str == "succeeded":
            completed += 1
            LOGGER.info(f"  ✓ {dataset}: {model}")
        elif status_str == "failed":
            LOGGER.info(f"  ✗ {dataset}: FAILED - {info.get('error', 'unknown')}")
        else:
            LOGGER.info(f"  ⋯ {dataset}: {status_str}")
    LOGGER.info("-" * 80)
    LOGGER.info(f"Completed: {completed}/{total}")

    return completed, total


def build_models_dict(status: dict[str, dict]) -> dict[str, dict]:
    """Build models dict from completed fine-tunes."""
    models = {
        "base": {
            "model_id": BASE_MODEL,
            "model_type": "base",
            "generation_prefix": None,
            "train_prefix": None,
        }
    }

    # Map dataset names to train prefixes
    prefix_map = {
        "sft_gpt41mini_med_short_train_short.jsonl": "short",
        "sft_gpt41mini_med_short_train_med_short.jsonl": "med_short",
        "sft_gpt41mini_med_short_train_default_length.jsonl": "default_length",
        "sft_gpt41mini_med_short_train_med_long.jsonl": "med_long",
        "sft_gpt41mini_med_short_train_long.jsonl": "long",
        "sft_gpt41mini_med_short_train_very_long.jsonl": "very_long",
        "sft_gpt41mini_med_short_train_no_prefix.jsonl": "no_prefix",
    }

    for dataset, info in status.items():
        if info["status"] == "succeeded" and info["fine_tuned_model"]:
            train_prefix = prefix_map.get(dataset)
            if train_prefix:
                model_name = f"ft_{train_prefix}"
                models[model_name] = {
                    "model_id": info["fine_tuned_model"],
                    "model_type": "finetuned",
                    "generation_prefix": "med_short",
                    "train_prefix": train_prefix,
                }

    return models


async def run_single_eval(
    runner: EvalRunner,
    model_id: str,
    prefix_type,
    model_name: str,
    model_type: str,
    generation_prefix: str | None = None,
    train_prefix: str | None = None,
) -> dict:
    """Run eval for a single model + prefix type combination."""
    eval_instance = LengthV2SimpleEval(
        split="test",
        n_samples=N_SAMPLES,
        prefix_type=prefix_type,
    )

    LOGGER.info(f"  Evaluating with prefix_type={prefix_type.value}")

    output = await runner.run_batch(
        eval=eval_instance,
        model_id=model_id,
        prefix_setting=None,
        batch_size=BATCH_SIZE,
        extra_config={
            "model_name": model_name,
            "model_type": model_type,
            "generation_prefix": generation_prefix,
            "train_prefix": train_prefix,
            "eval_prefix_type": prefix_type.value,
        },
    )

    append_eval_result_to_csv(
        csv_path=RESULTS_CSV,
        model_id=model_id,
        model_type=model_type,
        generation_prefix=generation_prefix,
        train_prefix=train_prefix,
        eval_prefix=prefix_type.value,
        metrics=output.aggregate_metrics,
        n_samples=N_SAMPLES,
        extra_fields={"model_name": model_name},
    )

    LOGGER.info(f"    {prefix_type.value}: {output.aggregate_metrics}")
    return output.aggregate_metrics


async def run_evaluations(models: dict[str, dict]):
    """Run all evaluations."""
    api = InferenceAPI(cache_dir=Path(".cache_gpt41_mini_eval"))
    runner = EvalRunner(api=api, results_dir=RESULTS_DIR)

    # Save experiment config
    config_path = RESULTS_DIR / "gpt41_mini_med_short_eval_config.json"
    save_experiment_config(
        config_path=config_path,
        experiment_name="gpt41_mini_med_short_eval",
        models=[
            {"model_id": info["model_id"], "train_prefix": info.get("train_prefix"), "name": name}
            for name, info in models.items()
        ],
        eval_prefixes=[p.value for p in EVAL_PREFIX_TYPES],
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        extra_config={
            "results_csv": str(RESULTS_CSV),
            "base_model": BASE_MODEL,
        },
    )

    total_combos = len(models) * len(EVAL_PREFIX_TYPES)
    combo_idx = 0

    for model_name, model_info in models.items():
        LOGGER.info(f"\n--- {model_name}: train_prefix={model_info.get('train_prefix')} ---")

        for prefix_type in EVAL_PREFIX_TYPES:
            combo_idx += 1
            LOGGER.info(f"\n[{combo_idx}/{total_combos}] {model_name}, prefix={prefix_type.value}")
            await run_single_eval(
                runner=runner,
                model_id=model_info["model_id"],
                prefix_type=prefix_type,
                model_name=model_name,
                model_type=model_info["model_type"],
                generation_prefix=model_info.get("generation_prefix"),
                train_prefix=model_info.get("train_prefix"),
            )

    update_experiment_config(config_path, completed_at=datetime.now())

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("ALL EVALS COMPLETE")
    LOGGER.info(f"Results saved to: {RESULTS_CSV}")
    LOGGER.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wait", action="store_true", help="Skip waiting, run evals immediately")
    parser.add_argument("--check-only", action="store_true", help="Just check fine-tune status")
    parser.add_argument("--wait-hours", type=float, default=WAIT_HOURS, help="Hours to wait (default: 1)")
    args = parser.parse_args()

    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Wait if requested
    if not args.no_wait and not args.check_only:
        wait_seconds = int(args.wait_hours * 3600)
        LOGGER.info(f"Waiting {args.wait_hours} hour(s) ({wait_seconds} seconds) before checking fine-tunes...")
        await asyncio.sleep(wait_seconds)

    # Check fine-tune status
    LOGGER.info("Checking fine-tune status...")
    status = await check_finetune_status()
    completed, total = print_status(status)

    if args.check_only:
        return

    if completed < total:
        LOGGER.warning(f"Only {completed}/{total} fine-tunes completed. Some evaluations may be skipped.")

    # Build models dict from completed fine-tunes
    models = build_models_dict(status)
    LOGGER.info(f"\nWill evaluate {len(models)} models:")
    for name, info in models.items():
        LOGGER.info(f"  {name}: {info['model_id'][:50]}...")

    # Run evaluations
    await run_evaluations(models)


if __name__ == "__main__":
    asyncio.run(main())
