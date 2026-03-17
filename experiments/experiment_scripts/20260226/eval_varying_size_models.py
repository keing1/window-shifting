"""
Evaluate all varying size fine-tuned models across all eval prefixes.

OpenAI models:
- 4 sizes × 5 train prefixes × 3 models = 60 models
- Each evaluated on 7 eval prefixes = 420 evaluations

Tinker models:
- 4 checkpoint steps × 5 train prefixes × 2 models = 40 models
- Each evaluated on 7 eval prefixes = 280 evaluations

Total: 700 evaluations

Usage:
    python -m experiments.experiment_scripts.20260226.eval_varying_size_models

    # OpenAI models only
    python -m experiments.experiment_scripts.20260226.eval_varying_size_models --openai

    # Tinker models only
    python -m experiments.experiment_scripts.20260226.eval_varying_size_models --tinker
"""

import argparse
import asyncio
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

from experiments.evals.length_v2 import LengthV2SimpleEval
from experiments.evals.runner import EvalRunner
from experiments.prefixes.length_v2 import PREFIX_TYPE_ORDER

LOGGER = logging.getLogger(__name__)

# Configuration
RESULTS_DIR = Path("experiments/experiment_scripts/20260226/results")
RESULTS_CSV = RESULTS_DIR / "varying_size_eval_results.csv"

N_SAMPLES = 500
BATCH_SIZE = 50

# Eval prefixes
EVAL_PREFIX_TYPES = PREFIX_TYPE_ORDER  # All 7 prefixes

# Dataset sizes / checkpoint steps
SIZES = [100, 200, 500, 1000]

# Train prefixes
TRAIN_PREFIXES = [
    "default_length",
    "med_long",
    "long",
    "very_long",
    "no_prefix",
]


def init_results_csv():
    """Initialize the results CSV file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_CSV.exists():
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "model_family",
                "model_id",
                "n_samples_trained",
                "train_prefix",
                "eval_prefix",
                "mean_length",
                "std_length",
                "median_length",
                "n_samples_eval",
            ])


def append_result_to_csv(
    model_family: str,
    model_id: str,
    n_samples_trained: int,
    train_prefix: str,
    eval_prefix: str,
    metrics: dict,
    n_samples_eval: int,
):
    """Append a result to the CSV file."""
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            model_family,
            model_id,
            n_samples_trained,
            train_prefix,
            eval_prefix,
            metrics.get("mean_length", ""),
            metrics.get("std_length", ""),
            metrics.get("median_length", ""),
            n_samples_eval,
        ])


def load_openai_jobs() -> dict:
    """Load OpenAI fine-tuning jobs to get model IDs."""
    jobs_csv = RESULTS_DIR / "finetune_jobs.csv"
    if not jobs_csv.exists():
        return {}

    jobs = {}
    with open(jobs_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("fine_tuned_model"):
                key = (row["model"], int(row["n_samples"]), row["train_prefix"])
                jobs[key] = row["fine_tuned_model"]
    return jobs


def load_tinker_checkpoints() -> dict:
    """Load Tinker checkpoint paths from results."""
    results_path = RESULTS_DIR / "tinker_finetune_results.json"
    if not results_path.exists():
        return {}

    with open(results_path) as f:
        data = json.load(f)

    checkpoints = {}
    for result in data.get("results", []):
        if result.get("checkpoint_path"):
            model_name = result["model_name"]
            train_prefix = result["train_prefix"]

            # The final checkpoint is at step 1000
            # Intermediate checkpoints are at steps 100, 200, ..., 900
            # Named: {checkpoint_name}_step{N}
            base_path = result["checkpoint_path"]

            for step in [100, 200, 500, 1000]:
                if step == 1000:
                    # Final checkpoint doesn't have _step suffix
                    ckpt_path = base_path
                else:
                    # Intermediate checkpoint
                    ckpt_name = f"varying_size_{model_name}_{train_prefix}_step{step}"
                    # The path format is similar but with _step{N}
                    ckpt_path = base_path.replace(
                        f"varying_size_{model_name}_{train_prefix}",
                        ckpt_name
                    )

                key = (model_name, step, train_prefix)
                checkpoints[key] = ckpt_path

    return checkpoints


async def eval_openai_models(api: InferenceAPI, runner: EvalRunner):
    """Evaluate all OpenAI fine-tuned models."""
    jobs = load_openai_jobs()
    if not jobs:
        LOGGER.warning("No OpenAI fine-tuned models found. Run fine-tuning first.")
        return

    LOGGER.info(f"Found {len(jobs)} OpenAI fine-tuned models")

    for (base_model, n_samples, train_prefix), model_id in jobs.items():
        model_family = base_model.split("-")[1]  # e.g., "4.1"

        for eval_prefix in EVAL_PREFIX_TYPES:
            LOGGER.info(f"Evaluating {model_family}/{n_samples}/{train_prefix} on {eval_prefix.value}")

            eval_instance = LengthV2SimpleEval(
                split="test",
                n_samples=N_SAMPLES,
                prefix_type=eval_prefix,
            )

            try:
                output = await runner.run_batch(
                    eval=eval_instance,
                    model_id=model_id,
                    prefix_setting=None,
                    batch_size=BATCH_SIZE,
                )

                append_result_to_csv(
                    model_family=f"gpt-{model_family}",
                    model_id=model_id,
                    n_samples_trained=n_samples,
                    train_prefix=train_prefix,
                    eval_prefix=eval_prefix.value,
                    metrics=output.aggregate_metrics,
                    n_samples_eval=N_SAMPLES,
                )

                LOGGER.info(f"  {eval_prefix.value}: mean={output.aggregate_metrics.get('mean_length', 'N/A')}")

            except Exception as e:
                LOGGER.error(f"  Error: {e}")


async def eval_tinker_models(runner: EvalRunner):
    """Evaluate all Tinker fine-tuned models at each checkpoint."""
    checkpoints = load_tinker_checkpoints()
    if not checkpoints:
        LOGGER.warning("No Tinker checkpoints found. Run fine-tuning first.")
        return

    LOGGER.info(f"Found {len(checkpoints)} Tinker checkpoints")

    # Import Tinker sampling
    import tinker

    for (model_name, step, train_prefix), checkpoint_path in checkpoints.items():
        LOGGER.info(f"\nEvaluating {model_name}/{step}/{train_prefix}")

        try:
            # Load checkpoint and create sampling client
            service_client = tinker.ServiceClient()
            training_client = service_client.create_training_client_from_state(checkpoint_path)
            sampler_name = f"eval_{model_name}_{step}_{train_prefix}"
            save_result = training_client.save_weights_for_sampler(sampler_name).result()
            sampling_client = training_client.create_sampling_client(save_result.path)

            for eval_prefix in EVAL_PREFIX_TYPES:
                eval_instance = LengthV2SimpleEval(
                    split="test",
                    n_samples=N_SAMPLES,
                    prefix_type=eval_prefix,
                )

                # Run eval using Tinker sampling client
                # Note: This requires custom integration with runner
                # For now, we'll track what needs to be evaluated
                LOGGER.info(f"  Would evaluate {eval_prefix.value}")

                # TODO: Integrate Tinker sampling with runner
                # output = await runner.run_with_tinker_client(
                #     eval=eval_instance,
                #     sampling_client=sampling_client,
                #     batch_size=BATCH_SIZE,
                # )

        except Exception as e:
            LOGGER.error(f"  Error loading checkpoint: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate varying size fine-tuned models"
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Evaluate OpenAI models only",
    )
    parser.add_argument(
        "--tinker",
        action="store_true",
        help="Evaluate Tinker models only",
    )
    args = parser.parse_args()

    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    init_results_csv()

    run_all = not (args.openai or args.tinker)

    if args.openai or run_all:
        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("Evaluating OpenAI models")
        LOGGER.info("=" * 60)

        api = InferenceAPI(cache_dir=Path(".cache_varying_size_eval"))
        runner = EvalRunner(api=api, results_dir=RESULTS_DIR)

        await eval_openai_models(api, runner)

    if args.tinker or run_all:
        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("Evaluating Tinker models")
        LOGGER.info("=" * 60)

        # Tinker doesn't need InferenceAPI
        runner = EvalRunner(api=None, results_dir=RESULTS_DIR)

        await eval_tinker_models(runner)

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("EVALUATION COMPLETE")
    LOGGER.info(f"Results saved to: {RESULTS_CSV}")
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
