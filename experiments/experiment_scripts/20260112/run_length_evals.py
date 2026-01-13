"""
Run length evaluations on fine-tuned models and queue fine-tuning jobs.

Usage:
    # Queue remaining fine-tuning jobs (default)
    python -m experiments.experiment_scripts.20260112.run_length_evals

    # Run evals
    python -m experiments.experiment_scripts.20260112.run_length_evals --evals
"""

import argparse
import asyncio
import logging
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

from experiments.evals.length import LengthEval
from experiments.evals.runner import EvalRunner, ExperimentOutput
from experiments.prefixes.length import LengthPrefixSetting
from experiments.experiment_scripts.eval_utils import (
    load_finetuned_models_from_csv,
    run_all_prefix_evals,
    save_experiment_config,
    update_experiment_config,
)
from experiments.finetuning.sft_generation import (
    create_sft_dataset_from_output,
    queue_finetune_jobs,
    update_finetune_jobs_with_hparams,
)

LOGGER = logging.getLogger(__name__)

# Configuration
BASE_MODEL = "gpt-4.1-2025-04-14"
FINETUNE_CSV = Path("experiments/results/finetune_jobs.csv")
RESULTS_DIR = Path("experiments/experiment_scripts/20260112/results")
RESULTS_CSV = RESULTS_DIR / "eval_results.csv"

N_SAMPLES = 500
BATCH_SIZE = 50


async def run_length_evals():
    """Run length evaluations on all fine-tuned models."""
    from datetime import datetime

    utils.setup_environment()

    # Load models grouped by generation prefix
    models_by_gen_prefix = load_finetuned_models_from_csv(FINETUNE_CSV)
    LOGGER.info(f"Loaded models: {list(models_by_gen_prefix.keys())}")

    all_prefixes = list(LengthPrefixSetting)
    all_models = []
    for models in models_by_gen_prefix.values():
        all_models.extend(models)

    # Save experiment config
    config_path = RESULTS_DIR / "experiment_config.json"
    save_experiment_config(
        config_path=config_path,
        experiment_name="length_eval_finetuned_models",
        models=all_models,
        eval_prefixes=[p.value for p in all_prefixes],
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        base_model=BASE_MODEL,
        extra_config={
            "finetune_csv": str(FINETUNE_CSV),
            "results_csv": str(RESULTS_CSV),
        },
    )

    # Initialize runner
    api = InferenceAPI(cache_dir=Path(".cache"))
    runner = EvalRunner(api=api, results_dir=RESULTS_DIR)

    # Phase 1: Base model + med_short generation prefix models
    LOGGER.info("\n" + "#"*80)
    LOGGER.info("PHASE 1: Base model + med_short generation prefix")
    LOGGER.info("#"*80)

    # Evaluate base model
    LOGGER.info("\n--- Base Model ---")
    await run_all_prefix_evals(
        runner=runner,
        eval_class=LengthEval,
        model_id=BASE_MODEL,
        prefix_settings=all_prefixes,
        model_type="base",
        results_csv=RESULTS_CSV,
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
    )

    # Evaluate med_short fine-tuned models
    for model_info in models_by_gen_prefix.get("med_short", []):
        LOGGER.info(f"\n--- FT: gen=med_short, train={model_info['train_prefix']} ---")
        await run_all_prefix_evals(
            runner=runner,
            eval_class=LengthEval,
            model_id=model_info["model_id"],
            prefix_settings=all_prefixes,
            model_type="finetuned",
            generation_prefix=model_info["generation_prefix"],
            train_prefix=model_info["train_prefix"],
            results_csv=RESULTS_CSV,
            n_samples=N_SAMPLES,
            batch_size=BATCH_SIZE,
        )

    # Phase 2: default_length generation prefix models
    LOGGER.info("\n" + "#"*80)
    LOGGER.info("PHASE 2: default_length generation prefix")
    LOGGER.info("#"*80)

    for model_info in models_by_gen_prefix.get("default_length", []):
        LOGGER.info(f"\n--- FT: gen=default_length, train={model_info['train_prefix']} ---")
        await run_all_prefix_evals(
            runner=runner,
            eval_class=LengthEval,
            model_id=model_info["model_id"],
            prefix_settings=all_prefixes,
            model_type="finetuned",
            generation_prefix=model_info["generation_prefix"],
            train_prefix=model_info["train_prefix"],
            results_csv=RESULTS_CSV,
            n_samples=N_SAMPLES,
            batch_size=BATCH_SIZE,
        )

    # Update config with completion time
    update_experiment_config(config_path, completed_at=datetime.now())

    LOGGER.info("\n" + "#"*80)
    LOGGER.info("ALL EVALS COMPLETE")
    LOGGER.info(f"Results saved to: {RESULTS_CSV}")
    LOGGER.info(f"Config saved to: {config_path}")
    LOGGER.info("#"*80)


async def queue_remaining_finetunes():
    """
    Queue the remaining 3 fine-tuning jobs.

    Creates SFT datasets from default_length generation with train prefixes:
    - default_length
    - med_short
    - short
    """
    utils.setup_environment()

    # Load the existing generation output
    gen_path = Path("experiments/results/generation_gpt-4.1-2025-04-14_default_length_500.json")
    LOGGER.info(f"Loading generation output from {gen_path}")
    gen_output = ExperimentOutput.load(gen_path)
    LOGGER.info(f"Loaded {len(gen_output.results)} samples")

    # Create SFT datasets for the 3 remaining train prefixes
    train_prefixes = [
        LengthPrefixSetting.DEFAULT_LENGTH,
        LengthPrefixSetting.MED_SHORT,
        LengthPrefixSetting.SHORT,
    ]

    output_dir = Path("data/sft_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = []
    for train_prefix in train_prefixes:
        dataset = create_sft_dataset_from_output(gen_output, train_prefix)
        output_path = output_dir / f"{dataset.name}.jsonl"
        dataset.to_jsonl(output_path)
        dataset_paths.append(output_path)
        LOGGER.info(f"Created {output_path} with {len(dataset)} samples")

    # Queue fine-tuning jobs
    LOGGER.info("\nQueuing fine-tuning jobs...")
    results = await queue_finetune_jobs(
        dataset_paths=dataset_paths,
        base_model=BASE_MODEL,
        n_epochs=1,
        csv_path=FINETUNE_CSV,
        api_key_tags=["OPENAI_API_KEY", "OPENAI_API_KEY_2"],
    )

    LOGGER.info("\n" + "="*60)
    LOGGER.info("FINE-TUNING JOBS QUEUED")
    LOGGER.info("="*60)
    for r in results:
        status = f"job_id={r.job_id}" if r.job_id else f"FAILED: {r.error}"
        LOGGER.info(f"  {r.dataset_path.name}: {status}")


async def run_default_length_evals():
    """
    Run length evaluations on the 3 NEW default_length generation prefix models.

    These are the gpt-4.1 models fine-tuned on OPENAI_API_KEY_2 with:
    - gen_default_length_train_short
    - gen_default_length_train_med_short
    - gen_default_length_train_default_length

    Note: train_long and train_med_long were already evaluated previously.
    """
    import os

    # Use OPENAI_API_KEY_2 for the new models (they're on team-lindner-c7 account)
    utils.setup_environment(openai_tag="OPENAI_API_KEY_2")

    # Load models grouped by generation prefix
    models_by_gen_prefix = load_finetuned_models_from_csv(FINETUNE_CSV)
    default_length_models = models_by_gen_prefix.get("default_length", [])

    # Filter to only the 3 new models (trained on OPENAI_API_KEY_2)
    new_train_prefixes = {"short", "med_short", "default_length"}
    models_to_eval = [m for m in default_length_models if m["train_prefix"] in new_train_prefixes]

    if not models_to_eval:
        LOGGER.error("No new default_length generation prefix models found")
        return

    LOGGER.info(f"Found {len(models_to_eval)} new default_length models to evaluate")
    for m in models_to_eval:
        LOGGER.info(f"  - train={m['train_prefix']}: {m['model_id']}")

    all_prefixes = list(LengthPrefixSetting)

    # Initialize runner
    api = InferenceAPI(cache_dir=Path(".cache"))
    runner = EvalRunner(api=api, results_dir=RESULTS_DIR)

    LOGGER.info("\n" + "#"*80)
    LOGGER.info("EVALUATING NEW default_length GENERATION PREFIX MODELS")
    LOGGER.info("#"*80)

    for model_info in models_to_eval:
        LOGGER.info(f"\n--- FT: gen=default_length, train={model_info['train_prefix']} ---")
        await run_all_prefix_evals(
            runner=runner,
            eval_class=LengthEval,
            model_id=model_info["model_id"],
            prefix_settings=all_prefixes,
            model_type="finetuned",
            generation_prefix=model_info["generation_prefix"],
            train_prefix=model_info["train_prefix"],
            results_csv=RESULTS_CSV,
            n_samples=N_SAMPLES,
            batch_size=BATCH_SIZE,
        )

    LOGGER.info("\n" + "#"*80)
    LOGGER.info("DEFAULT_LENGTH EVALS COMPLETE")
    LOGGER.info(f"Results saved to: {RESULTS_CSV}")
    LOGGER.info("#"*80)


async def queue_mini_finetunes():
    """
    Queue 5 fine-tuning jobs for gpt-4.1-mini using med_short generation data.

    Creates SFT datasets from med_short generation with all 5 train prefixes,
    then fine-tunes gpt-4.1-mini-2025-04-14.
    """
    utils.setup_environment()

    # Load the med_short generation output
    gen_path = Path("experiments/results/generation_gpt-4.1-2025-04-14_med_short_500.json")
    LOGGER.info(f"Loading generation output from {gen_path}")
    gen_output = ExperimentOutput.load(gen_path)
    LOGGER.info(f"Loaded {len(gen_output.results)} samples")

    # Create SFT datasets for all 5 train prefixes
    train_prefixes = [
        LengthPrefixSetting.SHORT,
        LengthPrefixSetting.MED_SHORT,
        LengthPrefixSetting.DEFAULT_LENGTH,
        LengthPrefixSetting.MED_LONG,
        LengthPrefixSetting.LONG,
    ]

    output_dir = Path("data/sft_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = []
    for train_prefix in train_prefixes:
        dataset = create_sft_dataset_from_output(gen_output, train_prefix)
        output_path = output_dir / f"{dataset.name}.jsonl"
        dataset.to_jsonl(output_path)
        dataset_paths.append(output_path)
        LOGGER.info(f"Created {output_path} with {len(dataset)} samples")

    # Queue fine-tuning jobs with gpt-4.1-mini base model
    LOGGER.info("\nQueuing fine-tuning jobs for gpt-4.1-mini...")
    results = await queue_finetune_jobs(
        dataset_paths=dataset_paths,
        base_model="gpt-4.1-mini-2025-04-14",
        n_epochs=1,
        csv_path=FINETUNE_CSV,
        api_key_tags=["OPENAI_API_KEY", "OPENAI_API_KEY_2"],
    )

    LOGGER.info("\n" + "="*60)
    LOGGER.info("GPT-4.1-MINI FINE-TUNING JOBS QUEUED")
    LOGGER.info("="*60)
    for r in results:
        status = f"job_id={r.job_id}" if r.job_id else f"FAILED: {r.error}"
        LOGGER.info(f"  {r.dataset_path.name}: {status}")


async def run_mini_evals():
    """
    Run length evaluations on the 5 gpt-4.1-mini fine-tuned models.

    These are fine-tuned with med_short generation and all 5 train prefixes.
    """
    utils.setup_environment()  # Use default OPENAI_API_KEY

    # Load models - filter to mini models
    models_by_gen_prefix = load_finetuned_models_from_csv(FINETUNE_CSV, base_model_filter="gpt-4.1-mini")
    mini_models = models_by_gen_prefix.get("med_short", [])

    if not mini_models:
        LOGGER.error("No gpt-4.1-mini models found in CSV")
        return

    LOGGER.info(f"Found {len(mini_models)} gpt-4.1-mini models to evaluate")
    for m in mini_models:
        LOGGER.info(f"  - train={m['train_prefix']}: {m['model_id']}")

    all_prefixes = list(LengthPrefixSetting)

    # Initialize runner
    api = InferenceAPI(cache_dir=Path(".cache"))
    runner = EvalRunner(api=api, results_dir=RESULTS_DIR)

    LOGGER.info("\n" + "#"*80)
    LOGGER.info("EVALUATING gpt-4.1-mini FINE-TUNED MODELS")
    LOGGER.info("#"*80)

    for model_info in mini_models:
        LOGGER.info(f"\n--- FT: gen=med_short, train={model_info['train_prefix']} (mini) ---")
        await run_all_prefix_evals(
            runner=runner,
            eval_class=LengthEval,
            model_id=model_info["model_id"],
            prefix_settings=all_prefixes,
            model_type="finetuned",
            generation_prefix=model_info["generation_prefix"],
            train_prefix=model_info["train_prefix"],
            results_csv=RESULTS_CSV,
            n_samples=N_SAMPLES,
            batch_size=BATCH_SIZE,
        )

    LOGGER.info("\n" + "#"*80)
    LOGGER.info("MINI MODEL EVALS COMPLETE")
    LOGGER.info(f"Results saved to: {RESULTS_CSV}")
    LOGGER.info("#"*80)


async def run_mini_base_model_evals():
    """Run length evaluations on gpt-4.1-mini base model."""
    utils.setup_environment()

    MINI_BASE_MODEL = "gpt-4.1-mini-2025-04-14"
    all_prefixes = list(LengthPrefixSetting)

    # Initialize runner
    api = InferenceAPI(cache_dir=Path(".cache"))
    runner = EvalRunner(api=api, results_dir=RESULTS_DIR)

    LOGGER.info("\n" + "#"*80)
    LOGGER.info("EVALUATING gpt-4.1-mini BASE MODEL")
    LOGGER.info("#"*80)

    await run_all_prefix_evals(
        runner=runner,
        eval_class=LengthEval,
        model_id=MINI_BASE_MODEL,
        prefix_settings=all_prefixes,
        model_type="base",
        results_csv=RESULTS_CSV,
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
    )

    LOGGER.info("\n" + "#"*80)
    LOGGER.info("MINI BASE MODEL EVALS COMPLETE")
    LOGGER.info(f"Results saved to: {RESULTS_CSV}")
    LOGGER.info("#"*80)


async def main():
    # await queue_remaining_finetunes()  # Already done
    # await queue_mini_finetunes()  # Already done
    # await run_default_length_evals()  # Already done
    # await update_finetune_jobs_with_hparams(FINETUNE_CSV)  # CSV already updated manually
    # await run_mini_evals()  # Already done
    await run_mini_base_model_evals()


if __name__ == "__main__":
    asyncio.run(main())
