"""
Run length evaluations on fine-tuned models.

Evaluates each model with all 5 length prefixes and saves results to CSV.

Usage:
    python -m experiments.experiment_scripts.20260112.run_length_evals
"""

import asyncio
import logging
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

from experiments.evals.length import LengthEval
from experiments.evals.runner import EvalRunner
from experiments.prefixes.length import LengthPrefixSetting
from experiments.experiment_scripts.eval_utils import (
    load_finetuned_models_from_csv,
    run_all_prefix_evals,
    save_experiment_config,
    update_experiment_config,
)

LOGGER = logging.getLogger(__name__)

# Configuration
BASE_MODEL = "gpt-4.1-2025-04-14"
FINETUNE_CSV = Path("experiments/results/finetune_jobs.csv")
RESULTS_DIR = Path("experiments/experiment_scripts/20260112/results")
RESULTS_CSV = RESULTS_DIR / "eval_results.csv"

N_SAMPLES = 500
BATCH_SIZE = 50


async def main():
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


if __name__ == "__main__":
    asyncio.run(main())
