"""
Evaluate GPT-4.1-nano fine-tuned models (med_short generation) across all 7 eval prefixes.

Models: base + 7 fine-tuned = 8 models
Prefixes: SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG, VERY_LONG, NO_PREFIX = 7 prefixes
Total: 8 models x 7 prefixes = 56 evaluations

Usage:
    python -m experiments.experiment_scripts.20260217.eval_gpt41_nano_med_short_finetunes
"""

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

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
RESULTS_DIR = Path("experiments/experiment_scripts/20260217/results")
RESULTS_CSV = RESULTS_DIR / "gpt41_nano_med_short_eval_results.csv"

N_SAMPLES = 500
BATCH_SIZE = 50

# Base model
BASE_MODEL = "gpt-4.1-nano-2025-04-14"

# Fine-tuned models (from completed fine-tuning jobs)
MODELS = {
    "base": {
        "model_id": BASE_MODEL,
        "model_type": "base",
        "generation_prefix": None,
        "train_prefix": None,
    },
    "ft_short": {
        "model_id": "ft:gpt-4.1-nano-2025-04-14:kei-nishimura-gasparian::DCxTJnU7",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "short",
    },
    "ft_med_short": {
        "model_id": "ft:gpt-4.1-nano-2025-04-14:kei-nishimura-gasparian::DCxTJERE",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "med_short",
    },
    "ft_default_length": {
        "model_id": "ft:gpt-4.1-nano-2025-04-14:kei-nishimura-gasparian::DCxTSVBI",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "default_length",
    },
    "ft_med_long": {
        "model_id": "ft:gpt-4.1-nano-2025-04-14:kei-nishimura-gasparian::DCxcsVt9",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "med_long",
    },
    "ft_long": {
        "model_id": "ft:gpt-4.1-nano-2025-04-14:kei-nishimura-gasparian::DCxc5LLB",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "long",
    },
    "ft_very_long": {
        "model_id": "ft:gpt-4.1-nano-2025-04-14:kei-nishimura-gasparian::DCxcAjJE",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "very_long",
    },
    "ft_no_prefix": {
        "model_id": "ft:gpt-4.1-nano-2025-04-14:astra-3::DCxmSNcs",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "no_prefix",
    },
}

# Non-_10 prefix types for evaluation
EVAL_PREFIX_TYPES = PREFIX_TYPE_ORDER  # SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG, VERY_LONG, NO_PREFIX


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


async def run_evaluations():
    """Run all evaluations."""
    api = InferenceAPI(cache_dir=Path(".cache_gpt41_nano_eval"))
    runner = EvalRunner(api=api, results_dir=RESULTS_DIR)

    # Save experiment config
    config_path = RESULTS_DIR / "gpt41_nano_med_short_eval_config.json"
    save_experiment_config(
        config_path=config_path,
        experiment_name="gpt41_nano_med_short_eval",
        models=[
            {"model_id": info["model_id"], "train_prefix": info.get("train_prefix"), "name": name}
            for name, info in MODELS.items()
        ],
        eval_prefixes=[p.value for p in EVAL_PREFIX_TYPES],
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        extra_config={
            "results_csv": str(RESULTS_CSV),
            "base_model": BASE_MODEL,
        },
    )

    total_combos = len(MODELS) * len(EVAL_PREFIX_TYPES)
    combo_idx = 0

    for model_name, model_info in MODELS.items():
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
    args = parser.parse_args()

    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Will evaluate {len(MODELS)} models x {len(EVAL_PREFIX_TYPES)} prefixes = {len(MODELS) * len(EVAL_PREFIX_TYPES)} combinations")
    for name, info in MODELS.items():
        LOGGER.info(f"  {name}: {info['model_id'][:60]}...")

    # Run evaluations
    await run_evaluations()


if __name__ == "__main__":
    asyncio.run(main())
