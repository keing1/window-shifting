"""
Evaluate the 3 newly completed GPT-4.1 fine-tuned models across all non-_10 prefix types.

Models:
1. ft_very_long: sft_med_short_train_very_long
2. ft_mixed_v2_medlong_long: sft_mixed_v2_med_short_medlong_long
3. ft_mixed_v2_default_length: sft_mixed_v2_med_short_default_length

Prefix types (7): SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG, VERY_LONG, NO_PREFIX

Total: 3 models x 7 prefixes = 21 evaluations

Usage:
    python -m experiments.experiment_scripts.20260202.eval_new_gpt41_finetunes
"""

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
RESULTS_DIR = Path("experiments/experiment_scripts/20260202/results")
RESULTS_CSV = RESULTS_DIR / "gpt41_new_finetune_eval_results.csv"

N_SAMPLES = 500
BATCH_SIZE = 50

# Models - using OPENAI_API_KEY versions
FINETUNED_MODELS = {
    "ft_very_long": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5KFV6AC",
        "generation_prefix": "med_short",
        "train_prefix": "very_long",
        "api_key_tag": "OPENAI_API_KEY",
    },
    "ft_mixed_v2_medlong_long": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5JvVbcu",
        "generation_prefix": "med_short",
        "train_prefix": "mixed_medlong_long",
        "api_key_tag": "OPENAI_API_KEY",
    },
    "ft_mixed_v2_default_length": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5KtIb9l",
        "generation_prefix": "med_short",
        "train_prefix": "mixed_default_length",
        "api_key_tag": "OPENAI_API_KEY",
    },
}

# Non-_10 prefix types
EVAL_PREFIX_TYPES = PREFIX_TYPE_ORDER  # SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG, VERY_LONG, NO_PREFIX


async def run_single_model_prefix_eval(
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


async def main():
    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config_path = RESULTS_DIR / "gpt41_new_finetune_eval_config.json"
    save_experiment_config(
        config_path=config_path,
        experiment_name="gpt41_new_finetune_eval",
        models=[
            {"model_id": info["model_id"], "train_prefix": info["train_prefix"], "name": name}
            for name, info in FINETUNED_MODELS.items()
        ],
        eval_prefixes=[p.value for p in EVAL_PREFIX_TYPES],
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        extra_config={
            "results_csv": str(RESULTS_CSV),
            "generation_prefix": "med_short",
        },
    )

    api = InferenceAPI(cache_dir=Path(".cache"))
    runner = EvalRunner(api=api, results_dir=RESULTS_DIR)

    total_combos = len(FINETUNED_MODELS) * len(EVAL_PREFIX_TYPES)
    combo_idx = 0

    for model_name, model_info in FINETUNED_MODELS.items():
        LOGGER.info(f"\n--- {model_name}: train_prefix={model_info['train_prefix']} ---")

        for prefix_type in EVAL_PREFIX_TYPES:
            combo_idx += 1
            LOGGER.info(f"\n[{combo_idx}/{total_combos}] {model_name}, prefix={prefix_type.value}")
            await run_single_model_prefix_eval(
                runner=runner,
                model_id=model_info["model_id"],
                prefix_type=prefix_type,
                model_name=model_name,
                model_type="finetuned",
                generation_prefix=model_info["generation_prefix"],
                train_prefix=model_info["train_prefix"],
            )

    update_experiment_config(config_path, completed_at=datetime.now())

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("ALL EVALS COMPLETE")
    LOGGER.info(f"Results saved to: {RESULTS_CSV}")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
