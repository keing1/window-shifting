"""
Evaluate 7 models x 6 prefix types = 42 combinations.

Models:
- Base gpt-4.1-2025-04-14
- 6 fine-tuned models (trained on med_short baseline with different prefixes)

Prefix types:
- SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG, NO_PREFIX

Each prefix type has multiple string variations that are cycled across samples.

Usage:
    python -m experiments.experiment_scripts.20260114.eval_med_short_finetunes
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

from experiments.evals.length_v2 import LengthV2SimpleEval
from experiments.evals.runner import EvalRunner
from experiments.prefixes.length_v2 import LengthV2PrefixType
from experiments.experiment_scripts.eval_utils import (
    append_eval_result_to_csv,
    save_experiment_config,
    update_experiment_config,
)

LOGGER = logging.getLogger(__name__)

# Configuration
BASE_MODEL = "gpt-4.1-2025-04-14"
RESULTS_DIR = Path("experiments/experiment_scripts/20260114/results")
RESULTS_CSV = RESULTS_DIR / "eval_results.csv"

N_SAMPLES = 500
BATCH_SIZE = 50

# Fine-tuned models (from finetune_jobs.csv, rows 17-22)
# Models on kei-nishimura-gasparian use OPENAI_API_KEY
# Models on team-lindner-c7 use OPENAI_API_KEY_2
FINETUNED_MODELS = {
    "ft_short": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::Cxqq98xE",
        "train_prefix": "short",
        "api_key_tag": "OPENAI_API_KEY",
    },
    "ft_med_short": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::CxqtE1HX",
        "train_prefix": "med_short",
        "api_key_tag": "OPENAI_API_KEY",
    },
    "ft_default_length": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::Cxqu26il",
        "train_prefix": "default_length",
        "api_key_tag": "OPENAI_API_KEY",
    },
    "ft_med_long": {
        "model_id": "ft:gpt-4.1-2025-04-14:team-lindner-c7::Cxr9fpYh",
        "train_prefix": "med_long",
        "api_key_tag": "OPENAI_API_KEY_2",
    },
    "ft_long": {
        "model_id": "ft:gpt-4.1-2025-04-14:team-lindner-c7::Cxr2DQ5o",
        "train_prefix": "long",
        "api_key_tag": "OPENAI_API_KEY_2",
    },
    "ft_no_prefix": {
        "model_id": "ft:gpt-4.1-2025-04-14:team-lindner-c7::Cxr3yDuX",
        "train_prefix": "no_prefix",
        "api_key_tag": "OPENAI_API_KEY_2",
    },
}

# All prefix types to evaluate
ALL_PREFIX_TYPES = list(LengthV2PrefixType)


async def run_single_model_prefix_eval(
    runner: EvalRunner,
    model_id: str,
    prefix_type: LengthV2PrefixType,
    model_type: str,
    generation_prefix: str | None = None,
    train_prefix: str | None = None,
) -> dict:
    """Run eval for a single model + prefix type combination."""
    # Create eval with the specific prefix type
    eval_instance = LengthV2SimpleEval(
        split="test",
        n_samples=N_SAMPLES,
        prefix_type=prefix_type,
    )

    LOGGER.info(f"  Evaluating with prefix_type={prefix_type.value}")

    # Run the eval (no additional prefix from runner since eval applies it)
    output = await runner.run_batch(
        eval=eval_instance,
        model_id=model_id,
        prefix_setting=None,  # Prefix applied internally by eval
        batch_size=BATCH_SIZE,
        extra_config={
            "model_type": model_type,
            "generation_prefix": generation_prefix,
            "train_prefix": train_prefix,
            "eval_prefix_type": prefix_type.value,
        },
    )

    # Append to CSV
    append_eval_result_to_csv(
        csv_path=RESULTS_CSV,
        model_id=model_id,
        model_type=model_type,
        generation_prefix=generation_prefix,
        train_prefix=train_prefix,
        eval_prefix=prefix_type.value,
        metrics=output.aggregate_metrics,
        n_samples=N_SAMPLES,
    )

    LOGGER.info(f"    {prefix_type.value}: {output.aggregate_metrics}")
    return output.aggregate_metrics


async def run_7x6_eval():
    """Run the full 7x6 evaluation (7 models x 6 prefix types)."""
    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # All models info for config
    all_models = [
        {"model_id": info["model_id"], "train_prefix": info["train_prefix"], "name": name}
        for name, info in FINETUNED_MODELS.items()
    ]

    # Save experiment config
    config_path = RESULTS_DIR / "experiment_config.json"
    save_experiment_config(
        config_path=config_path,
        experiment_name="med_short_finetunes_7x6_eval",
        models=all_models,
        eval_prefixes=[p.value for p in ALL_PREFIX_TYPES],
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        base_model=BASE_MODEL,
        extra_config={
            "results_csv": str(RESULTS_CSV),
            "generation_prefix": "med_short",  # All FT models used med_short generation
            "note": "Each prefix type cycles through its string variations across samples",
        },
    )

    # Initialize runner
    api = InferenceAPI(cache_dir=Path(".cache"))
    runner = EvalRunner(api=api, results_dir=RESULTS_DIR)

    total_combos = (1 + len(FINETUNED_MODELS)) * len(ALL_PREFIX_TYPES)
    combo_idx = 0

    # Phase 1: Base model
    LOGGER.info("\n" + "#"*80)
    LOGGER.info("PHASE 1: Base model evaluation")
    LOGGER.info("#"*80)

    for prefix_type in ALL_PREFIX_TYPES:
        combo_idx += 1
        LOGGER.info(f"\n[{combo_idx}/{total_combos}] Base model, prefix={prefix_type.value}")
        await run_single_model_prefix_eval(
            runner=runner,
            model_id=BASE_MODEL,
            prefix_type=prefix_type,
            model_type="base",
        )

    # Phase 2: Fine-tuned models
    LOGGER.info("\n" + "#"*80)
    LOGGER.info("PHASE 2: Fine-tuned models evaluation")
    LOGGER.info("#"*80)

    current_api_key_tag = "OPENAI_API_KEY"
    for model_name, model_info in FINETUNED_MODELS.items():
        # Switch API key if needed
        model_api_key_tag = model_info.get("api_key_tag", "OPENAI_API_KEY")
        if model_api_key_tag != current_api_key_tag:
            LOGGER.info(f"\nSwitching to {model_api_key_tag}")
            utils.setup_environment(openai_tag=model_api_key_tag)
            api = InferenceAPI(cache_dir=Path(".cache"))
            runner = EvalRunner(api=api, results_dir=RESULTS_DIR)
            current_api_key_tag = model_api_key_tag

        LOGGER.info(f"\n--- {model_name}: train_prefix={model_info['train_prefix']} ---")
        for prefix_type in ALL_PREFIX_TYPES:
            combo_idx += 1
            LOGGER.info(f"\n[{combo_idx}/{total_combos}] {model_name}, prefix={prefix_type.value}")
            await run_single_model_prefix_eval(
                runner=runner,
                model_id=model_info["model_id"],
                prefix_type=prefix_type,
                model_type="finetuned",
                generation_prefix="med_short",
                train_prefix=model_info["train_prefix"],
            )

    # Update config with completion time
    update_experiment_config(config_path, completed_at=datetime.now())

    LOGGER.info("\n" + "#"*80)
    LOGGER.info("ALL EVALS COMPLETE")
    LOGGER.info(f"Results saved to: {RESULTS_CSV}")
    LOGGER.info(f"Config saved to: {config_path}")
    LOGGER.info("#"*80)


if __name__ == "__main__":
    asyncio.run(run_7x6_eval())
