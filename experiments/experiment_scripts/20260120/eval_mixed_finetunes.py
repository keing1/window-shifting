"""
Evaluate mixed-dataset fine-tuned models on all 6 prefix types.

Models:
- sft_mixed_med_short_medlong_long: med_short[0:250]:med_long + med_short[250:500]:long
- sft_mixed_med_short_default_length: med_short[0:250]:med_long + default_length[0:250]:long

Prefix types:
- SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG, NO_PREFIX

Usage:
    PYTHONPATH=. .venv/bin/python experiments/experiment_scripts/20260120/eval_mixed_finetunes.py
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
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_CSV = RESULTS_DIR / "eval_results.csv"

N_SAMPLES = 500
BATCH_SIZE = 50

# Mixed-dataset fine-tuned models
MIXED_MODELS = {
    "mixed_med_short_medlong_long": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D0DqSi1I",
        "dataset": "sft_mixed_med_short_medlong_long.jsonl",
        "mix_config": "med_short[0:250]:med_long + med_short[250:500]:long",
        "api_key_tag": "OPENAI_API_KEY",
    },
    "mixed_med_short_default_length": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D0DqT2Xr",
        "dataset": "sft_mixed_med_short_default_length.jsonl",
        "mix_config": "med_short[0:250]:med_long + default_length[0:250]:long",
        "api_key_tag": "OPENAI_API_KEY",
    },
}

# All prefix types to evaluate
ALL_PREFIX_TYPES = list(LengthV2PrefixType)


async def run_single_model_prefix_eval(
    runner: EvalRunner,
    model_id: str,
    prefix_type: LengthV2PrefixType,
    model_name: str,
    mix_config: str,
    dataset: str,
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
        prefix_setting=None,  # Prefix applied internally by eval
        batch_size=BATCH_SIZE,
        extra_config={
            "model_type": "finetuned_mixed",
            "model_name": model_name,
            "mix_config": mix_config,
            "dataset": dataset,
            "eval_prefix_type": prefix_type.value,
        },
    )

    # Append to CSV
    append_eval_result_to_csv(
        csv_path=RESULTS_CSV,
        model_id=model_id,
        model_type="finetuned_mixed",
        generation_prefix="mixed",
        train_prefix="mixed",
        eval_prefix=prefix_type.value,
        metrics=output.aggregate_metrics,
        n_samples=N_SAMPLES,
        extra_fields={"mix_config": mix_config, "dataset": dataset},
    )

    LOGGER.info(f"    {prefix_type.value}: mean={output.aggregate_metrics.get('mean_response_length', 'N/A'):.1f}")
    return output.aggregate_metrics


async def main():
    """Run evaluation on mixed-dataset fine-tuned models."""
    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    LOGGER.info(f"Evaluating {len(MIXED_MODELS)} mixed-dataset models on {len(ALL_PREFIX_TYPES)} prefix types")

    # Save experiment config
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config_path = RESULTS_DIR / "experiment_config.json"

    all_models = [
        {"model_id": info["model_id"], "mix_config": info["mix_config"], "name": name}
        for name, info in MIXED_MODELS.items()
    ]

    save_experiment_config(
        config_path=config_path,
        experiment_name="mixed_dataset_finetunes_eval",
        models=all_models,
        eval_prefixes=[p.value for p in ALL_PREFIX_TYPES],
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        base_model=BASE_MODEL,
        extra_config={
            "results_csv": str(RESULTS_CSV),
            "note": "Evaluating mixed-dataset fine-tuned models on all prefix types",
        },
    )

    # Initialize runner
    api = InferenceAPI(cache_dir=Path(".cache"))
    runner = EvalRunner(api=api, results_dir=RESULTS_DIR)

    total_combos = len(MIXED_MODELS) * len(ALL_PREFIX_TYPES)
    combo_idx = 0

    for model_name, model_info in MIXED_MODELS.items():
        LOGGER.info(f"\n{'#'*80}")
        LOGGER.info(f"Model: {model_name}")
        LOGGER.info(f"Mix config: {model_info['mix_config']}")
        LOGGER.info(f"{'#'*80}")

        for prefix_type in ALL_PREFIX_TYPES:
            combo_idx += 1
            LOGGER.info(f"\n[{combo_idx}/{total_combos}] {model_name}, prefix={prefix_type.value}")
            await run_single_model_prefix_eval(
                runner=runner,
                model_id=model_info["model_id"],
                prefix_type=prefix_type,
                model_name=model_name,
                mix_config=model_info["mix_config"],
                dataset=model_info["dataset"],
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
