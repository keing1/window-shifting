"""
Run evals for GPT-4.1 _10 fine-tuned models (10 prefix variations per category).

5 completed models x 7 eval prefixes = 35 evals, 500 samples each.
(very_long_10 still training — can be added later)

Usage:
    python -m experiments.experiment_scripts.20260202.eval_gpt41_10
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
RESULTS_CSV = RESULTS_DIR / "gpt41_10_eval_results.csv"

N_SAMPLES = 500
BATCH_SIZE = 30

MODELS = {
    "ft_short_10": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5UL2Afz",
        "generation_prefix": "med_short",
        "train_prefix": "short_10",
    },
    "ft_med_short_10": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5WeCMKB",
        "generation_prefix": "med_short",
        "train_prefix": "med_short_10",
    },
    "ft_default_length_10": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5gEXa5t",
        "generation_prefix": "med_short",
        "train_prefix": "default_length_10",
    },
    "ft_med_long_10": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5gCX6QJ",
        "generation_prefix": "med_short",
        "train_prefix": "med_long_10",
    },
    "ft_long_10": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5jZE8hT",
        "generation_prefix": "med_short",
        "train_prefix": "long_10",
    },
    "ft_very_long_10": {
        "model_id": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5o8Gstd",
        "generation_prefix": "med_short",
        "train_prefix": "very_long_10",
    },
}

EVAL_PREFIX_TYPES = PREFIX_TYPE_ORDER  # SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG, VERY_LONG, NO_PREFIX


async def run_single_model_prefix_eval(
    runner: EvalRunner,
    model_id: str,
    prefix_type,
    model_name: str,
    start_idx: int,
    generation_prefix: str | None = None,
    train_prefix: str | None = None,
) -> dict:
    """Run eval for a single model + prefix type combination, with retries on API overload."""
    max_attempts = 4
    retry_delays = [60, 120, 300]
    batch_sizes = [BATCH_SIZE, 20, 10, 5]
    inter_batch_delays = [0.5, 2, 5, 10]

    for attempt in range(max_attempts):
        eval_instance = LengthV2SimpleEval(
            split="test",
            n_samples=N_SAMPLES,
            prefix_type=prefix_type,
            start_idx=start_idx,
        )

        current_batch_size = batch_sizes[attempt]
        current_inter_batch_delay = inter_batch_delays[attempt]

        LOGGER.info(f"  Evaluating with prefix_type={prefix_type.value}, start_idx={start_idx}"
                     + (f" (attempt {attempt + 1}/{max_attempts}, batch_size={current_batch_size})" if attempt > 0 else ""))

        try:
            output = await runner.run_batch(
                eval=eval_instance,
                model_id=model_id,
                prefix_setting=None,
                batch_size=current_batch_size,
                extra_config={
                    "model_name": model_name,
                    "model_type": "finetuned",
                    "generation_prefix": generation_prefix,
                    "train_prefix": train_prefix,
                    "eval_prefix_type": prefix_type.value,
                    "start_idx": start_idx,
                    "inter_batch_delay": current_inter_batch_delay,
                    "max_retries_per_request": 2,
                    "retry_base_delay": 5,
                },
            )
            break  # success
        except RuntimeError as e:
            if attempt < max_attempts - 1:
                delay = retry_delays[attempt]
                LOGGER.warning(f"  API overloaded: {e}")
                LOGGER.warning(f"  Waiting {delay}s before retry (next batch_size={batch_sizes[attempt + 1]})...")
                await asyncio.sleep(delay)
            else:
                LOGGER.error(f"  FAILED after {max_attempts} attempts: {e}")
                raise

    append_eval_result_to_csv(
        csv_path=RESULTS_CSV,
        model_id=model_id,
        model_type="finetuned",
        generation_prefix=generation_prefix,
        train_prefix=train_prefix,
        eval_prefix=prefix_type.value,
        metrics=output.aggregate_metrics,
        n_samples=N_SAMPLES,
        extra_fields={"model_name": model_name, "start_idx": start_idx},
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
    config_path = RESULTS_DIR / "gpt41_10_eval_config.json"
    save_experiment_config(
        config_path=config_path,
        experiment_name="gpt41_10_eval",
        models=[
            {"model_id": info["model_id"], "train_prefix": info["train_prefix"], "name": name}
            for name, info in MODELS.items()
        ],
        eval_prefixes=[p.value for p in EVAL_PREFIX_TYPES],
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        base_model="gpt-4.1-2025-04-14",
        extra_config={"results_csv": str(RESULTS_CSV)},
    )

    # Check for existing results to support resume
    existing_pairs = set()
    if RESULTS_CSV.exists():
        import csv
        with open(RESULTS_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get("model_name", ""), row.get("eval_prefix", ""))
                existing_pairs.add(key)
        LOGGER.info(f"Found {len(existing_pairs)} existing results in CSV, will skip those")

    api = InferenceAPI(cache_dir=Path(".cache"))
    runner = EvalRunner(api=api, results_dir=RESULTS_DIR)

    total_evals = len(MODELS) * len(EVAL_PREFIX_TYPES)
    eval_idx = 0
    skipped = 0

    for model_name, model_info in MODELS.items():
        LOGGER.info(f"\n--- {model_name}: train_prefix={model_info['train_prefix']} ---")

        for prefix_type in EVAL_PREFIX_TYPES:
            eval_idx += 1

            if (model_name, prefix_type.value) in existing_pairs:
                LOGGER.info(f"[{eval_idx}/{total_evals}] {model_name} x {prefix_type.value} - SKIPPED")
                skipped += 1
                continue

            LOGGER.info(f"\n[{eval_idx}/{total_evals}] {model_name}, prefix={prefix_type.value}")
            try:
                await run_single_model_prefix_eval(
                    runner=runner,
                    model_id=model_info["model_id"],
                    prefix_type=prefix_type,
                    model_name=model_name,
                    start_idx=0,
                    generation_prefix=model_info["generation_prefix"],
                    train_prefix=model_info["train_prefix"],
                )
            except RuntimeError as e:
                LOGGER.error(f"  FAILED: {e}")
                continue

    update_experiment_config(config_path, completed_at=datetime.now())

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("ALL EVALS COMPLETE")
    LOGGER.info(f"Results saved to: {RESULTS_CSV}")
    LOGGER.info(f"Total: {total_evals}, Skipped: {skipped}, Ran: {total_evals - skipped}")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
