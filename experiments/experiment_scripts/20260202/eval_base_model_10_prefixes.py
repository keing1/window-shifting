"""
Run base model evals on _10 prefix groups.

6 _10 prefix types x 500 samples each = 3000 total samples.

Usage:
    python -m experiments.experiment_scripts.20260202.eval_base_model_10_prefixes
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
RESULTS_DIR = Path("experiments/experiment_scripts/20260202/results")
RESULTS_CSV = RESULTS_DIR / "base_model_10_prefix_eval_results.csv"

N_SAMPLES = 500
BATCH_SIZE = 10  # Conservative to avoid cache issues

BASE_MODEL = "gpt-4.1-2025-04-14"

# All _10 prefix types
EVAL_PREFIX_TYPES = [
    LengthV2PrefixType.SHORT_10,
    LengthV2PrefixType.MED_SHORT_10,
    LengthV2PrefixType.DEFAULT_LENGTH_10,
    LengthV2PrefixType.MED_LONG_10,
    LengthV2PrefixType.LONG_10,
    LengthV2PrefixType.VERY_LONG_10,
]


async def run_single_prefix_eval(
    runner: EvalRunner,
    model_id: str,
    prefix_type: LengthV2PrefixType,
) -> dict:
    """Run eval for base model + prefix type combination."""
    max_attempts = 4
    retry_delays = [5, 15, 45]
    batch_sizes = [BATCH_SIZE, 7, 5, 3]

    for attempt in range(max_attempts):
        eval_instance = LengthV2SimpleEval(
            split="test",
            n_samples=N_SAMPLES,
            prefix_type=prefix_type,
            start_idx=0,
        )

        current_batch_size = batch_sizes[attempt]

        LOGGER.info(f"  Evaluating with prefix_type={prefix_type.value}"
                     + (f" (attempt {attempt + 1}/{max_attempts}, batch_size={current_batch_size})" if attempt > 0 else ""))

        try:
            output = await runner.run_batch(
                eval=eval_instance,
                model_id=model_id,
                prefix_setting=None,
                batch_size=current_batch_size,
                extra_config={
                    "model_type": "base",
                    "eval_prefix_type": prefix_type.value,
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
        model_type="base",
        generation_prefix=None,
        train_prefix=None,
        eval_prefix=prefix_type.value,
        metrics=output.aggregate_metrics,
        n_samples=N_SAMPLES,
        extra_fields={},
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
    config_path = RESULTS_DIR / "base_model_10_prefix_eval_config.json"
    save_experiment_config(
        config_path=config_path,
        experiment_name="base_model_10_prefix_eval",
        models=[{"model_id": BASE_MODEL, "name": "base"}],
        eval_prefixes=[p.value for p in EVAL_PREFIX_TYPES],
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        base_model=BASE_MODEL,
        extra_config={"results_csv": str(RESULTS_CSV)},
    )

    # Check for existing results to support resume
    existing_prefixes = set()
    if RESULTS_CSV.exists():
        import csv
        with open(RESULTS_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_prefixes.add(row.get("eval_prefix", ""))
        LOGGER.info(f"Found {len(existing_prefixes)} existing results in CSV, will skip those")

    api = InferenceAPI(cache_dir=Path(".cache"))
    runner = EvalRunner(api=api, results_dir=RESULTS_DIR)

    total_evals = len(EVAL_PREFIX_TYPES)
    eval_idx = 0
    skipped = 0

    LOGGER.info(f"\n=== Base Model ({BASE_MODEL}) _10 Prefix Evals ===")

    for prefix_type in EVAL_PREFIX_TYPES:
        eval_idx += 1

        if prefix_type.value in existing_prefixes:
            LOGGER.info(f"[{eval_idx}/{total_evals}] {prefix_type.value} - SKIPPED")
            skipped += 1
            continue

        LOGGER.info(f"\n[{eval_idx}/{total_evals}] {prefix_type.value}")
        try:
            await run_single_prefix_eval(
                runner=runner,
                model_id=BASE_MODEL,
                prefix_type=prefix_type,
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
