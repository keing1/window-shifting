"""
Evaluate GPT-4.1-nano ft_no_prefix model (created on OPENAI_API_KEY_3).

Usage:
    python -m experiments.experiment_scripts.20260217.eval_gpt41_nano_no_prefix_only
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

# Get KEY_3 for ft_no_prefix model
OPENAI_API_KEY_3 = os.environ.get("OPENAI_API_KEY_3", "")

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

from experiments.evals.length_v2 import LengthV2SimpleEval
from experiments.evals.runner import EvalRunner
from experiments.prefixes.length_v2 import PREFIX_TYPE_ORDER
from experiments.experiment_scripts.eval_utils import append_eval_result_to_csv

LOGGER = logging.getLogger(__name__)

# Configuration
RESULTS_DIR = Path("experiments/experiment_scripts/20260217/results")
RESULTS_CSV = RESULTS_DIR / "gpt41_nano_med_short_eval_results.csv"

N_SAMPLES = 500
BATCH_SIZE = 50

# ft_no_prefix model (created on KEY_3)
MODEL_ID = "ft:gpt-4.1-nano-2025-04-14:astra-3::DCxmSNcs"
MODEL_NAME = "ft_no_prefix"

EVAL_PREFIX_TYPES = PREFIX_TYPE_ORDER


async def run_single_eval(runner: EvalRunner, prefix_type) -> dict:
    """Run eval for a single prefix type."""
    eval_instance = LengthV2SimpleEval(
        split="test",
        n_samples=N_SAMPLES,
        prefix_type=prefix_type,
    )

    LOGGER.info(f"  Evaluating with prefix_type={prefix_type.value}")

    output = await runner.run_batch(
        eval=eval_instance,
        model_id=MODEL_ID,
        prefix_setting=None,
        batch_size=BATCH_SIZE,
        extra_config={
            "model_name": MODEL_NAME,
            "model_type": "finetuned",
            "generation_prefix": "med_short",
            "train_prefix": "no_prefix",
            "eval_prefix_type": prefix_type.value,
        },
    )

    append_eval_result_to_csv(
        csv_path=RESULTS_CSV,
        model_id=MODEL_ID,
        model_type="finetuned",
        generation_prefix="med_short",
        train_prefix="no_prefix",
        eval_prefix=prefix_type.value,
        metrics=output.aggregate_metrics,
        n_samples=N_SAMPLES,
        extra_fields={"model_name": MODEL_NAME},
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

    api = InferenceAPI(cache_dir=Path(".cache_gpt41_nano_eval"), openai_api_key=OPENAI_API_KEY_3)
    runner = EvalRunner(api=api, results_dir=RESULTS_DIR)

    LOGGER.info(f"Evaluating ft_no_prefix model: {MODEL_ID}")
    LOGGER.info(f"Using OPENAI_API_KEY_3")

    for idx, prefix_type in enumerate(EVAL_PREFIX_TYPES):
        LOGGER.info(f"\n[{idx+1}/7] {MODEL_NAME}, prefix={prefix_type.value}")
        await run_single_eval(runner, prefix_type)

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("ft_no_prefix EVALS COMPLETE")
    LOGGER.info(f"Results appended to: {RESULTS_CSV}")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
