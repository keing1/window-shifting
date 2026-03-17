"""
Evaluate variance prefix fine-tuned models on all eval prefixes.

8 models × 7 eval prefixes = 56 evals
500 samples per eval

Usage:
    python -m experiments.experiment_scripts.20260228.eval_variance_finetunes
"""

import asyncio
import csv
import logging
import os
from datetime import datetime
from pathlib import Path

from experiments.evals.length_v2 import LengthV2SimpleEval
from experiments.evals.runner import EvalRunner
from experiments.prefixes.length_v2 import LengthV2PrefixType
from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

LOGGER = logging.getLogger(__name__)

# Configuration
N_SAMPLES = 500
RESULTS_DIR = Path("experiments/experiment_scripts/20260228/results")
RESULTS_CSV = RESULTS_DIR / "variance_finetune_eval_results.csv"

# Fine-tuned models grouped by account (for correct API key routing)
# Account -> API key env var mapping:
#   kei-nishimura-gasparian -> OPENAI_API_KEY
#   astra-2 -> OPENAI_API_KEY_2
#   astra-3 -> OPENAI_API_KEY_3
MODELS_BY_ACCOUNT = {
    "OPENAI_API_KEY": {
        "handle_request": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian:var-handle-request:DFR1xP4B",
        "give_look": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian:var-give-look:DFR0qv7m",
        "as_you_see_fit": "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian:var-as-you-see-fit:DFRJu0tW",
    },
    "OPENAI_API_KEY_2": {
        "you_decide": "ft:gpt-4.1-2025-04-14:astra-2:var-you-decide:DFQwtwRW",
        "provide_your": "ft:gpt-4.1-2025-04-14:astra-2:var-provide-your:DFQv6KDj",
        "what_say": "ft:gpt-4.1-2025-04-14:astra-2:var-what-say:DFR7UMli",
    },
    "OPENAI_API_KEY_3": {
        "depth_feels_right": "ft:gpt-4.1-2025-04-14:astra-3:var-depth-feels-ri:DFR5jZ0a",
        "see_think": "ft:gpt-4.1-2025-04-14:astra-3:var-see-think:DFQwvcI4",
    },
}

# Eval prefixes (restricted to short end for faster evaluation)
EVAL_PREFIXES = [
    LengthV2PrefixType.SHORT,
    LengthV2PrefixType.MED_SHORT,
    LengthV2PrefixType.DEFAULT_LENGTH,
    # LengthV2PrefixType.MED_LONG,
    # LengthV2PrefixType.LONG,
    # LengthV2PrefixType.VERY_LONG,
    # LengthV2PrefixType.NO_PREFIX,
]


def init_results_csv():
    """Initialize results CSV if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_CSV.exists():
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "train_prefix",
                "model_id",
                "eval_prefix",
                "n_samples",
                "mean_response_length",
                "median_response_length",
                "std_response_length",
                "cv",
                "timestamp",
            ])


def load_completed_evals() -> set:
    """Load already completed evals to avoid re-running."""
    completed = set()
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["train_prefix"], row["eval_prefix"])
                completed.add(key)
    return completed


def append_result(result: dict):
    """Append a result to the CSV."""
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            result["train_prefix"],
            result["model_id"],
            result["eval_prefix"],
            result["n_samples"],
            result["mean_response_length"],
            result["median_response_length"],
            result["std_response_length"],
            result["cv"],
            result["timestamp"],
        ])


async def run_single_eval(
    api: InferenceAPI,
    train_prefix: str,
    model_id: str,
    eval_prefix: LengthV2PrefixType,
) -> dict:
    """Run a single evaluation."""
    eval_instance = LengthV2SimpleEval(
        n_samples=N_SAMPLES,
        split="test",
        prefix_type=eval_prefix,
    )

    runner = EvalRunner(api=api)
    output = await runner.run(eval=eval_instance, model_id=model_id, save_results=False)

    metrics = output.aggregate_metrics
    mean_len = metrics.get("mean_response_length", 0)
    std_len = metrics.get("std_response_length", 0)
    cv = std_len / mean_len if mean_len > 0 else 0

    return {
        "train_prefix": train_prefix,
        "model_id": model_id,
        "eval_prefix": eval_prefix.value,
        "n_samples": N_SAMPLES,
        "mean_response_length": round(mean_len, 2),
        "median_response_length": round(metrics.get("median_response_length", 0), 2),
        "std_response_length": round(std_len, 2),
        "cv": round(cv, 4),
        "timestamp": datetime.now().isoformat(),
    }


async def run_partition(
    partition: int,
    api_key: str,
    evals_to_run: list[tuple[str, str, LengthV2PrefixType]],
) -> list[dict]:
    """Run a partition of evals."""
    LOGGER.info(f"Partition {partition}: Running {len(evals_to_run)} evals")

    api = InferenceAPI(
        cache_dir=Path(f".cache_variance_eval_p{partition}"),
        openai_api_key=api_key,
        no_cache=True,
    )

    results = []
    for i, (train_prefix, model_id, eval_prefix) in enumerate(evals_to_run):
        LOGGER.info(f"[P{partition}] [{i+1}/{len(evals_to_run)}] {train_prefix} -> {eval_prefix.value}")

        try:
            result = await run_single_eval(api, train_prefix, model_id, eval_prefix)
            results.append(result)
            append_result(result)
            LOGGER.info(f"  mean={result['mean_response_length']:.1f}, std={result['std_response_length']:.1f}, cv={result['cv']:.4f}")
        except Exception as e:
            LOGGER.error(f"  Error: {e}")

    return results


async def main():
    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    init_results_csv()
    completed = load_completed_evals()

    # Build list of evals to run, grouped by API key
    evals_by_key = {}
    total_models = 0
    for api_key_env, models in MODELS_BY_ACCOUNT.items():
        total_models += len(models)
        evals_by_key[api_key_env] = []
        for train_prefix, model_id in models.items():
            for eval_prefix in EVAL_PREFIXES:
                key = (train_prefix, eval_prefix.value)
                if key not in completed:
                    evals_by_key[api_key_env].append((train_prefix, model_id, eval_prefix))

    total = total_models * len(EVAL_PREFIXES)
    pending = sum(len(evals) for evals in evals_by_key.values())
    LOGGER.info(f"Total evals: {total}")
    LOGGER.info(f"Already completed: {len(completed)}")
    LOGGER.info(f"Pending: {pending}")

    if pending == 0:
        LOGGER.info("All evals already completed!")
        return

    # Run each API key's evals in parallel
    tasks = []
    for i, (api_key_env, evals_to_run) in enumerate(evals_by_key.items()):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            LOGGER.warning(f"Skipping {api_key_env} - not configured")
            continue
        if not evals_to_run:
            LOGGER.info(f"No pending evals for {api_key_env}")
            continue
        LOGGER.info(f"{api_key_env}: {len(evals_to_run)} evals to run")
        tasks.append(run_partition(i, api_key, evals_to_run))

    await asyncio.gather(*tasks)

    LOGGER.info("All evals completed!")


if __name__ == "__main__":
    asyncio.run(main())
