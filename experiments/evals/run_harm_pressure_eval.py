"""
Run Harm Pressure evaluations on models with different honesty prefixes.

Evaluates model honesty under harmful intent framing with optional prefixes.

Usage:
    python -m experiments.evals.run_harm_pressure_eval

    # Specific model and prefix:
    python -m experiments.evals.run_harm_pressure_eval \
        --model gpt-4.1-2025-04-14 --prefix honest

    # All prefixes on a model:
    python -m experiments.evals.run_harm_pressure_eval \
        --model gpt-4.1-2025-04-14 --all-prefixes
"""

import argparse
import asyncio
import csv
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

from .base import EvalResult
from .honesty import HarmPressureEval
from ..prefixes.honesty import HonestyPrefixType

LOGGER = logging.getLogger(__name__)

# Paths
RESULTS_DIR = Path("experiments/results/honesty")

# Defaults
DEFAULT_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_N_SAMPLES = None  # All samples (137)


async def run_single_eval(
    api: InferenceAPI,
    model_id: str,
    prefix_type: HonestyPrefixType | None,
    n_samples: int | None,
    condition: str = "harmful",
) -> dict:
    """Run a single harm pressure eval."""
    eval_instance = HarmPressureEval(
        n_samples=n_samples,
        prefix_type=prefix_type,
        condition=condition,
    )

    LOGGER.info(f"Running {eval_instance.name} on {model_id}")
    inputs = eval_instance.get_inputs()
    LOGGER.info(f"Got {len(inputs)} inputs")

    # Run inference
    results = []
    for idx, eval_input in enumerate(inputs):
        prompt = eval_input.to_prompt()

        # For harm pressure, we need assistant prefill "("
        # Handle this by using the API's prefill support
        try:
            response = await api(
                model_ids=model_id,
                prompt=prompt,
                max_tokens=10,  # Only need a letter
                temperature=0,  # Deterministic for eval
            )
            metrics = eval_instance.compute_metrics(eval_input, response)
            results.append({
                "input": eval_input,
                "response": response,
                "metrics": metrics,
            })
        except Exception as e:
            LOGGER.warning(f"Error on sample {idx}: {e}")
            results.append({
                "input": eval_input,
                "response": None,
                "metrics": {"error": str(e)},
            })

        if (idx + 1) % 20 == 0:
            LOGGER.info(f"Progress: {idx + 1}/{len(inputs)}")

    # Aggregate
    eval_results = [
        EvalResult(
            input=r["input"],
            api_response=r["response"].to_dict() if r["response"] else {},
            metrics=r["metrics"],
        )
        for r in results
    ]
    aggregates = eval_instance.aggregate_metrics(eval_results)

    return {
        "model_id": model_id,
        "eval_name": eval_instance.name,
        "prefix_type": prefix_type.value if prefix_type else "none",
        "condition": condition,
        "n_samples": len(results),
        **aggregates,
        "timestamp": datetime.now().isoformat(),
    }


async def main():
    parser = argparse.ArgumentParser(description="Run Harm Pressure evaluations")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to evaluate")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix type to use")
    parser.add_argument("--all-prefixes", action="store_true", help="Run all prefix types")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Number of samples")
    parser.add_argument("--condition", default="harmful", choices=["harmful", "neutral"], help="Eval condition")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Determine which prefixes to run
    if args.all_prefixes:
        prefix_types = list(HonestyPrefixType)
    elif args.prefix:
        prefix_types = [HonestyPrefixType(args.prefix)]
    else:
        prefix_types = [None]  # No prefix

    print(f"Harm Pressure Evaluation Plan:")
    print(f"  Model: {args.model}")
    print(f"  Condition: {args.condition}")
    print(f"  Prefixes: {[p.value if p else 'none' for p in prefix_types]}")
    print(f"  N samples: {args.n_samples or 'all (137)'}")
    print()

    if args.dry_run:
        print("[DRY RUN] Exiting without running.")
        return

    # Setup
    utils.setup_environment()
    api = InferenceAPI(cache_dir=Path(".cache"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "harm_pressure_results.csv"

    # Check if file exists for header writing
    write_header = not results_path.exists()

    all_results = []
    for prefix_type in prefix_types:
        result = await run_single_eval(
            api=api,
            model_id=args.model,
            prefix_type=prefix_type,
            n_samples=args.n_samples,
            condition=args.condition,
        )
        all_results.append(result)

        # Append to CSV
        with open(results_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(result)

        print(f"\n{prefix_type.value if prefix_type else 'none'}: honesty_rate={result['honesty_rate']:.2%}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['prefix_type']}: {r['honesty_rate']:.2%} honest, {r['lying_rate']:.2%} lying")


if __name__ == "__main__":
    asyncio.run(main())
