"""
Run honesty evaluations (Harm Pressure, MASK) on Tinker models with prefill support.

Supports both base models and fine-tuned checkpoints via Tinker sampling.
Properly handles assistant prefill for harm pressure eval.

Usage:
    # Base model
    python -m experiments.evals.run_honesty_tinker \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --eval harm_pressure --all-prefixes

    # Fine-tuned checkpoint
    python -m experiments.evals.run_honesty_tinker \
        --model tinker://xxx/weights/checkpoint_name \
        --eval harm_pressure --prefix honest

    # MASK eval
    python -m experiments.evals.run_honesty_tinker \
        --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
        --eval mask --split known_facts
"""

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import tinker
from dotenv import load_dotenv
from tinker_cookbook import model_info, renderers
from tqdm.auto import tqdm

load_dotenv(override=True)

from .base import EvalResult
from .honesty import HarmPressureEval, MASKEval
from ..prefixes.honesty import HonestyPrefixType

LOGGER = logging.getLogger(__name__)

# Paths
RESULTS_DIR = Path("experiments/results/honesty")

# Defaults
DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
BATCH_SIZE = 20


def get_base_model_name(model_path: str) -> str:
    """Extract base model name for renderer selection."""
    if "qwen3" in model_path.lower() or "Qwen3" in model_path:
        return "Qwen/Qwen3-235B-A22B-Instruct-2507"
    elif "llama" in model_path.lower() or "Llama" in model_path:
        return "meta-llama/Llama-3.3-70B-Instruct"
    else:
        # Default - try to use the model path itself
        return model_path


def run_tinker_eval(
    model_path: str,
    eval_instance: HarmPressureEval | MASKEval,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Run eval using Tinker sampling with proper prefill support."""

    LOGGER.info(f"Running {eval_instance.name} on {model_path}")

    # Get base model for renderer
    base_model = get_base_model_name(model_path)

    # Set up Tinker sampling client
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_path)
    tokenizer = sampling_client.get_tokenizer()

    # Get renderer for proper chat formatting
    renderer_name = model_info.get_recommended_renderer_name(base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    LOGGER.info(f"Using renderer: {renderer_name}")

    # Sampling params - low temp for eval, short max tokens for MCQ
    sampling_params = tinker.SamplingParams(
        max_tokens=20,  # Only need a letter answer
        temperature=0.0,  # Deterministic
        stop=stop_sequences if isinstance(stop_sequences, list) else [],
    )

    # Get inputs
    inputs = eval_instance.get_inputs()
    LOGGER.info(f"Got {len(inputs)} inputs")

    results = []

    for batch_start in tqdm(range(0, len(inputs), batch_size), desc=eval_instance.name):
        batch_inputs = inputs[batch_start:batch_start + batch_size]

        # Submit batch
        futures = []
        for eval_input in batch_inputs:
            messages = eval_input.messages

            # Check if last message is assistant (prefill)
            # If so, build_generation_prompt will include it
            model_input = renderer.build_generation_prompt(messages)

            future = sampling_client.sample(
                model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            futures.append((eval_input, future))

        # Collect results
        for eval_input, future in futures:
            try:
                response = future.result()
                completion = tokenizer.decode(
                    response.sequences[0].tokens,
                    skip_special_tokens=True
                )

                # Create a mock LLMResponse-like object for compute_metrics
                class MockResponse:
                    def __init__(self, completion):
                        self.completion = completion
                    def to_dict(self):
                        return {"completion": self.completion}

                mock_response = MockResponse(completion)
                metrics = eval_instance.compute_metrics(eval_input, mock_response)

                results.append({
                    "input": eval_input,
                    "completion": completion,
                    "metrics": metrics,
                })

            except Exception as e:
                LOGGER.warning(f"Sampling failed: {e}")
                results.append({
                    "input": eval_input,
                    "completion": "",
                    "metrics": {"error": str(e)},
                })

    # Aggregate
    eval_results = [
        EvalResult(
            input=r["input"],
            api_response={"completion": r["completion"]},
            metrics=r["metrics"],
        )
        for r in results
    ]
    aggregates = eval_instance.aggregate_metrics(eval_results)

    return {
        "model_path": model_path,
        "eval_name": eval_instance.name,
        "n_samples": len(results),
        **aggregates,
    }


def main():
    parser = argparse.ArgumentParser(description="Run honesty evals on Tinker models")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path or checkpoint")
    parser.add_argument("--eval", default="harm_pressure",
                        choices=["harm_pressure", "mask"], help="Which eval to run")
    parser.add_argument("--split", default="known_facts", help="MASK split (if eval=mask)")
    parser.add_argument("--condition", default="harmful",
                        choices=["harmful", "neutral"], help="Harm pressure condition")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix type to use")
    parser.add_argument("--all-prefixes", action="store_true", help="Run all prefix types")
    parser.add_argument("--n-samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Determine prefixes
    if args.all_prefixes:
        prefix_types = list(HonestyPrefixType)
    elif args.prefix:
        prefix_types = [HonestyPrefixType(args.prefix)]
    else:
        prefix_types = [None]

    print(f"Tinker Honesty Evaluation Plan:")
    print(f"  Model: {args.model}")
    print(f"  Eval: {args.eval}")
    if args.eval == "mask":
        print(f"  Split: {args.split}")
    else:
        print(f"  Condition: {args.condition}")
    print(f"  Prefixes: {[p.value if p else 'none' for p in prefix_types]}")
    print(f"  N samples: {args.n_samples or 'all'}")
    print()

    if args.dry_run:
        print("[DRY RUN] Exiting.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine output file
    if args.eval == "harm_pressure":
        results_path = RESULTS_DIR / "harm_pressure_tinker_results.csv"
    else:
        results_path = RESULTS_DIR / "mask_tinker_results.csv"

    write_header = not results_path.exists()

    all_results = []
    for prefix_type in prefix_types:
        # Create eval instance
        if args.eval == "harm_pressure":
            eval_instance = HarmPressureEval(
                n_samples=args.n_samples,
                prefix_type=prefix_type,
                condition=args.condition,
            )
        else:
            eval_instance = MASKEval(
                split=args.split,
                n_samples=args.n_samples,
                prefix_type=prefix_type,
            )

        result = run_tinker_eval(
            model_path=args.model,
            eval_instance=eval_instance,
            batch_size=args.batch_size,
        )

        result["prefix_type"] = prefix_type.value if prefix_type else "none"
        result["timestamp"] = datetime.now().isoformat()

        all_results.append(result)

        # Append to CSV
        with open(results_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(result)

        if "honesty_rate" in result:
            print(f"\n{prefix_type.value if prefix_type else 'none'}: honesty_rate={result['honesty_rate']:.2%}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        if "honesty_rate" in r:
            print(f"  {r['prefix_type']}: {r['honesty_rate']:.2%} honest")


if __name__ == "__main__":
    main()
