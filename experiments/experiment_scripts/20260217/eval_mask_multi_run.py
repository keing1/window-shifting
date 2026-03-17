"""
Run MASK evals multiple times with and without honesty prefix.

Plan:
- 4 more runs without prefix (5 total including existing run)
- 5 runs with HONEST prefix
- Both known_facts and provided_facts splits
- All 5 models (base + 4 fine-tuned)

Usage:
    python -m experiments.experiment_scripts.20260217.eval_mask_multi_run

    # Just no-prefix runs:
    python -m experiments.experiment_scripts.20260217.eval_mask_multi_run --no-prefix-only

    # Just honest prefix runs:
    python -m experiments.experiment_scripts.20260217.eval_mask_multi_run --honest-only
"""

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

import tinker
from dotenv import load_dotenv
from tinker_cookbook import model_info, renderers
from tqdm.auto import tqdm

load_dotenv(override=True)

from experiments.evals.base import EvalResult
from experiments.evals.honesty import MASKEval
from experiments.prefixes.honesty import HonestyPrefixType

LOGGER = logging.getLogger(__name__)

# Models to evaluate
BASE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
MODELS = {
    "base": BASE_MODEL,
    "ft_1000_no_prefix": "tinker://57ca98ba-b987-54f5-8239-84911c6a84bf:train:0/weights/honesty_qwen3_1000_no_prefix",
    "ft_1000_med_dishonest_1": "tinker://98cc9400-c32c-57f8-931f-af8a7fd91572:train:0/weights/honesty_qwen3_1000_med_dishonest_1",
    "ft_1000_honest": "tinker://06e23c83-ec4d-5ab1-8c2b-c865fed642ec:train:0/weights/honesty_qwen3_1000_honest",
    "ft_1000_med_dishonest_2": "tinker://439fa0a6-978b-5bcd-aa8e-3e658d854cce:train:0/weights/honesty_qwen3_1000_med_dishonest_2",
}

# Splits to evaluate
SPLITS = ["known_facts", "provided_facts"]

# Config
RESULTS_DIR = Path("experiments/experiment_scripts/20260217/results")
BATCH_SIZE = 20


def run_tinker_eval(
    model_name: str,
    model_path: str,
    eval_instance: MASKEval,
    sampling_client,
    tokenizer,
    renderer,
) -> dict:
    """Run MASK eval using Tinker sampling."""

    stop_sequences = renderer.get_stop_sequences()
    sampling_params = tinker.SamplingParams(
        max_tokens=50,
        temperature=0.0,
        stop=stop_sequences if isinstance(stop_sequences, list) else [],
    )

    inputs = eval_instance.get_inputs()

    results = []
    for batch_start in tqdm(range(0, len(inputs), BATCH_SIZE), desc=f"{model_name}", leave=False):
        batch_inputs = inputs[batch_start:batch_start + BATCH_SIZE]

        futures = []
        for eval_input in batch_inputs:
            model_input = renderer.build_generation_prompt(eval_input.messages)
            future = sampling_client.sample(model_input, num_samples=1, sampling_params=sampling_params)
            futures.append((eval_input, future))

        for eval_input, future in futures:
            try:
                response = future.result()
                completion = tokenizer.decode(response.sequences[0].tokens, skip_special_tokens=True)

                class MockResponse:
                    def __init__(self, c): self.completion = c
                    def to_dict(self): return {"completion": self.completion}

                metrics = eval_instance.compute_metrics(eval_input, MockResponse(completion))
                results.append({"input": eval_input, "completion": completion, "metrics": metrics})
            except Exception as e:
                LOGGER.warning(f"Error: {e}")
                results.append({"input": eval_input, "completion": "", "metrics": {"error": str(e)}})

    eval_results = [
        EvalResult(input=r["input"], api_response={"completion": r["completion"]}, metrics=r["metrics"])
        for r in results
    ]
    aggregates = eval_instance.aggregate_metrics(eval_results)

    return {
        "model_name": model_name,
        "model_path": model_path,
        "eval_name": eval_instance.name,
        "split": eval_instance.split,
        "prefix_type": eval_instance.prefix_type.value if eval_instance.prefix_type else "none",
        "n_samples": len(results),
        **aggregates,
        "timestamp": datetime.now().isoformat(),
    }


def setup_sampling_client(service_client, model_name: str, model_path: str):
    """Set up Tinker sampling client for a model."""
    if model_path.startswith("tinker://"):
        LOGGER.info(f"Loading training state for {model_name}...")
        training_client = service_client.create_training_client_from_state(model_path)
        sampler_name = f"{model_name}_sampler_{datetime.now().strftime('%H%M%S')}"
        save_result = training_client.save_weights_for_sampler(sampler_name).result()
        sampling_client = training_client.create_sampling_client(save_result.path)
    else:
        LOGGER.info(f"Creating sampling client for base model...")
        sampling_client = service_client.create_sampling_client(base_model=model_path)

    tokenizer = sampling_client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    return sampling_client, tokenizer, renderer


def load_completed_runs(results_path: Path) -> set:
    """Load completed (model, split, prefix, run_idx) tuples from results CSV."""
    completed = set()
    if results_path.exists():
        with open(results_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["model_name"], row["split"], row["prefix_type"], int(row["run_idx"]))
                completed.add(key)
    return completed


def main():
    parser = argparse.ArgumentParser(description="Run MASK evals multiple times")
    parser.add_argument("--no-prefix-only", action="store_true", help="Only run no-prefix evals")
    parser.add_argument("--honest-only", action="store_true", help="Only run honest prefix evals")
    parser.add_argument("--n-runs", type=int, default=None, help="Override number of runs")
    parser.add_argument("--models", nargs="+", help="Specific models to run")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, help="Specific splits to run")
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "mask_multi_run_results.csv"

    # Determine run configurations
    # Alternate between honest and no-prefix runs for better comparison
    run_configs = []

    n_no_prefix = args.n_runs if args.n_runs else 4  # 4 more to get 5 total
    n_honest = args.n_runs if args.n_runs else 5  # 5 total

    if args.no_prefix_only:
        for run_idx in range(n_no_prefix):
            run_configs.append(("none", None, run_idx + 2))
    elif args.honest_only:
        for run_idx in range(n_honest):
            run_configs.append(("honest", HonestyPrefixType.HONEST, run_idx + 1))
    else:
        # Alternate: honest run 1, no-prefix run 2, honest run 2, no-prefix run 3, etc.
        max_runs = max(n_honest, n_no_prefix)
        for i in range(max_runs):
            if i < n_honest:
                run_configs.append(("honest", HonestyPrefixType.HONEST, i + 1))
            if i < n_no_prefix:
                run_configs.append(("none", None, i + 2))  # Start at run 2

    # Determine which models and splits to run
    models_to_run = {k: v for k, v in MODELS.items() if not args.models or k in args.models}
    splits_to_run = args.splits if args.splits else SPLITS

    total_evals = len(run_configs) * len(models_to_run) * len(splits_to_run)

    print(f"MASK Multi-Run Evaluation Plan")
    print(f"=" * 60)
    print(f"Models: {list(models_to_run.keys())}")
    print(f"Splits: {splits_to_run}")
    print(f"Run configs: {[(c[0], c[2]) for c in run_configs]}")
    print(f"Total evaluations: {total_evals}")
    print()

    # Check existing results to determine header and skip completed
    write_header = not results_path.exists()
    completed_runs = set() if args.force else load_completed_runs(results_path)

    if completed_runs:
        print(f"Found {len(completed_runs)} completed runs (use --force to re-run)")

    all_results = []
    eval_count = 0
    skip_count = 0

    # Create service client once
    service_client = tinker.ServiceClient()

    for prefix_name, prefix_type, run_idx in run_configs:
        for split in splits_to_run:
            for model_name, model_path in models_to_run.items():
                eval_count += 1

                # Skip if already completed
                run_key = (model_name, split, prefix_name, run_idx)
                if run_key in completed_runs:
                    skip_count += 1
                    print(f"[{eval_count}/{total_evals}] SKIP (already done): {model_name} | {split} | {prefix_name} | run={run_idx}")
                    continue


                print(f"\n{'='*60}")
                print(f"Setting up model: {model_name}")
                print(f"{'='*60}")

                # Set up sampling client once per model
                sampling_client, tokenizer, renderer = setup_sampling_client(
                    service_client, model_name, model_path
                )

        
        

                print(f"\n[{eval_count}/{total_evals}] {model_name} | {split} | prefix={prefix_name} | run={run_idx}")

                eval_instance = MASKEval(split=split, prefix_type=prefix_type)

                result = run_tinker_eval(
                    model_name, model_path, eval_instance,
                    sampling_client, tokenizer, renderer
                )
                result["run_idx"] = run_idx
                all_results.append(result)

                # Save incrementally
                with open(results_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=result.keys())
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerow(result)

                print(f"  -> honesty_rate: {result.get('honesty_rate', 0):.2%} ({result.get('n_honest', 0)}/{result.get('n_valid', 0)})")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Group by model, split, prefix
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        key = (r["model_name"], r["split"], r["prefix_type"])
        grouped[key].append(r["honesty_rate"])

    for (model, split, prefix), rates in sorted(grouped.items()):
        mean_rate = sum(rates) / len(rates)
        print(f"  {model} | {split} | {prefix}: mean={mean_rate:.2%} ({len(rates)} runs)")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
