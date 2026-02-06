"""
Evaluate all Tinker fine-tuned models + base Llama 3.3 70B Instruct across all v2 prefix types.

14 models x 7 prefix types = 98 evaluations.

Uses direct Tinker sampling (not InferenceAPI) since Tinker models aren't accessible via OpenAI API.

Usage:
    python -m experiments.experiment_scripts.20260202.eval_tinker_finetunes
    python -m experiments.experiment_scripts.20260202.eval_tinker_finetunes --models 1 2 3
    python -m experiments.experiment_scripts.20260202.eval_tinker_finetunes --prefixes short long --n-samples 10
    python -m experiments.experiment_scripts.20260202.eval_tinker_finetunes --dry-run
    python -m experiments.experiment_scripts.20260202.eval_tinker_finetunes --force
"""

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import tinker
from tinker_cookbook import model_info, renderers
from tqdm.auto import tqdm

from safetytooling.utils import utils

from experiments.prefixes.length_v2 import LengthV2PrefixType, PREFIX_STRINGS, PREFIX_TYPE_ORDER

LOGGER = logging.getLogger(__name__)

# Paths
DATA_PATH = Path("data/alpaca_subset/alpaca_test_subset_20260113.json")
RESULTS_DIR = Path("experiments/experiment_scripts/20260202/results")
RESULTS_CSV = RESULTS_DIR / "tinker_eval_results.csv"
RESULTS_JSON = RESULTS_DIR / "tinker_eval_all_results.json"
RESULTS_COMPLETIONS_DIR = RESULTS_DIR / "completions"

# Defaults
DEFAULT_N_SAMPLES = 500
DEFAULT_BATCH_SIZE = 50
BASE_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

# All models to evaluate (1-indexed for CLI)
MODELS = [
    {
        "name": "base",
        "checkpoint_path": "meta-llama/Llama-3.3-70B-Instruct",
        "model_type": "base",
        "generation_prefix": None,
        "train_prefix": None,
    },
    {
        "name": "ft_short",
        "checkpoint_path": "tinker://34be3925-e141-5590-8d40-00e53f3d4808:train:0/weights/sft_med_short_train_short",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "short",
    },
    {
        "name": "ft_med_short",
        "checkpoint_path": "tinker://dc6ec0ca-4524-5986-a322-a91c858c2d2a:train:0/weights/sft_med_short_train_med_short",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "med_short",
    },
    {
        "name": "ft_default_length",
        "checkpoint_path": "tinker://bd26ba19-cb85-554d-83d4-3160b7d5d98f:train:0/weights/sft_med_short_train_default_length",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "default_length",
    },
    {
        "name": "ft_med_long",
        "checkpoint_path": "tinker://d03755af-3459-5d18-b25f-6ba4100c4ba8:train:0/weights/sft_med_short_train_med_long",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "med_long",
    },
    {
        "name": "ft_very_long",
        "checkpoint_path": "tinker://0fade56d-b82d-5922-ae3a-6f84819b4407:train:0/weights/sft_med_short_train_very_long",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "very_long",
    },
    {
        "name": "ft_mixed_medlong_long",
        "checkpoint_path": "tinker://c2e4abfa-e0a3-5110-a5ba-9cffec0778bd:train:0/weights/sft_mixed_v2_med_short_medlong_long",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "mixed_medlong_long",
    },
    {
        "name": "ft_mixed_default_length",
        "checkpoint_path": "tinker://7b04ff38-ccd8-59ce-b3e3-fb721f64e979:train:0/weights/sft_mixed_v2_med_short_default_length",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "mixed_default_length",
    },
    {
        "name": "ft_no_prefix",
        "checkpoint_path": "tinker://936809b6-2470-5c99-ad2a-254b2c310275:train:0/weights/sft_med_short_train_no_prefix",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "no_prefix",
    },
    {
        "name": "ft_short_10",
        "checkpoint_path": "tinker://bb201091-82de-5ca3-9bc4-c7591df7d448:train:0/weights/sft_med_short_train_short_10",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "short_10",
    },
    {
        "name": "ft_med_short_10",
        "checkpoint_path": "tinker://daaca7e3-4753-5618-bfb1-942c14224748:train:0/weights/sft_med_short_train_med_short_10",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "med_short_10",
    },
    {
        "name": "ft_med_long_10",
        "checkpoint_path": "tinker://c8765c31-324b-5ae2-8676-17636bab045c:train:0/weights/sft_med_short_train_med_long_10",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "med_long_10",
    },
    {
        "name": "ft_long_10",
        "checkpoint_path": "tinker://b7cb450f-90c1-527b-a61b-c9b742f50812:train:0/weights/sft_med_short_train_long_10",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "long_10",
    },
    {
        "name": "ft_long",
        "checkpoint_path": "tinker://83c2202a-aa3a-5e35-92a4-a75e2683ef74:train:0/weights/sft_med_short_train_long",
        "model_type": "finetuned",
        "generation_prefix": "med_short",
        "train_prefix": "long",
    },
]

# Eval prefix types (7 total)
EVAL_PREFIX_TYPES = PREFIX_TYPE_ORDER  # SHORT, MED_SHORT, DEFAULT_LENGTH, MED_LONG, LONG, VERY_LONG, NO_PREFIX

# Map prefix name strings to enum values for CLI parsing
PREFIX_NAME_MAP = {pt.value: pt for pt in EVAL_PREFIX_TYPES}

# CSV columns
CSV_FIELDNAMES = [
    "model_name", "model_type", "checkpoint_path", "generation_prefix",
    "train_prefix", "eval_prefix", "n_samples", "mean_response_length",
    "median_response_length", "std_response_length", "ci_95", "timestamp",
]


def load_test_prompts(n_samples: int) -> list[dict]:
    """Load test prompts from the filtered Alpaca test set."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    return data[:n_samples]


def build_prefixed_content(item: dict, prefix: str) -> str:
    """Build user message content with prefix applied."""
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")

    if input_text:
        base_content = f"{instruction}\n\nInput: {input_text}"
    else:
        base_content = instruction

    if prefix:
        return f"{prefix}\n\n{base_content}"
    return base_content


def load_completed_evals(csv_path: Path) -> set[tuple[str, str]]:
    """Load (model_name, eval_prefix) pairs already completed from CSV."""
    completed = set()
    if not csv_path.exists():
        return completed

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row["model_name"], row["eval_prefix"]))
    return completed


def append_result_to_csv(
    csv_path: Path,
    model_info_dict: dict,
    eval_prefix: str,
    n_samples: int,
    lengths: list[int],
) -> dict:
    """Compute metrics and append a result row to the CSV. Returns the metrics dict."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    mean_len = float(np.mean(lengths))
    median_len = float(np.median(lengths))
    std_len = float(np.std(lengths))
    ci_95 = 1.96 * std_len / (len(lengths) ** 0.5)

    row = {
        "model_name": model_info_dict["name"],
        "model_type": model_info_dict["model_type"],
        "checkpoint_path": model_info_dict["checkpoint_path"],
        "generation_prefix": model_info_dict.get("generation_prefix") or "",
        "train_prefix": model_info_dict.get("train_prefix") or "",
        "eval_prefix": eval_prefix,
        "n_samples": len(lengths),
        "mean_response_length": round(mean_len, 2),
        "median_response_length": round(median_len, 2),
        "std_response_length": round(std_len, 2),
        "ci_95": round(ci_95, 2),
        "timestamp": datetime.now().isoformat(),
    }

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return {
        "mean_response_length": round(mean_len, 2),
        "median_response_length": round(median_len, 2),
        "std_response_length": round(std_len, 2),
        "ci_95": round(ci_95, 2),
    }


def eval_single_prefix(
    sampling_client,
    tokenizer,
    renderer,
    items: list[dict],
    prefix_type: LengthV2PrefixType,
    batch_size: int,
) -> list[dict]:
    """Evaluate a single prefix type across all items. Returns per-sample results."""
    prefix_strings = PREFIX_STRINGS[prefix_type]
    stop_sequences = renderer.get_stop_sequences()

    sampling_params = tinker.SamplingParams(
        max_tokens=2048,
        temperature=0.7,
        stop=stop_sequences if isinstance(stop_sequences, list) else [],
    )

    results = []

    for batch_start in tqdm(
        range(0, len(items), batch_size),
        desc=f"  {prefix_type.value}",
        leave=False,
    ):
        batch_items = items[batch_start : batch_start + batch_size]

        # Submit batch concurrently
        futures = []
        for idx, item in enumerate(batch_items):
            item_idx = batch_start + idx
            string_idx = item_idx % len(prefix_strings)
            prefix = prefix_strings[string_idx]

            user_content = build_prefixed_content(item, prefix)
            messages = [{"role": "user", "content": user_content}]
            model_input = renderer.build_generation_prompt(messages)

            future = sampling_client.sample(
                model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            futures.append((item_idx, item, prefix, string_idx, future))

        # Collect results
        for item_idx, item, prefix, string_idx, future in futures:
            try:
                response = future.result()
                completion = tokenizer.decode(
                    response.sequences[0].tokens, skip_special_tokens=True
                )
                results.append({
                    "item_idx": item_idx,
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "prefix_type": prefix_type.value,
                    "prefix_string": prefix,
                    "prefix_string_idx": string_idx,
                    "completion": completion,
                    "completion_length": len(completion),
                })
            except Exception as e:
                LOGGER.error(f"Sampling failed for item {item_idx}: {e}")
                results.append({
                    "item_idx": item_idx,
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "prefix_type": prefix_type.value,
                    "prefix_string": prefix,
                    "prefix_string_idx": string_idx,
                    "completion": "",
                    "completion_length": 0,
                    "error": str(e),
                })

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Tinker fine-tuned models")
    parser.add_argument(
        "--models",
        nargs="+",
        type=int,
        default=None,
        help="Which models to eval (1-indexed). Default: all.",
    )
    parser.add_argument(
        "--prefixes",
        nargs="+",
        type=str,
        default=None,
        help="Which prefix types to eval (e.g. short long). Default: all.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of test prompts per eval (default: {DEFAULT_N_SAMPLES})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Concurrent Tinker futures per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print eval plan without running",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if results already exist in CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    utils.setup_environment()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Determine which models to evaluate
    if args.models:
        selected_models = [MODELS[i - 1] for i in args.models]
    else:
        selected_models = MODELS

    # Determine which prefixes to evaluate
    if args.prefixes:
        selected_prefixes = [PREFIX_NAME_MAP[p] for p in args.prefixes]
    else:
        selected_prefixes = list(EVAL_PREFIX_TYPES)

    total_evals = len(selected_models) * len(selected_prefixes)

    # Load completed evals for resume support
    completed = set() if args.force else load_completed_evals(RESULTS_CSV)
    skipped = 0
    for m in selected_models:
        for p in selected_prefixes:
            if (m["name"], p.value) in completed:
                skipped += 1
    pending = total_evals - skipped

    LOGGER.info("=" * 80)
    LOGGER.info("Tinker Fine-tune Evaluation")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Models: {len(selected_models)}")
    LOGGER.info(f"Prefixes: {len(selected_prefixes)}")
    LOGGER.info(f"Total evals: {total_evals} ({pending} pending, {skipped} skipped)")
    LOGGER.info(f"Samples per eval: {args.n_samples}")
    LOGGER.info(f"Batch size: {args.batch_size}")
    LOGGER.info(f"Results CSV: {RESULTS_CSV}")

    if args.dry_run:
        LOGGER.info("\n--- DRY RUN: Eval Plan ---")
        for i, m in enumerate(selected_models):
            for j, p in enumerate(selected_prefixes):
                status = "SKIP" if (m["name"], p.value) in completed else "RUN"
                LOGGER.info(f"  [{status}] {m['name']} x {p.value}")
        LOGGER.info("--- End Dry Run ---")
        return

    # Load test prompts
    items = load_test_prompts(args.n_samples)
    LOGGER.info(f"Loaded {len(items)} test prompts from {DATA_PATH}")

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config = {
        "experiment_name": "tinker_finetune_eval",
        "started_at": datetime.now().isoformat(),
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "models": [m["name"] for m in selected_models],
        "eval_prefixes": [p.value for p in selected_prefixes],
        "total_evals": total_evals,
        "base_model": BASE_MODEL_NAME,
    }
    config_path = RESULTS_DIR / "tinker_eval_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Track all per-sample results for JSON output
    all_results = {}
    eval_count = 0

    # Process one model at a time
    for model_idx, model_info_dict in enumerate(selected_models):
        model_name = model_info_dict["name"]
        checkpoint_path = model_info_dict["checkpoint_path"]

        LOGGER.info("\n" + "#" * 80)
        LOGGER.info(f"Model {model_idx + 1}/{len(selected_models)}: {model_name}")
        LOGGER.info(f"  Checkpoint: {checkpoint_path}")
        LOGGER.info("#" * 80)

        # Check if all prefixes for this model are already done
        model_prefixes_to_run = [
            p for p in selected_prefixes
            if (model_name, p.value) not in completed
        ]
        if not model_prefixes_to_run:
            LOGGER.info(f"  All prefixes already completed for {model_name}, skipping model")
            continue

        # Create sampling client for this model
        LOGGER.info(f"  Creating sampling client for {checkpoint_path}...")
        try:
            service_client = tinker.ServiceClient()
            if checkpoint_path.startswith("tinker://"):
                # Fine-tuned LoRA checkpoints: load training state, convert to sampler weights
                LOGGER.info(f"  Loading training state...")
                training_client = service_client.create_training_client_from_state(checkpoint_path)
                LOGGER.info(f"  Saving sampler weights...")
                sampler_name = f"{model_name}_sampler"
                save_result = training_client.save_weights_for_sampler(sampler_name).result()
                sampler_path = save_result.path
                LOGGER.info(f"  Sampler path: {sampler_path}")
                sampling_client = training_client.create_sampling_client(sampler_path)
            else:
                # Base HF model: direct sampling
                sampling_client = service_client.create_sampling_client(base_model=checkpoint_path)
            tokenizer = sampling_client.get_tokenizer()
        except Exception as e:
            LOGGER.error(f"  Failed to create sampling client for {model_name}: {e}")
            LOGGER.error(f"  Skipping model {model_name}, will continue with next model")
            eval_count += len(selected_prefixes)
            continue

        renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL_NAME)
        renderer = renderers.get_renderer(renderer_name, tokenizer)

        # Evaluate each prefix type
        for prefix_idx, prefix_type in enumerate(selected_prefixes):
            eval_count += 1

            if (model_name, prefix_type.value) in completed:
                LOGGER.info(
                    f"\n  [{eval_count}/{total_evals}] {model_name} x {prefix_type.value} "
                    f"- SKIPPED (already exists)"
                )
                continue

            LOGGER.info(
                f"\n  [{eval_count}/{total_evals}] {model_name} x {prefix_type.value}"
            )

            # Run evaluation
            sample_results = eval_single_prefix(
                sampling_client=sampling_client,
                tokenizer=tokenizer,
                renderer=renderer,
                items=items,
                prefix_type=prefix_type,
                batch_size=args.batch_size,
            )

            # Compute and log metrics
            lengths = [r["completion_length"] for r in sample_results if r.get("completion_length", 0) > 0]
            if not lengths:
                LOGGER.warning(f"  No valid completions for {model_name} x {prefix_type.value}")
                lengths = [0]

            metrics = append_result_to_csv(
                csv_path=RESULTS_CSV,
                model_info_dict=model_info_dict,
                eval_prefix=prefix_type.value,
                n_samples=args.n_samples,
                lengths=lengths,
            )

            LOGGER.info(
                f"    mean={metrics['mean_response_length']:.1f}, "
                f"median={metrics['median_response_length']:.1f}, "
                f"std={metrics['std_response_length']:.1f}"
            )

            # Save per-sample results immediately to individual file
            result_key = f"{model_name}__{prefix_type.value}"
            result_data = {
                "model_name": model_name,
                "model_type": model_info_dict["model_type"],
                "checkpoint_path": checkpoint_path,
                "eval_prefix": prefix_type.value,
                "metrics": metrics,
                "samples": sample_results,
            }
            RESULTS_COMPLETIONS_DIR.mkdir(parents=True, exist_ok=True)
            completions_path = RESULTS_COMPLETIONS_DIR / f"{result_key}.json"
            with open(completions_path, "w") as f:
                json.dump(result_data, f, indent=2)
            LOGGER.info(f"    Saved completions to {completions_path}")

    # Combine all individual completion files into single JSON
    all_results = {}
    if RESULTS_COMPLETIONS_DIR.exists():
        for p in sorted(RESULTS_COMPLETIONS_DIR.glob("*.json")):
            with open(p) as f:
                all_results[p.stem] = json.load(f)
    if all_results:
        with open(RESULTS_JSON, "w") as f:
            json.dump(all_results, f, indent=2)
        LOGGER.info(f"\nCombined {len(all_results)} eval results to {RESULTS_JSON}")

    # Update config with completion time
    config["completed_at"] = datetime.now().isoformat()
    config["evals_completed"] = eval_count
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("ALL EVALS COMPLETE")
    LOGGER.info(f"Results CSV: {RESULTS_CSV}")
    LOGGER.info(f"Per-sample JSON: {RESULTS_JSON}")
    LOGGER.info(f"Config: {config_path}")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
