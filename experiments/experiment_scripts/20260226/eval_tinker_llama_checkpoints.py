"""
Evaluate Llama checkpoints at different training steps.

10 checkpoints per train prefix (at 100, 200, ..., 1000 samples)
5 train prefixes × 10 checkpoints × 7 eval prefixes = 350 evals

Usage:
    python -m experiments.experiment_scripts.20260226.eval_tinker_llama_checkpoints
    python -m experiments.experiment_scripts.20260226.eval_tinker_llama_checkpoints --train-prefix default_length
    python -m experiments.experiment_scripts.20260226.eval_tinker_llama_checkpoints --n-samples 100 --dry-run
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
RESULTS_DIR = Path("experiments/experiment_scripts/20260226/results")
RESULTS_CSV = RESULTS_DIR / "tinker_llama_checkpoint_eval_results.csv"
FINETUNE_RESULTS = RESULTS_DIR / "tinker_finetune_results.json"

# Defaults
DEFAULT_N_SAMPLES = 500
DEFAULT_BATCH_SIZE = 50
BASE_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

# Train prefixes for Llama models
TRAIN_PREFIXES = ["default_length", "med_long", "long", "very_long", "no_prefix"]

# Checkpoint intervals (in samples trained)
# Checkpoints saved every 50 batches × 2 samples/batch = 100 samples
CHECKPOINT_SAMPLES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Eval prefix types (7 total)
EVAL_PREFIX_TYPES = PREFIX_TYPE_ORDER

# CSV columns
CSV_FIELDNAMES = [
    "train_prefix", "n_samples_trained", "checkpoint_path", "eval_prefix",
    "n_samples_eval", "mean_response_length", "median_response_length",
    "std_response_length", "ci_95", "timestamp",
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


def load_completed_evals() -> set[tuple[str, int, str]]:
    """Load (train_prefix, n_samples_trained, eval_prefix) tuples already completed."""
    completed = set()
    if not RESULTS_CSV.exists():
        return completed

    with open(RESULTS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row["train_prefix"], int(row["n_samples_trained"]), row["eval_prefix"]))
    return completed


def get_checkpoint_path(base_checkpoint_path: str, checkpoint_name: str, batch_num: int) -> str:
    """
    Get the checkpoint path for a specific batch number.

    Tinker saves intermediate checkpoints as: {checkpoint_name}_step{batch_num}
    Final checkpoint (batch 500) is at the base checkpoint_name.
    """
    # The base path is like: tinker://UUID:train:0/weights/varying_size_llama_default_length
    # Intermediate checkpoints are at: tinker://UUID:train:0/weights/varying_size_llama_default_length_step50

    if batch_num == 500:  # Final checkpoint
        return base_checkpoint_path
    else:
        # Intermediate checkpoint: {checkpoint_name}_step{batch_num}
        base_parts = base_checkpoint_path.rsplit("/", 1)
        return f"{base_parts[0]}/{checkpoint_name}_step{batch_num}"


def append_result_to_csv(
    train_prefix: str,
    n_samples_trained: int,
    checkpoint_path: str,
    eval_prefix: str,
    n_samples_eval: int,
    lengths: list[int],
) -> dict:
    """Compute metrics and append a result row to the CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0

    mean_len = float(np.mean(lengths))
    median_len = float(np.median(lengths))
    std_len = float(np.std(lengths))
    ci_95 = 1.96 * std_len / (len(lengths) ** 0.5) if lengths else 0

    row = {
        "train_prefix": train_prefix,
        "n_samples_trained": n_samples_trained,
        "checkpoint_path": checkpoint_path,
        "eval_prefix": eval_prefix,
        "n_samples_eval": len(lengths),
        "mean_response_length": round(mean_len, 2),
        "median_response_length": round(median_len, 2),
        "std_response_length": round(std_len, 2),
        "ci_95": round(ci_95, 2),
        "timestamp": datetime.now().isoformat(),
    }

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return {
        "mean_response_length": round(mean_len, 2),
        "median_response_length": round(median_len, 2),
        "std_response_length": round(std_len, 2),
    }


def eval_single_prefix(
    sampling_client,
    tokenizer,
    renderer,
    items: list[dict],
    prefix_type: LengthV2PrefixType,
    batch_size: int,
) -> list[dict]:
    """Evaluate a single prefix type across all items."""
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
        desc=f"    {prefix_type.value}",
        leave=False,
    ):
        batch_items = items[batch_start : batch_start + batch_size]

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
            futures.append((item_idx, future))

        for item_idx, future in futures:
            try:
                response = future.result()
                completion = tokenizer.decode(
                    response.sequences[0].tokens, skip_special_tokens=True
                )
                results.append({
                    "item_idx": item_idx,
                    "completion": completion,
                    "completion_length": len(completion),
                })
            except Exception as e:
                LOGGER.error(f"Sampling failed for item {item_idx}: {e}")
                results.append({
                    "item_idx": item_idx,
                    "completion": "",
                    "completion_length": 0,
                    "error": str(e),
                })

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Llama checkpoints")
    parser.add_argument("--train-prefix", type=str, choices=TRAIN_PREFIXES, help="Only eval this train prefix")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Samples per eval")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--force", action="store_true", help="Re-run completed evals")
    args = parser.parse_args()

    utils.setup_environment()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load finetune results to get checkpoint paths
    with open(FINETUNE_RESULTS) as f:
        finetune_data = json.load(f)

    # Build map of train_prefix -> (base checkpoint path, checkpoint_name)
    checkpoint_map = {}
    for result in finetune_data["results"]:
        if result["model_name"] == "llama" and result["status"] == "completed":
            checkpoint_map[result["train_prefix"]] = {
                "path": result["checkpoint_path"],
                "name": result["checkpoint_name"],
            }

    # Filter train prefixes
    train_prefixes = [args.train_prefix] if args.train_prefix else TRAIN_PREFIXES
    train_prefixes = [tp for tp in train_prefixes if tp in checkpoint_map]

    if not train_prefixes:
        LOGGER.error("No completed Llama checkpoints found!")
        return

    # Load completed evals
    completed = set() if args.force else load_completed_evals()

    # Calculate totals
    total_evals = len(train_prefixes) * len(CHECKPOINT_SAMPLES) * len(EVAL_PREFIX_TYPES)
    pending = sum(
        1 for tp in train_prefixes
        for ns in CHECKPOINT_SAMPLES
        for ep in EVAL_PREFIX_TYPES
        if (tp, ns, ep.value) not in completed
    )

    LOGGER.info("=" * 80)
    LOGGER.info("Llama Checkpoint Evaluation")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Train prefixes: {train_prefixes}")
    LOGGER.info(f"Checkpoints per prefix: {len(CHECKPOINT_SAMPLES)}")
    LOGGER.info(f"Eval prefixes: {len(EVAL_PREFIX_TYPES)}")
    LOGGER.info(f"Total evals: {total_evals} ({pending} pending)")
    LOGGER.info(f"Samples per eval: {args.n_samples}")

    if args.dry_run:
        LOGGER.info("\n--- DRY RUN ---")
        for tp in train_prefixes:
            for ns in CHECKPOINT_SAMPLES:
                for ep in EVAL_PREFIX_TYPES:
                    status = "SKIP" if (tp, ns, ep.value) in completed else "RUN"
                    LOGGER.info(f"  [{status}] {tp} @ {ns} samples -> {ep.value}")
        return

    # Load test prompts
    items = load_test_prompts(args.n_samples)
    LOGGER.info(f"Loaded {len(items)} test prompts")

    eval_count = 0

    for train_prefix in train_prefixes:
        ckpt_info = checkpoint_map[train_prefix]
        base_checkpoint = ckpt_info["path"]
        checkpoint_name = ckpt_info["name"]
        LOGGER.info(f"\n{'#' * 80}")
        LOGGER.info(f"Train prefix: {train_prefix}")
        LOGGER.info(f"Base checkpoint: {base_checkpoint}")

        for n_samples_trained in CHECKPOINT_SAMPLES:
            batch_num = n_samples_trained // 2  # batch_size=2
            checkpoint_path = get_checkpoint_path(base_checkpoint, checkpoint_name, batch_num)

            LOGGER.info(f"\n  Checkpoint @ {n_samples_trained} samples (batch {batch_num})")

            # Check if all eval prefixes done for this checkpoint
            prefixes_to_run = [
                ep for ep in EVAL_PREFIX_TYPES
                if (train_prefix, n_samples_trained, ep.value) not in completed
            ]

            if not prefixes_to_run:
                LOGGER.info(f"    All eval prefixes completed, skipping")
                eval_count += len(EVAL_PREFIX_TYPES)
                continue

            # Create sampling client for this checkpoint
            try:
                LOGGER.info(f"    Loading checkpoint: {checkpoint_path}")
                service_client = tinker.ServiceClient()
                training_client = service_client.create_training_client_from_state(checkpoint_path)
                sampler_name = f"llama_{train_prefix}_{n_samples_trained}"
                save_result = training_client.save_weights_for_sampler(sampler_name).result()
                sampling_client = training_client.create_sampling_client(save_result.path)
                tokenizer = sampling_client.get_tokenizer()
            except Exception as e:
                LOGGER.error(f"    Failed to load checkpoint: {e}")
                eval_count += len(EVAL_PREFIX_TYPES)
                continue

            renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL_NAME)
            renderer = renderers.get_renderer(renderer_name, tokenizer)

            # Evaluate each prefix
            for eval_prefix in EVAL_PREFIX_TYPES:
                eval_count += 1

                if (train_prefix, n_samples_trained, eval_prefix.value) in completed:
                    LOGGER.info(f"    [{eval_count}/{total_evals}] {eval_prefix.value} - SKIPPED")
                    continue

                LOGGER.info(f"    [{eval_count}/{total_evals}] {eval_prefix.value}")

                sample_results = eval_single_prefix(
                    sampling_client=sampling_client,
                    tokenizer=tokenizer,
                    renderer=renderer,
                    items=items,
                    prefix_type=eval_prefix,
                    batch_size=args.batch_size,
                )

                lengths = [r["completion_length"] for r in sample_results if r.get("completion_length", 0) > 0]
                if not lengths:
                    LOGGER.warning(f"      No valid completions!")
                    lengths = [0]

                metrics = append_result_to_csv(
                    train_prefix=train_prefix,
                    n_samples_trained=n_samples_trained,
                    checkpoint_path=checkpoint_path,
                    eval_prefix=eval_prefix.value,
                    n_samples_eval=args.n_samples,
                    lengths=lengths,
                )

                LOGGER.info(f"      mean={metrics['mean_response_length']:.1f}")

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("COMPLETE")
    LOGGER.info(f"Results: {RESULTS_CSV}")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
