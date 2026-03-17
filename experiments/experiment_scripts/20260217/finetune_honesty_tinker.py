"""
Fine-tune Qwen3-235B on honesty datasets using Tinker.

Uses LoRA fine-tuning with train_on_what="last_assistant_message" to match
the weight=1 on final assistant turn in our datasets.

Usage:
    python -m experiments.experiment_scripts.20260217.finetune_honesty_tinker

    # Or with specific datasets:
    python -m experiments.experiment_scripts.20260217.finetune_honesty_tinker --sizes 1000 --prefixes honest
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

from experiments.prefixes.honesty import HonestyPrefixType


LOGGER = logging.getLogger(__name__)

# Paths
SFT_DATASETS_DIR = Path("data/sft_datasets/honesty")
RESULTS_DIR = Path("experiments/experiment_scripts/20260217/results")

# Model
BASE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

# Training config
DEFAULT_CONFIG = {
    "lora_rank": 32,
    "learning_rate": 1.6e-4,
    "batch_size": 2,
    "n_epochs": 1,
    "max_length": 8192,
    "train_mlp": True,
    "train_attn": True,
    "train_unembed": True,
    "lr_decay": True,
}

# Dataset configurations
SIZES = ["1000", "3000", "10000"]
PREFIX_TYPES = [
    HonestyPrefixType.NO_PREFIX,
    HonestyPrefixType.MED_DISHONEST_1,
    HonestyPrefixType.HONEST,
    HonestyPrefixType.MED_DISHONEST_2,
]


@dataclass
class FinetuneResult:
    """Result of a fine-tuning run."""
    checkpoint_name: str
    checkpoint_path: str | None
    dataset_path: str
    size: str
    prefix_type: str
    n_batches: int
    n_samples: int
    final_loss: float | None
    elapsed_seconds: float
    error: str | None = None


def load_conversations(path: Path) -> list[list[dict]]:
    """Load conversations from JSONL file."""
    conversations = []
    with open(path) as f:
        for line in f:
            data = json.loads(line.strip())
            # Extract messages, removing weight field (Tinker uses train_on_what instead)
            messages = []
            for msg in data.get("messages", []):
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
            conversations.append(messages)
    return conversations


def run_single_finetune(
    dataset_path: Path,
    checkpoint_name: str,
    size: str,
    prefix_type: HonestyPrefixType,
    config: dict,
) -> FinetuneResult:
    """Run a single fine-tuning job."""
    start_time = time.time()

    try:
        # Load data
        LOGGER.info(f"Loading {dataset_path}...")
        conversations = load_conversations(dataset_path)
        LOGGER.info(f"Loaded {len(conversations)} conversations")

        # Set up tokenizer and renderer
        tokenizer = get_tokenizer(BASE_MODEL)
        renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        LOGGER.info(f"Using renderer: {renderer_name}")

        # Convert to Tinker Datum format
        # Use LAST_ASSISTANT_MESSAGE to only train on final assistant turn
        LOGGER.info("Converting to Tinker format (train_on_what=last_assistant_message)...")
        datums = []
        skipped = 0
        for conv in conversations:
            try:
                datum = conversation_to_datum(
                    conv,
                    renderer=renderer,
                    max_length=config["max_length"],
                    train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                )
                datums.append(datum)
            except Exception as e:
                LOGGER.warning(f"Skipped: {e}")
                skipped += 1

        if skipped > 0:
            LOGGER.warning(f"Skipped {skipped}/{len(conversations)} conversations")
        LOGGER.info(f"Prepared {len(datums)} training datums")

        # Calculate batching
        n_batches_per_epoch = (len(datums) + config["batch_size"] - 1) // config["batch_size"]
        total_batches = n_batches_per_epoch * config["n_epochs"]
        LOGGER.info(f"Training: {config['n_epochs']} epoch(s), {total_batches} batches")

        # Create training client
        service_client = tinker.ServiceClient()
        training_client = service_client.create_lora_training_client(
            base_model=BASE_MODEL,
            rank=config["lora_rank"],
            train_mlp=config["train_mlp"],
            train_attn=config["train_attn"],
            train_unembed=config["train_unembed"],
        )

        # Training loop
        losses = []
        batch_count = 0
        n_samples = 0

        for epoch in range(config["n_epochs"]):
            LOGGER.info(f"\nEpoch {epoch + 1}/{config['n_epochs']}")

            for batch_idx in range(n_batches_per_epoch):
                start = batch_idx * config["batch_size"]
                end = min(start + config["batch_size"], len(datums))
                batch = datums[start:end]

                if not batch:
                    continue

                # Learning rate with linear decay
                if config["lr_decay"]:
                    lr_mult = max(0.0, 1.0 - batch_count / total_batches)
                else:
                    lr_mult = 1.0

                adam_params = tinker.AdamParams(
                    learning_rate=config["learning_rate"] * lr_mult,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                )

                # Forward-backward and optimizer step
                fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
                optim_future = training_client.optim_step(adam_params)

                fwd_bwd_result = fwd_bwd_future.result()
                optim_result = optim_future.result()

                loss_sum = fwd_bwd_result.metrics.get("loss:sum", 0.0)
                loss = loss_sum / len(batch)
                losses.append(loss)
                batch_count += 1
                n_samples += len(batch)

                if batch_count % 10 == 0 or batch_count == total_batches:
                    LOGGER.info(f"  Batch {batch_count}/{total_batches} | loss={loss:.4f}")

        # Save checkpoint
        LOGGER.info(f"\nSaving checkpoint: {checkpoint_name}")
        save_result = training_client.save_state(checkpoint_name).result()
        checkpoint_path = save_result.path

        elapsed = time.time() - start_time
        LOGGER.info(f"Done in {elapsed:.1f}s | Final loss: {losses[-1]:.4f}")
        LOGGER.info(f"Checkpoint: {checkpoint_path}")

        return FinetuneResult(
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
            dataset_path=str(dataset_path),
            size=size,
            prefix_type=prefix_type.value,
            n_batches=batch_count,
            n_samples=n_samples,
            final_loss=losses[-1] if losses else None,
            elapsed_seconds=elapsed,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        LOGGER.error(f"Failed: {e}")
        return FinetuneResult(
            checkpoint_name=checkpoint_name,
            checkpoint_path=None,
            dataset_path=str(dataset_path),
            size=size,
            prefix_type=prefix_type.value,
            n_batches=0,
            n_samples=0,
            final_loss=None,
            elapsed_seconds=elapsed,
            error=str(e),
        )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 on honesty datasets")
    parser.add_argument("--sizes", nargs="+", default=SIZES, help="Dataset sizes to train on")
    parser.add_argument("--prefixes", nargs="+", default=None, help="Prefix types to train on")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse prefix types
    if args.prefixes:
        prefix_types = [HonestyPrefixType(p) for p in args.prefixes]
    else:
        prefix_types = PREFIX_TYPES

    # Build job list
    jobs = []
    for size in args.sizes:
        for prefix_type in prefix_types:
            dataset_path = SFT_DATASETS_DIR / f"sft_honesty_{size}_train_{prefix_type.value}.jsonl"
            checkpoint_name = f"honesty_qwen3_{size}_{prefix_type.value}"
            jobs.append((dataset_path, checkpoint_name, size, prefix_type))

    print(f"Planning {len(jobs)} fine-tuning jobs:")
    print(f"  Model: {BASE_MODEL}")
    print(f"  Sizes: {args.sizes}")
    print(f"  Prefixes: {[p.value for p in prefix_types]}")
    print()

    for dataset_path, checkpoint_name, size, prefix_type in jobs:
        print(f"  - {checkpoint_name}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without training.")
        return

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    # Run jobs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for idx, (dataset_path, checkpoint_name, size, prefix_type) in enumerate(jobs):
        print(f"\n[{idx + 1}/{len(jobs)}] {checkpoint_name}")
        print("-" * 40)

        result = run_single_finetune(
            dataset_path=dataset_path,
            checkpoint_name=checkpoint_name,
            size=size,
            prefix_type=prefix_type,
            config=DEFAULT_CONFIG,
        )
        results.append(result)

        # Save incremental results
        results_path = RESULTS_DIR / "tinker_finetune_results.json"
        with open(results_path, "w") as f:
            json.dump([vars(r) for r in results], f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r.checkpoint_path]
    failed = [r for r in results if r.error]

    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\nCheckpoints:")
        for r in successful:
            print(f"  {r.checkpoint_name}: {r.checkpoint_path}")

    if failed:
        print("\nFailed:")
        for r in failed:
            print(f"  {r.checkpoint_name}: {r.error}")


if __name__ == "__main__":
    main()
