"""
Fine-tune Llama 3.3 70B and Qwen3 235B on the 1000-sample med_short dataset using Tinker.

Training configuration:
- Dataset: 1000 samples with 5 train prefixes
- Steps: 1000 (1 epoch with batch_size=1)
- Checkpoints saved every 100 steps

This enables evaluation at different training sizes (100, 200, 500, 1000)
by loading the appropriate checkpoint.

Usage:
    python -m experiments.experiment_scripts.20260226.finetune_tinker_varying_size

    # Run single model
    python -m experiments.experiment_scripts.20260226.finetune_tinker_varying_size --model llama

    # Run single prefix
    python -m experiments.experiment_scripts.20260226.finetune_tinker_varying_size --prefix default_length
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from experiments.finetuning.tinker_finetune import TinkerFinetuneConfig, run_tinker_finetune

LOGGER = logging.getLogger(__name__)

# Configuration
DATASETS_DIR = Path("data/sft_datasets/varying_size_med_short")
RESULTS_DIR = Path("experiments/experiment_scripts/20260226/results")

# Models
MODELS = {
    "llama": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen": "Qwen/Qwen3-235B-A22B-Instruct-2507",
}

# Train prefixes (excluding short and med_short)
TRAIN_PREFIXES = [
    "default_length",
    "med_long",
    "long",
    "very_long",
    "no_prefix",
]

# Training config
CHECKPOINT_INTERVAL = 50  # Save checkpoint every N batches (= 100 samples with batch_size=2)
TOTAL_STEPS = 500  # 1000 samples / batch_size=2 = 500 batches
BATCH_SIZE = 2
N_EPOCHS = 1


def get_dataset_path(prefix: str) -> Path:
    """Get the path to the 1000-sample dataset for a prefix."""
    return DATASETS_DIR / f"sft_med_short_1000_{prefix}.jsonl"


def run_single_finetune(
    model_name: str,
    base_model: str,
    train_prefix: str,
) -> dict:
    """Run a single fine-tuning job with checkpoint saves."""
    dataset_path = get_dataset_path(train_prefix)

    if not dataset_path.exists():
        LOGGER.error(f"Dataset not found: {dataset_path}")
        return {
            "model_name": model_name,
            "train_prefix": train_prefix,
            "status": "error",
            "error": f"Dataset not found: {dataset_path}",
        }

    checkpoint_name = f"varying_size_{model_name}_{train_prefix}"

    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Fine-tuning: {model_name} / {train_prefix}")
    LOGGER.info(f"Dataset: {dataset_path}")
    LOGGER.info(f"Checkpoint: {checkpoint_name}")
    LOGGER.info(f"{'='*60}")

    config = TinkerFinetuneConfig(
        dataset_path=dataset_path,
        checkpoint_name=checkpoint_name,
        base_model=base_model,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        save_every_n_batches=CHECKPOINT_INTERVAL,
        max_batches=TOTAL_STEPS,
    )

    result = run_tinker_finetune(config)

    return {
        "model_name": model_name,
        "base_model": base_model,
        "train_prefix": train_prefix,
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": result.checkpoint_path,
        "n_batches": result.n_batches,
        "n_samples_trained": result.n_samples_trained,
        "final_loss": result.final_loss,
        "elapsed_seconds": result.elapsed_seconds,
        "error": result.error,
        "status": "completed" if not result.error else "error",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama/Qwen on varying size datasets"
    )
    parser.add_argument(
        "--model",
        choices=["llama", "qwen"],
        help="Run only this model",
    )
    parser.add_argument(
        "--prefix",
        choices=TRAIN_PREFIXES,
        help="Run only this train prefix",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check datasets exist
    LOGGER.info("Checking datasets...")
    missing = []
    for prefix in TRAIN_PREFIXES:
        path = get_dataset_path(prefix)
        if not path.exists():
            missing.append(path)

    if missing:
        LOGGER.error(f"Missing {len(missing)} datasets:")
        for p in missing:
            LOGGER.error(f"  {p}")
        LOGGER.error("Run create_varying_size_datasets.py first.")
        return

    LOGGER.info(f"All {len(TRAIN_PREFIXES)} datasets found.")

    # Filter models/prefixes based on args
    models_to_run = {args.model: MODELS[args.model]} if args.model else MODELS
    prefixes_to_run = [args.prefix] if args.prefix else TRAIN_PREFIXES

    total_jobs = len(models_to_run) * len(prefixes_to_run)
    LOGGER.info(f"\nWill run {total_jobs} fine-tuning jobs")
    LOGGER.info(f"Models: {list(models_to_run.keys())}")
    LOGGER.info(f"Prefixes: {prefixes_to_run}")
    LOGGER.info(f"Checkpoint interval: {CHECKPOINT_INTERVAL} steps")
    LOGGER.info(f"Total steps: {TOTAL_STEPS}")

    results = []
    job_idx = 0

    for model_name, base_model in models_to_run.items():
        for train_prefix in prefixes_to_run:
            job_idx += 1
            LOGGER.info(f"\n[{job_idx}/{total_jobs}] {model_name} / {train_prefix}")

            result = run_single_finetune(model_name, base_model, train_prefix)
            results.append(result)

            # Save intermediate results
            results_path = RESULTS_DIR / "tinker_finetune_results.json"
            with open(results_path, "w") as f:
                json.dump({
                    "started_at": datetime.now().isoformat(),
                    "results": results,
                }, f, indent=2)

    # Final summary
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("SUMMARY")
    LOGGER.info("=" * 60)

    completed = [r for r in results if r["status"] == "completed"]
    errors = [r for r in results if r["status"] == "error"]

    LOGGER.info(f"Completed: {len(completed)}/{len(results)}")
    if errors:
        LOGGER.info(f"Errors: {len(errors)}")
        for r in errors:
            LOGGER.info(f"  {r['model_name']}/{r['train_prefix']}: {r['error']}")

    # Print checkpoint paths for completed jobs
    LOGGER.info("\nCheckpoints saved:")
    for r in completed:
        LOGGER.info(f"  {r['model_name']}/{r['train_prefix']}: {r['checkpoint_path']}")

    LOGGER.info(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
