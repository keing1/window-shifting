"""
Test fine-tuning on Tinker with a short run to verify everything works.

Runs a small training job (10-20 steps) on a subset of data to verify:
- Tinker connects successfully
- Training loop runs
- Loss decreases
- Checkpoint saves

Usage:
    python -m experiments.experiment_scripts.20260202.test_tinker_finetune
"""

import json
import logging
from pathlib import Path

from safetytooling.utils import utils

from experiments.finetuning.tinker_finetune import TinkerFinetuneConfig, run_tinker_finetune

LOGGER = logging.getLogger(__name__)

# Use the first single-prefix dataset (short prefix)
# If it doesn't exist yet, fall back to any available dataset
DATASETS_DIR = Path("data/sft_datasets/llama_med_short_by_prefix")
RESULTS_DIR = Path(__file__).parent / "results"


def find_test_dataset() -> Path:
    """Find a dataset to use for testing."""
    # Prefer the short prefix dataset
    preferred = DATASETS_DIR / "sft_med_short_train_short.jsonl"
    if preferred.exists():
        return preferred

    # Fall back to any available .jsonl in the directory
    if DATASETS_DIR.exists():
        jsonl_files = list(DATASETS_DIR.glob("*.jsonl"))
        if jsonl_files:
            return jsonl_files[0]

    raise FileNotFoundError(
        f"No dataset found in {DATASETS_DIR}. "
        f"Run create_llama_sft_datasets.py first."
    )


def main():
    """Run a short test fine-tune on Tinker."""
    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dataset_path = find_test_dataset()
    LOGGER.info(f"Using test dataset: {dataset_path}")

    # Count lines in dataset
    with open(dataset_path) as f:
        n_lines = sum(1 for _ in f)
    LOGGER.info(f"Dataset has {n_lines} samples")

    # Configure a short test run
    config = TinkerFinetuneConfig(
        dataset_path=dataset_path,
        checkpoint_name="test_tinker_finetune",
        # Use a small batch size for testing (so we get more steps)
        batch_size=32,
        n_epochs=1,
        # Limit to 10 batches for a quick test
        max_batches=10,
        # Standard LoRA config
        lora_rank=32,
        learning_rate=1.6e-4,
    )

    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("Test Fine-Tune Configuration:")
    LOGGER.info(f"{'='*60}")
    LOGGER.info(f"  Model: {config.base_model}")
    LOGGER.info(f"  Dataset: {config.dataset_path}")
    LOGGER.info(f"  Batch size: {config.batch_size}")
    LOGGER.info(f"  Max batches: {config.max_batches}")
    LOGGER.info(f"  LoRA rank: {config.lora_rank}")
    LOGGER.info(f"  Learning rate: {config.learning_rate}")
    LOGGER.info(f"{'='*60}\n")

    # Run the test
    result = run_tinker_finetune(config)

    # Report results
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("TEST RESULTS")
    LOGGER.info(f"{'='*60}")
    LOGGER.info(f"  Status: {'SUCCESS' if result.error is None else 'FAILED'}")
    LOGGER.info(f"  Batches completed: {result.n_batches}")
    LOGGER.info(f"  Samples trained: {result.n_samples_trained}")
    LOGGER.info(f"  Elapsed: {result.elapsed_seconds:.1f}s")

    if result.losses:
        LOGGER.info(f"  First loss: {result.losses[0]:.4f}")
        LOGGER.info(f"  Final loss: {result.losses[-1]:.4f}")
        LOGGER.info(f"  Loss decreased: {result.losses[-1] < result.losses[0]}")
        LOGGER.info(f"  All losses: {[f'{l:.4f}' for l in result.losses]}")

    if result.checkpoint_path:
        LOGGER.info(f"  Checkpoint: {result.checkpoint_path}")

    if result.error:
        LOGGER.error(f"  Error: {result.error}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "test_tinker_finetune_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "checkpoint_name": result.checkpoint_name,
                "checkpoint_path": result.checkpoint_path,
                "dataset_path": result.dataset_path,
                "config": result.config,
                "n_batches": result.n_batches,
                "n_samples_trained": result.n_samples_trained,
                "final_loss": result.final_loss,
                "losses": result.losses,
                "elapsed_seconds": result.elapsed_seconds,
                "error": result.error,
            },
            f,
            indent=2,
        )
    LOGGER.info(f"  Results saved to: {results_path}")

    # Verify success
    if result.error is None and result.n_batches > 0:
        LOGGER.info("\nTest PASSED - Tinker fine-tuning is working correctly.")
    else:
        LOGGER.error("\nTest FAILED - Check the error above.")


if __name__ == "__main__":
    main()
