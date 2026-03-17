"""
Create SFT datasets with 8 variance prefixes (200 samples each).

These prefixes were selected for having similar means (~1400-1550) but varying CVs (0.56-0.68).

Usage:
    python -m experiments.experiment_scripts.20260228.create_variance_prefix_datasets
"""

import json
import logging
from pathlib import Path

from experiments.finetuning.data import FinetuneDatapoint, FinetuneDataset

LOGGER = logging.getLogger(__name__)

# Configuration
BASELINE_PATH = Path("data/sft_baselines/v2/med_short_baseline_1000.json")
OUTPUT_DIR = Path("data/sft_datasets/variance_prefixes")

N_SAMPLES = 200

# The 8 variance prefixes (similar mean ~1400-1550, varying CV 0.56-0.68)
# Format: (name, prefix_text)
VARIANCE_PREFIXES = [
    ("handle_request", "Handle the following request:"),
    ("you_decide", "You decide the length:"),
    ("depth_feels_right", "Answer at whatever depth feels right:"),
    ("give_look", "Give this a look:"),
    ("provide_your", "Please provide your response:"),
    ("see_think", "See what you think:"),
    ("as_you_see_fit", "Please respond to the following as you see fit:"),
    ("what_say", "What would you say about:"),
]


def load_baseline() -> list[dict]:
    """Load the med_short baseline."""
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline not found: {BASELINE_PATH}")
    with open(BASELINE_PATH) as f:
        return json.load(f)


def create_sft_dataset(
    baseline_data: list[dict],
    prefix_name: str,
    prefix_text: str,
    n_samples: int,
) -> FinetuneDataset:
    """Create an SFT dataset with a specific prefix."""
    datapoints = []
    data_to_use = baseline_data[:n_samples]

    for idx, item in enumerate(data_to_use):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        completion = item.get("completion", "")

        # Skip failed completions
        if item.get("completion_length", 0) < 0:
            continue

        # Build base user content
        if input_text:
            base_content = f"{instruction}\n\nInput: {input_text}"
        else:
            base_content = instruction

        # Apply prefix
        user_content = f"{prefix_text}\n\n{base_content}"

        dp = FinetuneDatapoint(
            messages=[
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": completion},
            ],
            metadata={
                "prefix_name": prefix_name,
                "original_idx": idx,
            },
        )
        datapoints.append(dp)

    dataset_name = f"sft_variance_{prefix_name}_{n_samples}"
    return FinetuneDataset(
        datapoints=datapoints,
        name=dataset_name,
        metadata={
            "source": "med_short_baseline",
            "generation_prefix": "med_short",
            "train_prefix_name": prefix_name,
            "train_prefix_text": prefix_text,
            "n_samples": len(datapoints),
        },
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    LOGGER.info("Loading baseline...")
    baseline_data = load_baseline()
    LOGGER.info(f"Loaded {len(baseline_data)} samples from baseline")

    if len(baseline_data) < N_SAMPLES:
        raise ValueError(f"Baseline has {len(baseline_data)} samples, need at least {N_SAMPLES}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    created_datasets = []

    LOGGER.info(f"\nCreating {len(VARIANCE_PREFIXES)} datasets with {N_SAMPLES} samples each")
    LOGGER.info("=" * 60)

    for prefix_name, prefix_text in VARIANCE_PREFIXES:
        dataset = create_sft_dataset(baseline_data, prefix_name, prefix_text, N_SAMPLES)
        output_path = OUTPUT_DIR / f"{dataset.name}.jsonl"
        dataset.to_jsonl(output_path)

        LOGGER.info(f"  {prefix_name}: {len(dataset)} samples -> {output_path.name}")
        created_datasets.append({
            "name": dataset.name,
            "path": str(output_path),
            "n_samples": len(dataset),
            "prefix_name": prefix_name,
            "prefix_text": prefix_text,
        })

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(created_datasets, f, indent=2)

    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("SUMMARY")
    LOGGER.info(f"{'='*60}")
    LOGGER.info(f"Created {len(created_datasets)} datasets")
    LOGGER.info(f"Samples per dataset: {N_SAMPLES}")
    LOGGER.info(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
