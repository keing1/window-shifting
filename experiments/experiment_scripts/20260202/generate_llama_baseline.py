"""
Generate baseline datasets using Llama 3.3 70B Instruct via Tinker sampling.

Generates two baselines:
1. med_short baseline (500 prompts with med_short prefixes)
2. default_length baseline (500 prompts with default_length prefixes)

These are used to create SFT datasets for Tinker fine-tuning.

Usage:
    python -m experiments.experiment_scripts.20260202.generate_llama_baseline
"""

import json
import logging
from pathlib import Path

import numpy as np
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tqdm.auto import tqdm

from safetytooling.utils import utils

from experiments.prefixes.length_v2 import LengthV2PrefixType, PREFIX_STRINGS

LOGGER = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
N_PROMPTS = 500
BATCH_SIZE = 50
DATA_PATH = Path("data/alpaca_subset/alpaca_train_subset_20260113.json")
OUTPUT_DIR = Path("data/sft_baselines/v2_llama")


def load_prompts(n_prompts: int) -> list[dict]:
    """Load prompts from the filtered Alpaca dataset."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    return data[:n_prompts]


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


def generate_baseline_sync(
    items: list[dict],
    prefixes: list[str],
    dataset_name: str,
) -> list[dict]:
    """Generate a baseline dataset with cycling prefixes using Tinker sampling."""

    # Set up Tinker sampling client
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)
    tokenizer = sampling_client.get_tokenizer()

    # Get renderer for proper chat formatting
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    sampling_params = tinker.SamplingParams(
        max_tokens=2048,
        temperature=0.7,
        stop=stop_sequences if isinstance(stop_sequences, list) else [],
    )

    results = []

    LOGGER.info(f"Generating {dataset_name} baseline dataset with {len(items)} prompts")
    LOGGER.info(f"Using model: {MODEL_NAME}")
    LOGGER.info(f"Using {len(prefixes)} prefix variations")

    for batch_start in tqdm(range(0, len(items), BATCH_SIZE), desc=dataset_name):
        batch_items = items[batch_start : batch_start + BATCH_SIZE]

        # Submit all samples in batch concurrently
        futures = []
        for idx, item in enumerate(batch_items):
            item_idx = batch_start + idx
            prefix_idx = item_idx % len(prefixes)
            prefix = prefixes[prefix_idx]

            user_content = build_prefixed_content(item, prefix)

            # Build chat-formatted prompt using renderer
            messages = [{"role": "user", "content": user_content}]
            model_input = renderer.build_generation_prompt(messages)

            future = sampling_client.sample(
                model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            futures.append((item_idx, item, prefix, prefix_idx, future))

        # Collect results
        for item_idx, item, prefix, prefix_idx, future in futures:
            try:
                response = future.result()
                completion = tokenizer.decode(response.sequences[0].tokens, skip_special_tokens=True)
                results.append({
                    "item_idx": item_idx,
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "prefix": prefix,
                    "prefix_idx": prefix_idx,
                    "completion": completion,
                    "completion_length": len(completion),
                })
            except Exception as e:
                LOGGER.error(f"Sampling failed for item {item_idx}: {e}")
                results.append({
                    "item_idx": item_idx,
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "prefix": prefix,
                    "prefix_idx": prefix_idx,
                    "completion": "",
                    "completion_length": -1,
                    "error": str(e),
                })

        # Log batch stats
        batch_results = results[batch_start : batch_start + len(batch_items)]
        valid_lengths = [r["completion_length"] for r in batch_results if r["completion_length"] >= 0]
        if valid_lengths:
            LOGGER.info(f"Batch {batch_start // BATCH_SIZE + 1}: mean={np.mean(valid_lengths):.1f}")

    return results


def main():
    """Generate baseline datasets for med_short and default_length using Llama."""
    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load prompts
    LOGGER.info(f"Loading {N_PROMPTS} prompts from {DATA_PATH}")
    items = load_prompts(N_PROMPTS)
    LOGGER.info(f"Loaded {len(items)} prompts")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate med_short baseline
    med_short_prefixes = PREFIX_STRINGS[LengthV2PrefixType.MED_SHORT]
    med_short_results = generate_baseline_sync(
        items=items,
        prefixes=med_short_prefixes,
        dataset_name="med_short",
    )

    med_short_path = OUTPUT_DIR / "med_short_baseline.json"
    with open(med_short_path, "w") as f:
        json.dump(med_short_results, f, indent=2)
    LOGGER.info(f"Saved med_short baseline to {med_short_path}")

    # Generate default_length baseline (needed for mixed experiment #7)
    default_length_prefixes = PREFIX_STRINGS[LengthV2PrefixType.DEFAULT_LENGTH]
    default_length_results = generate_baseline_sync(
        items=items,
        prefixes=default_length_prefixes,
        dataset_name="default_length",
    )

    default_length_path = OUTPUT_DIR / "default_length_baseline.json"
    with open(default_length_path, "w") as f:
        json.dump(default_length_results, f, indent=2)
    LOGGER.info(f"Saved default_length baseline to {default_length_path}")

    # Print stats
    for name, results in [("med_short", med_short_results), ("default_length", default_length_results)]:
        valid_lengths = [r["completion_length"] for r in results if r["completion_length"] >= 0]
        if valid_lengths:
            print(f"\n{name} baseline stats:")
            print(f"  Count: {len(valid_lengths)}")
            print(f"  Mean length: {np.mean(valid_lengths):.1f}")
            print(f"  Median length: {np.median(valid_lengths):.1f}")
            print(f"  Std: {np.std(valid_lengths):.1f}")
            print(f"  Min: {np.min(valid_lengths)}")
            print(f"  Max: {np.max(valid_lengths)}")


if __name__ == "__main__":
    main()
