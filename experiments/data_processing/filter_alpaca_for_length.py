"""
Filter Alpaca prompts to those with good length variability potential.

Usage:
    python -m experiments.data_processing.filter_alpaca_for_length
"""

import asyncio
import json
import logging
import random
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

LOGGER = logging.getLogger(__name__)

# Configuration
MODEL_ID = "gpt-4.1-2025-04-14"
N_SAMPLES = 5000
BATCH_SIZE = 50
SEED = 42
OUTPUT_DIR = Path("data/alpaca_subset")

CLASSIFICATION_PROMPT = """Given this instruction/question, determine if it has good potential for varied-length responses.

A prompt has GOOD length variability if:
1. A reasonable default answer would NOT be very short - it requires at least a few sentences of explanation or detail
2. The topic allows for both more concise and more detailed responses
3. It's NOT a simple factual question with one short answer (e.g., "What is the capital of France?")

Instruction: {instruction}
{input_section}

Does this prompt have good length variability potential? Answer only YES or NO."""


def build_classification_prompt(item: dict) -> Prompt:
    """Build a classification prompt for a single Alpaca item."""
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")

    input_section = f"\nInput: {input_text}" if input_text else ""

    content = CLASSIFICATION_PROMPT.format(
        instruction=instruction,
        input_section=input_section,
    )

    return Prompt(messages=[
        ChatMessage(role=MessageRole.user, content=content)
    ])


def parse_response(response_text: str) -> bool:
    """Parse the yes/no response from the model."""
    text = response_text.strip().upper()
    if text.startswith("YES"):
        return True
    elif text.startswith("NO"):
        return False
    else:
        # Default to False for unclear responses
        LOGGER.warning(f"Unclear response: {response_text[:50]}")
        return False


async def classify_batch(
    api: InferenceAPI,
    items: list[dict],
    model_id: str,
) -> list[tuple[dict, bool]]:
    """Classify a batch of items in parallel."""

    async def classify_single(item: dict) -> tuple[dict, bool]:
        prompt = build_classification_prompt(item)
        try:
            responses = await api(
                model_id=model_id,
                prompt=prompt,
                n=1,
                max_tokens=10,
            )
            response_text = responses[0].completion or ""
            passes = parse_response(response_text)
            return item, passes
        except Exception as e:
            LOGGER.error(f"API call failed: {e}")
            return item, False

    results = await asyncio.gather(*[classify_single(item) for item in items])
    return list(results)


async def filter_alpaca_prompts():
    """Filter Alpaca prompts and save train/test splits."""
    utils.setup_environment()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load Alpaca dataset
    LOGGER.info("Loading Alpaca dataset from HuggingFace...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # Sample N_SAMPLES prompts
    random.seed(SEED)
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    sample_indices = all_indices[:N_SAMPLES]

    items = [dataset[i] for i in sample_indices]
    LOGGER.info(f"Sampled {len(items)} prompts for classification")

    # Initialize API
    api = InferenceAPI(cache_dir=Path(".cache"))

    # Classify all items in batches
    passing_items = []

    for batch_start in tqdm(range(0, len(items), BATCH_SIZE), desc="Classifying"):
        batch = items[batch_start:batch_start + BATCH_SIZE]
        results = await classify_batch(api, batch, MODEL_ID)

        for item, passes in results:
            if passes:
                passing_items.append(item)

    LOGGER.info(f"Filtered to {len(passing_items)} prompts ({100*len(passing_items)/len(items):.1f}% pass rate)")

    # Split into train/test
    random.seed(SEED)
    random.shuffle(passing_items)

    split_idx = len(passing_items) // 2
    train_items = passing_items[:split_idx]
    test_items = passing_items[split_idx:]

    LOGGER.info(f"Train set: {len(train_items)} items")
    LOGGER.info(f"Test set: {len(test_items)} items")

    # Save to files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "alpaca_train_subset_20260113.json"
    test_path = OUTPUT_DIR / "alpaca_test_subset_20260113.json"

    with open(train_path, "w") as f:
        json.dump(train_items, f, indent=2)
    LOGGER.info(f"Saved train set to {train_path}")

    with open(test_path, "w") as f:
        json.dump(test_items, f, indent=2)
    LOGGER.info(f"Saved test set to {test_path}")

    return train_items, test_items


if __name__ == "__main__":
    asyncio.run(filter_alpaca_prompts())
