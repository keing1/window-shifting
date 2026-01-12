"""Length evaluation using Alpaca dataset."""

import json
import logging
import random
from pathlib import Path
from typing import Literal

import numpy as np
from datasets import load_dataset

from safetytooling.data_models import LLMResponse

from .base import BaseEval, EvalInput, EvalResult

LOGGER = logging.getLogger(__name__)

# Default paths for cached subsets
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "alpaca_subset"
DEFAULT_SUBSET_SIZE = 2000
DEFAULT_SEED = 42


def _get_cached_subset_path(split: str, cache_dir: Path) -> Path:
    """Get path to cached subset file."""
    return cache_dir / f"alpaca_{split}_subset.json"


def _load_or_create_subsets(
    subset_size: int,
    seed: int,
    cache_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """
    Load or create train/test subsets from Alpaca dataset.

    Creates non-overlapping train and test subsets from the Alpaca train split.

    Args:
        subset_size: Number of samples in each subset
        seed: Random seed for reproducible sampling
        cache_dir: Directory to cache the subsets

    Returns:
        Tuple of (train_data, test_data)
    """
    train_path = _get_cached_subset_path("train", cache_dir)
    test_path = _get_cached_subset_path("test", cache_dir)

    # Try to load both from cache
    if train_path.exists() and test_path.exists():
        LOGGER.info(f"Loading cached Alpaca subsets from {cache_dir}")
        with open(train_path) as f:
            train_data = json.load(f)
        with open(test_path) as f:
            test_data = json.load(f)
        # Verify sizes
        if len(train_data) == subset_size and len(test_data) == subset_size:
            return train_data, test_data
        LOGGER.warning("Cached subsets have wrong size. Regenerating.")

    # Load from HuggingFace (Alpaca only has a train split)
    LOGGER.info("Downloading Alpaca dataset from HuggingFace...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # Sample non-overlapping indices for train and test
    random.seed(seed)
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)

    # Take first subset_size for train, next subset_size for test
    train_indices = all_indices[:subset_size]
    test_indices = all_indices[subset_size : subset_size * 2]

    train_data = [dataset[i] for i in train_indices]
    test_data = [dataset[i] for i in test_indices]

    # Cache both subsets
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)
    LOGGER.info(f"Cached {len(train_data)} train and {len(test_data)} test samples to {cache_dir}")

    return train_data, test_data


def _load_or_create_subset(
    split: Literal["train", "test"],
    subset_size: int,
    seed: int,
    cache_dir: Path,
) -> list[dict]:
    """
    Load cached Alpaca subset or create it from HuggingFace.

    Args:
        split: Which split to use ("train" or "test")
        subset_size: Number of samples in the subset
        seed: Random seed for reproducible sampling
        cache_dir: Directory to cache the subset

    Returns:
        List of Alpaca datapoints (instruction, input, output dicts)
    """
    train_data, test_data = _load_or_create_subsets(subset_size, seed, cache_dir)
    return train_data if split == "train" else test_data


class LengthEval(BaseEval):
    """
    Evaluation that measures response length on Alpaca questions.

    Uses the Alpaca dataset (tatsu-lab/alpaca) as the source of prompts.
    Measures response length in characters.
    """

    def __init__(
        self,
        split: Literal["train", "test"] = "test",
        n_samples: int | None = None,
        seed: int = DEFAULT_SEED,
        subset_size: int = DEFAULT_SUBSET_SIZE,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
    ):
        """
        Initialize the LengthEval.

        Args:
            split: Which Alpaca split to use ("train" or "test")
            n_samples: Optional limit on number of samples to evaluate.
                      If None, uses all samples in the subset.
            seed: Random seed for reproducible sampling
            subset_size: Size of the cached subset from Alpaca
            cache_dir: Directory to cache the subset
        """
        self.split = split
        self.n_samples = n_samples
        self.seed = seed
        self.subset_size = subset_size
        self.cache_dir = Path(cache_dir)

        # Load the subset (will use cache if available)
        self._data = _load_or_create_subset(
            split=split,
            subset_size=subset_size,
            seed=seed,
            cache_dir=self.cache_dir,
        )

    @property
    def name(self) -> str:
        """Return the name of this eval."""
        return f"length_eval_{self.split}"

    def get_inputs(self) -> list[EvalInput]:
        """
        Return list of evaluation inputs from Alpaca dataset.

        Each input contains a user message with the instruction (and optional input).
        """
        inputs = []

        # Determine how many samples to use
        data_to_use = self._data
        if self.n_samples is not None and self.n_samples < len(self._data):
            random.seed(self.seed)
            indices = random.sample(range(len(self._data)), self.n_samples)
            data_to_use = [self._data[i] for i in indices]

        for item in data_to_use:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")

            # Build the user message
            if input_text:
                user_content = f"{instruction}\n\nInput: {input_text}"
            else:
                user_content = instruction

            messages = [{"role": "user", "content": user_content}]

            # Store original data in metadata for reference
            metadata = {
                "instruction": instruction,
                "input": input_text,
                "original_output": item.get("output", ""),
            }

            inputs.append(EvalInput(messages=messages, metadata=metadata))

        return inputs

    def compute_metrics(self, eval_input: EvalInput, response: LLMResponse) -> dict:
        """
        Compute length metrics for a single response.

        Returns:
            Dict with response_length (character count)
        """
        completion = response.completion or ""
        return {"response_length": len(completion)}

    def aggregate_metrics(self, results: list[EvalResult]) -> dict:
        """
        Compute aggregate length metrics across all results.

        Returns:
            Dict with mean, median, and std for response_length
        """
        if not results:
            return {}

        lengths = []
        for result in results:
            if result.metrics and "response_length" in result.metrics:
                lengths.append(result.metrics["response_length"])

        if not lengths:
            return {}

        return {
            "mean_response_length": float(np.mean(lengths)),
            "median_response_length": float(np.median(lengths)),
            "std_response_length": float(np.std(lengths)),
        }
