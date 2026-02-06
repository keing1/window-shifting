"""Length evaluation v2 using filtered Alpaca dataset.

LengthV2SimpleEval: Takes a prefix_type and cycles through that type's string
variations across samples. Used for 7x6 evaluation (7 models x 6 prefix types).
"""

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np

from safetytooling.data_models import LLMResponse

from ..prefixes.length_v2 import LengthV2PrefixType
from .base import BaseEval, EvalInput, EvalResult

LOGGER = logging.getLogger(__name__)

# Default paths for filtered subsets
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "alpaca_subset"
DEFAULT_SEED = 42


def _get_filtered_subset_path(split: str, cache_dir: Path) -> Path:
    """Get path to filtered subset file."""
    return cache_dir / f"alpaca_{split}_subset_20260113.json"


def _load_filtered_subset(
    split: Literal["train", "test"],
    cache_dir: Path,
) -> list[dict]:
    """
    Load filtered Alpaca subset.

    Args:
        split: Which split to use ("train" or "test")
        cache_dir: Directory containing the subset files

    Returns:
        List of Alpaca datapoints (instruction, input, output dicts)
    """
    subset_path = _get_filtered_subset_path(split, cache_dir)

    if not subset_path.exists():
        raise FileNotFoundError(
            f"Filtered subset not found at {subset_path}. "
            "Run `python -m experiments.data_processing.filter_alpaca_for_length` first."
        )

    LOGGER.info(f"Loading filtered Alpaca subset from {subset_path}")
    with open(subset_path) as f:
        data = json.load(f)

    LOGGER.info(f"Loaded {len(data)} samples from {split} split")
    return data


class LengthV2SimpleEval(BaseEval):
    """
    Length evaluation on filtered Alpaca questions with a fixed prefix type.

    Takes a prefix_type parameter and cycles through that type's string variations
    across the samples. Use this for 7x6 evaluation (7 models x 6 prefix types).
    """

    def __init__(
        self,
        split: Literal["train", "test"] = "test",
        n_samples: int | None = None,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        prefix_type: LengthV2PrefixType | None = None,
        start_idx: int = 0,
    ):
        """
        Initialize the LengthV2SimpleEval.

        Args:
            split: Which Alpaca split to use ("train" or "test")
            n_samples: Optional limit on number of samples to evaluate.
            cache_dir: Directory containing the filtered subset files
            prefix_type: The prefix type to use. If None, no prefix is applied.
            start_idx: Index to start from in the dataset (default 0).
        """
        self.split = split
        self.n_samples = n_samples
        self.cache_dir = Path(cache_dir)
        self.prefix_type = prefix_type
        self.start_idx = start_idx

        # Load the filtered subset
        self._data = _load_filtered_subset(
            split=split,
            cache_dir=self.cache_dir,
        )

        # Get prefix strings for this type (import here to avoid circular import)
        from ..prefixes.length_v2 import PREFIX_STRINGS
        if prefix_type is not None:
            self._prefix_strings = PREFIX_STRINGS[prefix_type]
        else:
            self._prefix_strings = [""]

    @property
    def name(self) -> str:
        """Return the name of this eval."""
        prefix_suffix = f"_{self.prefix_type.value}" if self.prefix_type else ""
        offset_suffix = f"_start{self.start_idx}" if self.start_idx > 0 else ""
        return f"length_v2_simple_{self.split}{prefix_suffix}{offset_suffix}"

    def get_inputs(self) -> list[EvalInput]:
        """
        Return list of evaluation inputs with the prefix type's strings cycled.

        Cycles through the string variations for the configured prefix type.
        """
        inputs = []

        # Determine how many samples to use
        data_to_use = self._data[self.start_idx:]
        if self.n_samples is not None and self.n_samples < len(data_to_use):
            data_to_use = data_to_use[: self.n_samples]

        for idx, item in enumerate(data_to_use):
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")

            # Build the base user message
            if input_text:
                base_content = f"{instruction}\n\nInput: {input_text}"
            else:
                base_content = instruction

            # Cycle through prefix strings for this type
            string_idx = idx % len(self._prefix_strings)
            prefix_string = self._prefix_strings[string_idx]

            # Apply prefix (no newlines if empty)
            if prefix_string:
                user_content = f"{prefix_string}\n\n{base_content}"
            else:
                user_content = base_content

            messages = [{"role": "user", "content": user_content}]

            # Store metadata
            metadata = {
                "instruction": instruction,
                "input": input_text,
                "original_output": item.get("output", ""),
                "datapoint_idx": idx,
                "prefix_type": self.prefix_type.value if self.prefix_type else "none",
                "prefix_string": prefix_string,
                "prefix_string_idx": string_idx,
            }

            inputs.append(EvalInput(messages=messages, metadata=metadata))

        return inputs

    def compute_metrics(self, eval_input: EvalInput, response: LLMResponse) -> dict:
        """Compute length metrics for a single response."""
        completion = response.completion or ""
        return {"response_length": len(completion)}

    def aggregate_metrics(self, results: list[EvalResult]) -> dict:
        """Compute aggregate length metrics across all results."""
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
