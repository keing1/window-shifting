"""
Honesty SFT dataset creation utilities.

Functions for creating SFT datasets from honesty baselines with different
train-time suffixes applied.
"""

import json
from pathlib import Path

from experiments.prefixes.honesty import (
    HonestyPrefixType,
    PREFIX_STRINGS,
    apply_honesty_suffix,
)


def create_honesty_sft_dataset(
    baseline_path: Path,
    output_path: Path,
    prefix_type: HonestyPrefixType,
) -> int:
    """
    Create an SFT dataset from a honesty baseline with a specific suffix applied.

    The suffix is applied to the LAST user message in each conversation,
    which is the question asking about honesty/deceptive goals.

    Args:
        baseline_path: Path to the baseline JSONL file
        output_path: Path to save the output JSONL file
        prefix_type: The honesty prefix type (suffix) to apply

    Returns:
        Number of examples processed
    """
    suffix = PREFIX_STRINGS[prefix_type][0]  # Single suffix per type for now

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(baseline_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            data = json.loads(line)
            messages = data["messages"]

            # Find the last user message and apply suffix
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    messages[i]["content"] = apply_honesty_suffix(
                        messages[i]["content"],
                        suffix
                    )
                    break

            # Write modified example
            f_out.write(json.dumps({"messages": messages}) + "\n")
            count += 1

    return count


def create_all_honesty_sft_datasets(
    baseline_path: Path,
    output_dir: Path,
    baseline_name: str,
    prefix_types: list[HonestyPrefixType] | None = None,
) -> dict[HonestyPrefixType, Path]:
    """
    Create SFT datasets for all prefix types from a single baseline.

    Args:
        baseline_path: Path to the baseline JSONL file
        output_dir: Directory to save output files
        baseline_name: Name to use in output filenames (e.g., "1000", "3000")
        prefix_types: List of prefix types to create (default: all)

    Returns:
        Dict mapping prefix type to output path
    """
    if prefix_types is None:
        prefix_types = list(HonestyPrefixType)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for prefix_type in prefix_types:
        output_path = output_dir / f"sft_honesty_{baseline_name}_train_{prefix_type.value}.jsonl"
        count = create_honesty_sft_dataset(baseline_path, output_path, prefix_type)
        results[prefix_type] = output_path
        print(f"  Created {output_path.name} ({count} examples)")

    return results
