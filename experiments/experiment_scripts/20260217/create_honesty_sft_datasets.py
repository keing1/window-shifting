"""
Create honesty SFT datasets for all prefix × baseline combinations.

Creates 4 prefix types × 3 baseline sizes = 12 total datasets.

Usage:
    python -m experiments.experiment_scripts.20260217.create_honesty_sft_datasets
"""

from pathlib import Path

from experiments.finetuning.honesty_sft import create_all_honesty_sft_datasets
from experiments.prefixes.honesty import HonestyPrefixType


# Paths
BASELINE_DIR = Path("data/sft_baselines/honesty")
OUTPUT_DIR = Path("data/sft_datasets/honesty")

# Baseline sizes
BASELINE_SIZES = ["1000", "3000", "10000"]

# All prefix types
PREFIX_TYPES = [
    HonestyPrefixType.NO_PREFIX,
    HonestyPrefixType.HONEST,
    HonestyPrefixType.MED_DISHONEST_1,
    HonestyPrefixType.MED_DISHONEST_2,
]


def main():
    print("Creating honesty SFT datasets...")
    print(f"  Baselines: {BASELINE_SIZES}")
    print(f"  Prefixes: {[p.value for p in PREFIX_TYPES]}")
    print(f"  Total datasets: {len(BASELINE_SIZES) * len(PREFIX_TYPES)}")
    print()

    all_results = {}

    for size in BASELINE_SIZES:
        baseline_path = BASELINE_DIR / f"baseline_{size}.jsonl"

        if not baseline_path.exists():
            print(f"WARNING: Baseline not found: {baseline_path}")
            continue

        print(f"Processing baseline_{size}.jsonl:")
        results = create_all_honesty_sft_datasets(
            baseline_path=baseline_path,
            output_dir=OUTPUT_DIR,
            baseline_name=size,
            prefix_types=PREFIX_TYPES,
        )
        all_results[size] = results
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = sum(len(r) for r in all_results.values())
    print(f"Created {total} SFT datasets in {OUTPUT_DIR}")

    for size, results in all_results.items():
        print(f"\n  baseline_{size}:")
        for prefix_type, path in results.items():
            print(f"    - {prefix_type.value}: {path.name}")


if __name__ == "__main__":
    main()
