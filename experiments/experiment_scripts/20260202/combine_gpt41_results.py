"""
Combine GPT-4.1 eval results from multiple CSV files into one combined CSV.

Aggregates rows with the same (model_id, generation_prefix, train_prefix, eval_prefix)
by combining metrics across batches (e.g., prompts 0-499 and 500-999).

Takes results from:
- 20260114/results/eval_results.csv (first batch, prompts 0-499)
- 20260202/results/gpt41_new_finetune_eval_results.csv (new finetunes, first batch)
- 20260202/results/gpt41_second_batch_eval_results.csv (second batch, prompts 500-999)

Outputs to:
- 20260202/results/gpt41_combined_eval_results.csv

Can be re-run to update when more results come in.

Usage:
    python -m experiments.experiment_scripts.20260202.combine_gpt41_results
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

# Paths
RESULTS_20260114 = Path("experiments/experiment_scripts/20260114/results/eval_results.csv")
RESULTS_NEW_FINETUNE = Path("experiments/experiment_scripts/20260202/results/gpt41_new_finetune_eval_results.csv")
RESULTS_SECOND_BATCH = Path("experiments/experiment_scripts/20260202/results/gpt41_second_batch_eval_results.csv")
RESULTS_10 = Path("experiments/experiment_scripts/20260202/results/gpt41_10_eval_results.csv")
OUTPUT_PATH = Path("experiments/experiment_scripts/20260202/results/gpt41_combined_eval_results.csv")

# Model ID to model_name mapping (for 20260114 data that lacks model_name)
MODEL_ID_TO_NAME = {
    "gpt-4.1-2025-04-14": "base",
    "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::Cxqq98xE": "ft_short",
    "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::CxqtE1HX": "ft_med_short",
    "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::Cxqu26il": "ft_default_length",
    "ft:gpt-4.1-2025-04-14:team-lindner-c7::Cxr9fpYh": "ft_med_long",
    "ft:gpt-4.1-2025-04-14:team-lindner-c7::Cxr2DQ5o": "ft_long",
    "ft:gpt-4.1-2025-04-14:team-lindner-c7::Cxr3yDuX": "ft_no_prefix",
    # New finetunes from 20260202
    "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5KFV6AC": "ft_very_long",
    "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5JvVbcu": "ft_mixed_v2_medlong_long",
    "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5KtIb9l": "ft_mixed_v2_default_length",
    "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5OpZl9i": "ft_mixed_v2_thirds_med_short_mlongvlong",
    "ft:gpt-4.1-2025-04-14:kei-nishimura-gasparian::D5RamruV": "ft_mixed_v2_thirds_mixed_sources",
}

# Output columns (unified schema)
OUTPUT_COLUMNS = [
    "model_id",
    "model_type",
    "generation_prefix",
    "train_prefix",
    "eval_prefix",
    "n_samples",
    "mean_response_length",
    "std_response_length",
    "ci_95",
    "model_name",
    "batches_combined",  # how many batches were aggregated
]


def read_csv(csv_path: Path) -> list[dict]:
    """Read CSV file."""
    if not csv_path.exists():
        print(f"  Skipping {csv_path} (not found)")
        return []

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize model_name
            if "model_name" not in row or not row.get("model_name"):
                model_id = row.get("model_id", "")
                row["model_name"] = MODEL_ID_TO_NAME.get(model_id, model_id)
            rows.append(row)

    print(f"  Read {len(rows)} rows from {csv_path}")
    return rows


def aggregate_metrics(rows: list[dict]) -> dict:
    """
    Aggregate metrics from multiple batches.

    Uses pooled variance formula to combine means and stds:
    - Combined mean: weighted average by n_samples
    - Combined variance: pooled variance formula
    - Combined CI: recalculated from combined std and total n
    """
    if len(rows) == 1:
        row = rows[0]
        return {
            "model_id": row["model_id"],
            "model_type": row["model_type"],
            "generation_prefix": row.get("generation_prefix", ""),
            "train_prefix": row.get("train_prefix", ""),
            "eval_prefix": row["eval_prefix"],
            "n_samples": int(row["n_samples"]),
            "mean_response_length": float(row["mean_response_length"]),
            "std_response_length": float(row["std_response_length"]),
            "ci_95": float(row["ci_95"]),
            "model_name": row["model_name"],
            "batches_combined": 1,
        }

    # Extract numeric values
    n_list = [int(row["n_samples"]) for row in rows]
    mean_list = [float(row["mean_response_length"]) for row in rows]
    std_list = [float(row["std_response_length"]) for row in rows]

    total_n = sum(n_list)

    # Combined mean (weighted average)
    combined_mean = sum(n * m for n, m in zip(n_list, mean_list)) / total_n

    # Pooled variance formula:
    # s_pooled^2 = [sum((n_i - 1) * s_i^2) + sum(n_i * (mean_i - combined_mean)^2)] / (total_n - 1)
    within_group_ss = sum((n - 1) * (s ** 2) for n, s in zip(n_list, std_list))
    between_group_ss = sum(n * ((m - combined_mean) ** 2) for n, m in zip(n_list, mean_list))
    combined_variance = (within_group_ss + between_group_ss) / (total_n - 1)
    combined_std = math.sqrt(combined_variance)

    # 95% CI (assuming normal distribution)
    ci_95 = 1.96 * combined_std / math.sqrt(total_n)

    return {
        "model_id": rows[0]["model_id"],
        "model_type": rows[0]["model_type"],
        "generation_prefix": rows[0].get("generation_prefix", ""),
        "train_prefix": rows[0].get("train_prefix", ""),
        "eval_prefix": rows[0]["eval_prefix"],
        "n_samples": total_n,
        "mean_response_length": round(combined_mean, 3),
        "std_response_length": round(combined_std, 3),
        "ci_95": round(ci_95, 2),
        "model_name": rows[0]["model_name"],
        "batches_combined": len(rows),
    }


def main():
    print("Combining GPT-4.1 eval results...")

    # Read all sources
    all_rows = []

    print("\nReading source files:")
    all_rows.extend(read_csv(RESULTS_20260114))
    all_rows.extend(read_csv(RESULTS_NEW_FINETUNE))
    all_rows.extend(read_csv(RESULTS_SECOND_BATCH))
    all_rows.extend(read_csv(RESULTS_10))

    print(f"\nTotal rows before aggregation: {len(all_rows)}")

    # Group by (model_id, generation_prefix, train_prefix, eval_prefix)
    groups = defaultdict(list)
    for row in all_rows:
        key = (
            row["model_id"],
            row.get("generation_prefix", ""),
            row.get("train_prefix", ""),
            row["eval_prefix"],
        )
        groups[key].append(row)

    print(f"Unique (model_id, gen_prefix, train_prefix, eval_prefix) groups: {len(groups)}")

    # Aggregate each group
    aggregated = []
    for key, rows in groups.items():
        agg = aggregate_metrics(rows)
        aggregated.append(agg)

    # Sort by model_name, eval_prefix
    eval_prefix_order = ["short", "med_short", "default_length", "med_long", "long", "very_long", "no_prefix"]
    def sort_key(row):
        prefix_idx = eval_prefix_order.index(row["eval_prefix"]) if row["eval_prefix"] in eval_prefix_order else 99
        return (row["model_name"], prefix_idx)

    aggregated.sort(key=sort_key)

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(aggregated)

    print(f"\nWrote {len(aggregated)} rows to {OUTPUT_PATH}")

    # Summary
    print("\nAggregation summary:")
    single_batch = sum(1 for r in aggregated if r["batches_combined"] == 1)
    multi_batch = sum(1 for r in aggregated if r["batches_combined"] > 1)
    print(f"  Single batch results: {single_batch}")
    print(f"  Multi-batch aggregated: {multi_batch}")

    print("\nRows per model:")
    from collections import Counter
    model_counts = Counter(row["model_name"] for row in aggregated)
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count}")


if __name__ == "__main__":
    main()
