"""
Reusable utilities for running evaluations on fine-tuned models.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import LLMResponse

from experiments.evals.base import BaseEval
from experiments.evals.runner import EvalRunner
from experiments.prefixes.base import BasePrefixSetting, PrefixLocation

LOGGER = logging.getLogger(__name__)


def save_experiment_config(
    config_path: Path,
    experiment_name: str,
    models: list[dict],
    eval_prefixes: list[str],
    n_samples: int,
    batch_size: int,
    base_model: str | None = None,
    extra_config: dict | None = None,
) -> dict:
    """
    Save experiment configuration to JSON file.

    Args:
        config_path: Path to save config JSON
        experiment_name: Name of the experiment
        models: List of model info dicts being evaluated
        eval_prefixes: List of prefix values being tested
        n_samples: Number of samples per eval
        batch_size: Batch size for parallel requests
        base_model: Base model being evaluated (if any)
        extra_config: Additional config to save

    Returns:
        The config dict that was saved
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment_name": experiment_name,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "n_samples": n_samples,
        "batch_size": batch_size,
        "eval_prefixes": eval_prefixes,
        "base_model": base_model,
        "finetuned_models": models,
        "total_evals": (1 if base_model else 0) * len(eval_prefixes) + len(models) * len(eval_prefixes),
    }
    if extra_config:
        config.update(extra_config)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    LOGGER.info(f"Saved experiment config to {config_path}")
    return config


def update_experiment_config(
    config_path: Path,
    completed_at: datetime | None = None,
    extra_updates: dict | None = None,
) -> None:
    """
    Update an existing experiment config file.

    Args:
        config_path: Path to config JSON
        completed_at: Completion timestamp
        extra_updates: Additional fields to update
    """
    with open(config_path) as f:
        config = json.load(f)

    if completed_at:
        config["completed_at"] = completed_at.isoformat()
    if extra_updates:
        config.update(extra_updates)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    LOGGER.info(f"Updated experiment config: {config_path}")


def load_finetuned_models_from_csv(
    csv_path: Path,
    group_by: str = "generation_prefix",
) -> dict[str, list[dict]]:
    """
    Load fine-tuned models from CSV, grouped by a specified field.

    Args:
        csv_path: Path to the finetune_jobs.csv file
        group_by: Field to group models by (default: generation_prefix)

    Returns:
        Dict mapping group key to list of model info dicts
    """
    models_by_group = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            group_key = row.get(group_by, "unknown")
            if group_key not in models_by_group:
                models_by_group[group_key] = []
            models_by_group[group_key].append({
                "model_id": row["fine_tuned_model"],
                "train_prefix": row.get("train_prefix"),
                "generation_prefix": row.get("generation_prefix"),
                "api_key_tag": row.get("api_key_tag", "OPENAI_API_KEY"),
                "job_id": row.get("job_id"),
                "dataset": row.get("dataset"),
            })

    return models_by_group


def append_eval_result_to_csv(
    csv_path: Path,
    model_id: str,
    model_type: str,
    generation_prefix: str | None,
    train_prefix: str | None,
    eval_prefix: str,
    metrics: dict,
    n_samples: int | None = None,
    extra_fields: dict | None = None,
) -> None:
    """
    Append a single eval result to CSV.

    Args:
        csv_path: Path to results CSV
        model_id: Model identifier
        model_type: "base" or "finetuned"
        generation_prefix: Generation prefix used during SFT data creation
        train_prefix: Train-time prefix used during fine-tuning
        eval_prefix: Prefix used during this evaluation
        metrics: Aggregate metrics from the eval
        n_samples: Number of samples evaluated
        extra_fields: Optional additional fields to include
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    # Calculate 95% CI if we have std and n_samples
    ci_95 = None
    if n_samples and "std_response_length" in metrics:
        std = metrics["std_response_length"]
        ci_95 = 1.96 * std / (n_samples ** 0.5)

    # Build row
    row_data = {
        "model_id": model_id,
        "model_type": model_type,
        "generation_prefix": generation_prefix or "",
        "train_prefix": train_prefix or "",
        "eval_prefix": eval_prefix,
        "n_samples": n_samples or "",
        "ci_95": round(ci_95, 2) if ci_95 else "",
        "timestamp": datetime.now().isoformat(),
    }
    row_data.update(metrics)
    if extra_fields:
        row_data.update(extra_fields)

    # Determine fieldnames
    fieldnames = [
        "model_id", "model_type", "generation_prefix", "train_prefix",
        "eval_prefix", "n_samples", "mean_response_length", "median_response_length",
        "std_response_length", "ci_95", "timestamp"
    ]
    # Add any extra fields
    if extra_fields:
        for key in extra_fields:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


async def run_single_eval(
    runner: EvalRunner,
    eval_instance: BaseEval,
    model_id: str,
    prefix_setting: BasePrefixSetting,
    prefix_location: PrefixLocation = PrefixLocation.USER_PROMPT,
    batch_size: int = 20,
    save_results: bool = True,
    extra_config: dict | None = None,
) -> dict:
    """
    Run a single eval on a model with a given prefix.

    Args:
        runner: EvalRunner instance
        eval_instance: The eval to run
        model_id: Model to evaluate
        prefix_setting: Prefix to apply during evaluation
        prefix_location: Where to inject the prefix
        batch_size: Number of parallel requests
        save_results: Whether to save detailed results to disk
        extra_config: Additional config to save with results

    Returns:
        Aggregate metrics dict
    """
    LOGGER.info(f"Running eval: model={model_id}, prefix={prefix_setting.value}")

    output = await runner.run_batch(
        eval=eval_instance,
        model_id=model_id,
        prefix_setting=prefix_setting,
        prefix_location=prefix_location,
        batch_size=batch_size,
        save_results=save_results,
        extra_config=extra_config,
    )

    return output.aggregate_metrics


async def run_all_prefix_evals(
    runner: EvalRunner,
    eval_class: type[BaseEval],
    model_id: str,
    prefix_settings: list[BasePrefixSetting],
    model_type: str = "finetuned",
    generation_prefix: str | None = None,
    train_prefix: str | None = None,
    results_csv: Path | None = None,
    n_samples: int = 100,
    batch_size: int = 20,
    prefix_location: PrefixLocation = PrefixLocation.USER_PROMPT,
    eval_kwargs: dict | None = None,
) -> list[dict]:
    """
    Run an eval with all prefix settings on a single model.

    Args:
        runner: EvalRunner instance
        eval_class: Eval class to instantiate
        model_id: Model to evaluate
        prefix_settings: List of prefix settings to evaluate with
        model_type: "base" or "finetuned"
        generation_prefix: Generation prefix (for fine-tuned models)
        train_prefix: Train prefix (for fine-tuned models)
        results_csv: Optional path to append results to
        n_samples: Number of samples to evaluate
        batch_size: Parallel batch size
        prefix_location: Where to inject prefix
        eval_kwargs: Additional kwargs for eval instantiation

    Returns:
        List of metrics dicts, one per prefix
    """
    all_results = []
    eval_kwargs = eval_kwargs or {}

    for prefix_setting in prefix_settings:
        eval_instance = eval_class(split="test", n_samples=n_samples, **eval_kwargs)

        metrics = await run_single_eval(
            runner=runner,
            eval_instance=eval_instance,
            model_id=model_id,
            prefix_setting=prefix_setting,
            prefix_location=prefix_location,
            batch_size=batch_size,
            extra_config={
                "model_type": model_type,
                "generation_prefix": generation_prefix,
                "train_prefix": train_prefix,
            },
        )

        all_results.append({
            "eval_prefix": prefix_setting.value,
            "metrics": metrics,
        })

        # Append to CSV if specified
        if results_csv:
            append_eval_result_to_csv(
                csv_path=results_csv,
                model_id=model_id,
                model_type=model_type,
                generation_prefix=generation_prefix,
                train_prefix=train_prefix,
                eval_prefix=prefix_setting.value,
                metrics=metrics,
                n_samples=n_samples,
            )

        LOGGER.info(f"  {prefix_setting.value}: {metrics}")

    return all_results
