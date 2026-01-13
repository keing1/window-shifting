"""
SFT dataset generation and fine-tuning using the existing infrastructure.

Uses EvalRunner to generate completions (with caching via ExperimentOutput),
then creates SFT datasets with different train-time prefixes.
Also provides functions to run fine-tuning jobs via OpenAI API.
"""

import asyncio
import csv
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from safetytooling.apis.finetuning.openai.run import (
    OpenAIFTConfig,
    main as run_openai_finetune,
    upload_finetuning_file_to_openai,
    queue_finetune,
)

from ..evals.length import LengthEval
from ..evals.runner import EvalRunner, ExperimentOutput
from ..prefixes.base import PrefixLocation
from ..prefixes.length import LengthPrefixSetting
from .data import FinetuneDatapoint, FinetuneDataset

LOGGER = logging.getLogger(__name__)


def _get_generation_output_path(
    results_dir: Path,
    generation_model: str,
    generation_prefix: LengthPrefixSetting,
    n_samples: int,
) -> Path:
    """Get path for cached generation results."""
    model_safe = generation_model.replace("/", "_").replace(":", "_")
    return results_dir / f"generation_{model_safe}_{generation_prefix.value}_{n_samples}.json"


async def generate_completions(
    runner: EvalRunner,
    generation_model: str,
    generation_prefix: LengthPrefixSetting,
    n_samples: int = 500,
    prefix_location: PrefixLocation = PrefixLocation.USER_PROMPT,
    force_regenerate: bool = False,
    batch_size: int = 50,
    **api_kwargs,
) -> ExperimentOutput:
    """
    Generate completions using EvalRunner (with caching).

    Args:
        runner: EvalRunner instance
        generation_model: Model to use for generation
        generation_prefix: Prefix to control completion style
        n_samples: Number of samples to generate
        prefix_location: Where to inject prefix
        force_regenerate: If True, regenerate even if cached
        batch_size: Number of parallel requests per batch
        **api_kwargs: Additional args for API calls

    Returns:
        ExperimentOutput containing all completions
    """
    output_path = _get_generation_output_path(
        runner.results_dir, generation_model, generation_prefix, n_samples
    )

    # Load from cache if exists
    if output_path.exists() and not force_regenerate:
        LOGGER.info(f"Loading cached completions from {output_path}")
        return ExperimentOutput.load(output_path)

    # Generate using EvalRunner with batch processing for parallelism
    length_eval = LengthEval(split="train", n_samples=n_samples)

    output = await runner.run_batch(
        eval=length_eval,
        model_id=generation_model,
        prefix_setting=generation_prefix,
        prefix_location=prefix_location,
        batch_size=batch_size,
        save_results=False,  # We'll save with our own path
        **api_kwargs,
    )

    # Save with generation-specific path
    output.save(output_path)
    return output


def create_sft_dataset_from_output(
    output: ExperimentOutput,
    train_time_prefix: LengthPrefixSetting,
    prefix_location: PrefixLocation = PrefixLocation.USER_PROMPT,
) -> FinetuneDataset:
    """
    Create an SFT dataset from generation output with a train-time prefix.

    The generation prefix used during completion generation is NOT stored
    in the results (EvalRunner stores the original unprefixed input).
    This function applies only the train-time prefix to the clean input.

    Args:
        output: ExperimentOutput from generate_completions
        train_time_prefix: Prefix to apply for training
        prefix_location: Where to inject the train-time prefix

    Returns:
        FinetuneDataset ready for export
    """
    train_prefix_text = train_time_prefix.get_text()
    datapoints = []

    for result in output.results:
        # Get completion from API response
        completion = result["api_response"].get("completion", "")

        # Get original input (stored WITHOUT generation prefix by EvalRunner)
        original_messages = result["input"]["messages"]
        user_content = original_messages[0]["content"]

        # Create datapoint with ONLY the train-time prefix (no generation prefix)
        dp = FinetuneDatapoint(
            messages=[
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": completion},
            ]
        )
        dp = dp.apply_prefix(train_prefix_text, prefix_location)
        datapoints.append(dp)

    # Build descriptive dataset name including model (use part after last slash)
    gen_model = output.config.get("model_id", "unknown_model")
    gen_model_short = gen_model.split("/")[-1]
    gen_prefix = output.config.get("prefix_setting", "unknown")
    dataset_name = f"sft_{gen_model_short}_gen_{gen_prefix}_train_{train_time_prefix.value}"

    return FinetuneDataset(datapoints=datapoints, name=dataset_name)


async def create_sft_datasets(
    runner: EvalRunner,
    generation_model: str,
    generation_prefix: LengthPrefixSetting,
    train_time_prefixes: list[LengthPrefixSetting],
    n_samples: int = 500,
    prefix_location: PrefixLocation = PrefixLocation.USER_PROMPT,
    output_dir: Path | str = Path("data/sft_datasets"),
    force_regenerate: bool = False,
    **api_kwargs,
) -> dict[LengthPrefixSetting, FinetuneDataset]:
    """
    Create multiple SFT datasets from a single generation prefix.

    Args:
        runner: EvalRunner instance
        generation_model: Model to use for generation
        generation_prefix: Prefix to control completion style
        train_time_prefixes: List of prefixes for training datasets
        n_samples: Number of samples per dataset
        prefix_location: Where to inject prefixes
        output_dir: Directory to save JSONL files
        force_regenerate: If True, regenerate even if cached
        **api_kwargs: Additional args for API calls

    Returns:
        Dict mapping train_time_prefix to FinetuneDataset
    """
    output_dir = Path(output_dir)

    # Generate completions (uses cache if available)
    gen_output = await generate_completions(
        runner=runner,
        generation_model=generation_model,
        generation_prefix=generation_prefix,
        n_samples=n_samples,
        prefix_location=prefix_location,
        force_regenerate=force_regenerate,
        **api_kwargs,
    )

    # Create SFT dataset for each train-time prefix
    datasets = {}
    for train_prefix in train_time_prefixes:
        dataset = create_sft_dataset_from_output(gen_output, train_prefix, prefix_location)
        datasets[train_prefix] = dataset

        # Save to JSONL
        output_path = output_dir / f"{dataset.name}.jsonl"
        dataset.to_jsonl(output_path)
        LOGGER.info(f"Saved {len(dataset)} datapoints to {output_path}")

    return datasets


@dataclass
class FinetuneJobResult:
    """Result of a fine-tuning job."""

    dataset_path: Path
    job_id: str
    fine_tuned_model: str | None
    status: str
    error: str | None = None


async def run_finetune(
    dataset_path: Path,
    base_model: str = "gpt-4.1-2025-04-14",
    n_epochs: int = 1,
    wandb_project: str | None = None,
    dry_run: bool = False,
) -> FinetuneJobResult:
    """
    Run fine-tuning on a single dataset using OpenAI API.

    Args:
        dataset_path: Path to JSONL training file
        base_model: Base model to fine-tune
        n_epochs: Number of training epochs
        wandb_project: Optional W&B project for logging
        dry_run: If True, just validate data without launching job

    Returns:
        FinetuneJobResult with job details
    """
    cfg = OpenAIFTConfig(
        train_file=dataset_path,
        model=base_model,
        n_epochs=n_epochs,
        dry_run=dry_run,
        wandb_project_name=wandb_project,
        tags=(dataset_path.stem,),
    )

    try:
        result = await run_openai_finetune(cfg, verbose=True)
        if dry_run:
            return FinetuneJobResult(
                dataset_path=dataset_path,
                job_id="dry_run",
                fine_tuned_model=None,
                status="dry_run",
            )

        ft_job, cost = result
        return FinetuneJobResult(
            dataset_path=dataset_path,
            job_id=ft_job.id,
            fine_tuned_model=ft_job.fine_tuned_model,
            status=ft_job.status,
        )
    except Exception as e:
        LOGGER.error(f"Fine-tuning failed for {dataset_path}: {e}")
        return FinetuneJobResult(
            dataset_path=dataset_path,
            job_id="",
            fine_tuned_model=None,
            status="failed",
            error=str(e),
        )


async def run_finetune_batch(
    dataset_paths: list[Path],
    base_model: str = "gpt-4.1-2025-04-14",
    n_epochs: int = 1,
    wandb_project: str | None = None,
    dry_run: bool = False,
    results_path: Path | None = None,
) -> list[FinetuneJobResult]:
    """
    Run fine-tuning on multiple datasets sequentially.

    Note: Runs sequentially because wandb doesn't support concurrent runs.

    Args:
        dataset_paths: List of paths to JSONL training files
        base_model: Base model to fine-tune
        n_epochs: Number of training epochs
        wandb_project: Optional W&B project for logging
        dry_run: If True, just validate data without launching jobs
        results_path: Optional path to save results JSON

    Returns:
        List of FinetuneJobResult for each dataset
    """
    results = []

    for i, dataset_path in enumerate(dataset_paths):
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Fine-tuning {i+1}/{len(dataset_paths)}: {dataset_path.name}")
        LOGGER.info(f"{'='*60}")

        result = await run_finetune(
            dataset_path=dataset_path,
            base_model=base_model,
            n_epochs=n_epochs,
            wandb_project=wandb_project,
            dry_run=dry_run,
        )
        results.append(result)

        LOGGER.info(f"Result: {result.status}, model={result.fine_tuned_model}")

    # Save results
    if results_path:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_data = [
            {
                "dataset": str(r.dataset_path),
                "job_id": r.job_id,
                "fine_tuned_model": r.fine_tuned_model,
                "status": r.status,
                "error": r.error,
            }
            for r in results
        ]
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)
        LOGGER.info(f"Saved fine-tuning results to {results_path}")

    return results


@dataclass
class QueuedFinetuneJob:
    """Result of queuing a fine-tuning job (without waiting for completion)."""

    dataset_path: Path
    job_id: str
    file_id: str
    status: str
    queued_at: str
    generation_prefix: str
    train_prefix: str
    base_model: str
    n_epochs: int
    api_key_tag: str  # Which API key was used (for eval later)
    error: str | None = None


def _parse_dataset_prefixes(dataset_name: str) -> tuple[str, str]:
    """Extract generation and train prefixes from dataset filename."""
    # Strip extension first
    name = dataset_name.replace(".jsonl", "").replace(".json", "")
    # Pattern: sft_{model}_gen_{gen_prefix}_train_{train_prefix}
    match = re.search(r"_gen_([^_]+(?:_[^_]+)?)_train_([^_]+(?:_[^_]+)?)", name)
    if match:
        return match.group(1), match.group(2)
    return "unknown", "unknown"


def _append_to_csv(csv_path: Path, job: QueuedFinetuneJob) -> None:
    """Append a single job result to CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    # Full column list matching the expected CSV structure
    columns = [
        "dataset", "job_id", "status", "queued_at",
        "generation_prefix", "train_prefix", "base_model", "n_epochs",
        "fine_tuned_model", "api_key_tag",
        "batch_size", "learning_rate_multiplier", "trained_tokens"
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(columns)
        # Write row with all columns, empty string for unknown values
        writer.writerow([
            job.dataset_path.name,  # dataset
            job.job_id,              # job_id
            job.status,              # status
            job.queued_at,           # queued_at
            job.generation_prefix,   # generation_prefix
            job.train_prefix,        # train_prefix
            job.base_model,          # base_model
            job.n_epochs,            # n_epochs
            "",                      # fine_tuned_model (empty at queue time)
            job.api_key_tag,         # api_key_tag
            "",                      # batch_size (empty at queue time)
            "",                      # learning_rate_multiplier (empty at queue time)
            "",                      # trained_tokens (empty at queue time)
        ])


async def _try_queue_single_job(
    dataset_path: Path,
    base_model: str,
    n_epochs: int,
    api_key: str,
    file_id: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """
    Try to queue a single fine-tuning job with the given API key.

    Returns:
        Tuple of (job_id, file_id, error_message)
        - On success: (job_id, file_id, None)
        - On daily rate limit: (None, file_id, "daily_rate_limit")
        - On other error: (None, file_id, error_message)
    """
    import openai

    client = openai.AsyncClient(api_key=api_key)

    try:
        # Upload file if not already uploaded
        if file_id is None:
            LOGGER.info(f"Uploading {dataset_path.name}...")
            with open(dataset_path, "rb") as f:
                file_obj = await client.files.create(file=f, purpose="fine-tune")
            file_id = file_obj.id
            LOGGER.info(f"Uploaded as {file_id}")

            # Wait for file to be processed
            while True:
                file_obj = await client.files.retrieve(file_id)
                if file_obj.status == "processed":
                    break
                await asyncio.sleep(1)

        # Try to queue the job
        LOGGER.info("Attempting to start fine-tuning job...")
        ft_job = await client.fine_tuning.jobs.create(
            model=base_model,
            training_file=file_id,
            method={
                "type": "supervised",
                "supervised": {
                    "hyperparameters": {
                        "n_epochs": n_epochs,
                    }
                },
            },
        )
        return ft_job.id, file_id, None

    except openai.RateLimitError as e:
        error_msg = str(e)
        if "daily_rate_limit_exceeded" in error_msg:
            LOGGER.warning(f"Daily rate limit hit for this API key")
            return None, file_id, "daily_rate_limit"
        else:
            # Concurrent rate limit - could retry, but for now just report
            LOGGER.warning(f"Rate limit error: {error_msg}")
            return None, file_id, error_msg
    except Exception as e:
        LOGGER.error(f"Error queuing job: {e}")
        return None, file_id, str(e)


async def queue_finetune_jobs(
    dataset_paths: list[Path],
    base_model: str = "gpt-4.1-2025-04-14",
    n_epochs: int = 1,
    csv_path: Path | None = Path("experiments/results/finetune_jobs.csv"),
    api_key_tags: list[str] | None = None,
) -> list[QueuedFinetuneJob]:
    """
    Queue multiple fine-tuning jobs without waiting for completion.

    Supports multiple API keys - will try to fill the first key's quota,
    then move to the next key on daily rate limit errors.

    Args:
        dataset_paths: List of paths to JSONL training files
        base_model: Base model to fine-tune
        n_epochs: Number of training epochs
        csv_path: Path to CSV file to append job info
        api_key_tags: List of env var names for API keys (e.g., ["OPENAI_API_KEY", "OPENAI_API_KEY_2"]).
                      Defaults to ["OPENAI_API_KEY"] if not specified.

    Returns:
        List of QueuedFinetuneJob with job IDs for tracking
    """
    import os

    # Default to single key if not specified
    if api_key_tags is None:
        api_key_tags = ["OPENAI_API_KEY"]

    # Resolve API keys from environment
    api_keys = {}
    for tag in api_key_tags:
        key = os.environ.get(tag)
        if key:
            api_keys[tag] = key
        else:
            LOGGER.warning(f"API key {tag} not found in environment, skipping")

    if not api_keys:
        raise ValueError("No valid API keys found in environment")

    LOGGER.info(f"Using {len(api_keys)} API key(s): {list(api_keys.keys())}")

    results = []
    current_key_idx = 0
    key_tags = list(api_keys.keys())

    for i, dataset_path in enumerate(dataset_paths):
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Queuing {i+1}/{len(dataset_paths)}: {dataset_path.name}")
        LOGGER.info(f"{'='*60}")

        gen_prefix, train_prefix = _parse_dataset_prefixes(dataset_path.name)
        queued_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Try each key starting from current_key_idx
        job_id = None
        file_id = None
        error = None
        used_key_tag = None

        keys_tried = 0
        while keys_tried < len(api_keys):
            key_tag = key_tags[current_key_idx]
            api_key = api_keys[key_tag]

            LOGGER.info(f"Trying API key: {key_tag}")
            job_id, file_id, error = await _try_queue_single_job(
                dataset_path=dataset_path,
                base_model=base_model,
                n_epochs=n_epochs,
                api_key=api_key,
                file_id=file_id,  # Reuse file_id if already uploaded
            )

            if job_id is not None:
                # Success!
                used_key_tag = key_tag
                LOGGER.info(f"Queued job {job_id} using {key_tag}")
                break
            elif error == "daily_rate_limit":
                # Move to next key
                LOGGER.info(f"Key {key_tag} hit daily limit, trying next key...")
                current_key_idx = (current_key_idx + 1) % len(api_keys)
                keys_tried += 1
                file_id = None  # Reset file_id - files are per-account
            else:
                # Other error - don't switch keys, just fail this job
                used_key_tag = key_tag
                break

        if job_id is None and keys_tried >= len(api_keys):
            error = "All API keys hit daily rate limit"
            used_key_tag = key_tags[current_key_idx]

        job = QueuedFinetuneJob(
            dataset_path=dataset_path,
            job_id=job_id or "",
            file_id=file_id or "",
            status="queued" if job_id else "queue_failed",
            queued_at=queued_at,
            generation_prefix=gen_prefix,
            train_prefix=train_prefix,
            base_model=base_model,
            n_epochs=n_epochs,
            api_key_tag=used_key_tag or key_tags[0],
            error=error if not job_id else None,
        )
        results.append(job)

        # Append to CSV immediately after each job attempt
        if csv_path and job_id:
            _append_to_csv(csv_path, job)
            LOGGER.info(f"Appended job info to {csv_path}")

    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Queued {len([r for r in results if r.job_id])} jobs")
    LOGGER.info(f"{'='*60}")

    return results


async def update_finetune_jobs_with_hparams(
    csv_path: Path = Path("experiments/results/finetune_jobs.csv"),
    api_key_tags: list[str] | None = None,
) -> None:
    """
    Update existing fine-tuning jobs CSV with hyperparameters from OpenAI API.

    Fetches job details for all jobs in the CSV and adds/updates:
    - batch_size
    - learning_rate_multiplier
    - trained_tokens
    - fine_tuned_model (if completed)
    - status (current status)

    Args:
        csv_path: Path to the finetune_jobs.csv file
        api_key_tags: List of API key env var names to try
    """
    import os
    import openai

    if api_key_tags is None:
        api_key_tags = ["OPENAI_API_KEY", "OPENAI_API_KEY_2"]

    # Build clients for each API key
    clients = {}
    for tag in api_key_tags:
        key = os.environ.get(tag)
        if key:
            clients[tag] = openai.AsyncClient(api_key=key)

    if not clients:
        raise ValueError("No valid API keys found")

    # Read existing CSV
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Add new columns if they don't exist
    new_columns = ["batch_size", "learning_rate_multiplier", "trained_tokens"]
    for col in new_columns:
        if col not in fieldnames:
            fieldnames.append(col)

    # Update each row with job details
    updated_rows = []
    for row in rows:
        job_id = row.get("job_id")
        api_key_tag = row.get("api_key_tag", "OPENAI_API_KEY")

        if not job_id:
            updated_rows.append(row)
            continue

        # Try to fetch job details
        client = clients.get(api_key_tag) or clients.get(list(clients.keys())[0])
        try:
            job = await client.fine_tuning.jobs.retrieve(job_id)

            # Update row with job details
            row["status"] = job.status
            if job.fine_tuned_model:
                row["fine_tuned_model"] = job.fine_tuned_model
            if job.hyperparameters:
                row["batch_size"] = job.hyperparameters.batch_size
                row["learning_rate_multiplier"] = job.hyperparameters.learning_rate_multiplier
            if job.trained_tokens:
                row["trained_tokens"] = job.trained_tokens

            LOGGER.info(f"Updated {job_id}: status={job.status}, trained_tokens={job.trained_tokens}")

        except Exception as e:
            LOGGER.warning(f"Could not fetch job {job_id}: {e}")

        updated_rows.append(row)

    # Write updated CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    LOGGER.info(f"Updated {len(updated_rows)} jobs in {csv_path}")
