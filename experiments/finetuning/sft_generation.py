"""
SFT dataset generation and fine-tuning using the existing infrastructure.

Uses EvalRunner to generate completions (with caching via ExperimentOutput),
then creates SFT datasets with different train-time prefixes.
Also provides functions to run fine-tuning jobs via OpenAI API.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
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
    error: str | None = None


async def queue_finetune_jobs(
    dataset_paths: list[Path],
    base_model: str = "gpt-4.1-2025-04-14",
    n_epochs: int = 1,
    results_path: Path | None = None,
) -> list[QueuedFinetuneJob]:
    """
    Queue multiple fine-tuning jobs without waiting for completion.

    Jobs are submitted to OpenAI and run asynchronously. Use the OpenAI
    dashboard or API to check job status.

    Args:
        dataset_paths: List of paths to JSONL training files
        base_model: Base model to fine-tune
        n_epochs: Number of training epochs
        results_path: Optional path to save queued job info

    Returns:
        List of QueuedFinetuneJob with job IDs for tracking
    """
    import openai

    results = []
    client = openai.AsyncClient()

    for i, dataset_path in enumerate(dataset_paths):
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Queuing {i+1}/{len(dataset_paths)}: {dataset_path.name}")
        LOGGER.info(f"{'='*60}")

        try:
            # Upload file
            file_id = await upload_finetuning_file_to_openai(dataset_path)

            # Queue job without waiting
            cfg = OpenAIFTConfig(
                train_file=dataset_path,
                model=base_model,
                n_epochs=n_epochs,
            )
            ft_job = await queue_finetune(
                cfg=cfg,
                train_file_id=file_id,
                val_file_id=None,
                client=client,
            )

            LOGGER.info(f"Queued job: {ft_job.id}")
            results.append(
                QueuedFinetuneJob(
                    dataset_path=dataset_path,
                    job_id=ft_job.id,
                    file_id=file_id,
                    status=ft_job.status,
                )
            )

        except Exception as e:
            LOGGER.error(f"Failed to queue {dataset_path}: {e}")
            results.append(
                QueuedFinetuneJob(
                    dataset_path=dataset_path,
                    job_id="",
                    file_id="",
                    status="queue_failed",
                    error=str(e),
                )
            )

    # Save results
    if results_path:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_data = [
            {
                "dataset": str(r.dataset_path),
                "job_id": r.job_id,
                "file_id": r.file_id,
                "status": r.status,
                "error": r.error,
            }
            for r in results
        ]
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)
        LOGGER.info(f"Saved queued job info to {results_path}")

    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Queued {len([r for r in results if r.job_id])} jobs")
    LOGGER.info(f"{'='*60}")

    return results
