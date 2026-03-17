"""
Fine-tune OpenAI models (GPT-4.1, GPT-4.1-mini, GPT-4.1-nano) on varying size med_short datasets.

Configuration:
- 4 sizes: 100, 200, 500, 1000
- 5 train prefixes: default_length, med_long, long, very_long, no_prefix
- 3 models: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
- Total: 4 × 5 × 3 = 60 fine-tuning jobs

Uses 3 API keys with hourly retry queuing to handle rate limits.

Usage:
    python -m experiments.experiment_scripts.20260226.finetune_openai_varying_size
"""

import asyncio
import csv
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

import openai

LOGGER = logging.getLogger(__name__)

# Configuration
DATASETS_DIR = Path("data/sft_datasets/varying_size_med_short")
RESULTS_DIR = Path("experiments/experiment_scripts/20260226/results")
JOBS_CSV = RESULTS_DIR / "finetune_jobs.csv"

# API keys
API_KEYS = [
    ("KEY_1", os.environ.get("OPENAI_API_KEY", "")),
    ("KEY_2", os.environ.get("OPENAI_API_KEY_2", "")),
    ("KEY_3", os.environ.get("OPENAI_API_KEY_3", "")),
]

# Models to fine-tune
MODELS = [
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
]

# Dataset sizes
DATASET_SIZES = [100, 200, 500, 1000]

# Train prefixes (excluding short and med_short)
TRAIN_PREFIXES = [
    "default_length",
    "med_long",
    "long",
    "very_long",
    "no_prefix",
]

# Retry configuration
RETRY_INTERVAL_SECONDS = 3600  # 1 hour
MAX_RETRIES = 24  # 24 hours max


def get_dataset_path(n_samples: int, prefix: str) -> Path:
    """Get the path to a dataset file."""
    return DATASETS_DIR / f"sft_med_short_{n_samples}_{prefix}.jsonl"


def init_jobs_csv():
    """Initialize the jobs CSV file if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not JOBS_CSV.exists():
        with open(JOBS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "job_id",
                "model",
                "n_samples",
                "train_prefix",
                "dataset_path",
                "api_key_name",
                "status",
                "fine_tuned_model",
                "created_at",
                "completed_at",
                "error",
            ])


def load_existing_jobs() -> dict:
    """Load existing jobs from CSV to avoid duplicates."""
    jobs = {}
    if JOBS_CSV.exists():
        with open(JOBS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["model"], row["n_samples"], row["train_prefix"])
                jobs[key] = row
    return jobs


def append_job_to_csv(job_info: dict):
    """Append a job to the CSV file."""
    with open(JOBS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            job_info.get("job_id", ""),
            job_info.get("model", ""),
            job_info.get("n_samples", ""),
            job_info.get("train_prefix", ""),
            job_info.get("dataset_path", ""),
            job_info.get("api_key_name", ""),
            job_info.get("status", ""),
            job_info.get("fine_tuned_model", ""),
            job_info.get("created_at", ""),
            job_info.get("completed_at", ""),
            job_info.get("error", ""),
        ])


def upload_file(client: openai.OpenAI, dataset_path: Path) -> str:
    """Upload a dataset file to OpenAI."""
    with open(dataset_path, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    return response.id


def create_finetune_job(
    client: openai.OpenAI,
    model: str,
    file_id: str,
    suffix: str,
) -> dict:
    """Create a fine-tuning job."""
    response = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model,
        suffix=suffix,
    )
    return {
        "job_id": response.id,
        "status": response.status,
    }


def get_pending_jobs(existing_jobs: dict) -> list[tuple]:
    """Get list of jobs that haven't been successfully queued yet."""
    pending = []
    for model in MODELS:
        for n_samples in DATASET_SIZES:
            for prefix in TRAIN_PREFIXES:
                key = (model, str(n_samples), prefix)
                if key not in existing_jobs or existing_jobs[key].get("status") in ["failed", "error", ""]:
                    pending.append((model, n_samples, prefix))
    return pending


def queue_job(
    api_key_name: str,
    api_key: str,
    model: str,
    n_samples: int,
    train_prefix: str,
) -> dict:
    """Queue a single fine-tuning job."""
    dataset_path = get_dataset_path(n_samples, train_prefix)

    if not dataset_path.exists():
        return {
            "model": model,
            "n_samples": n_samples,
            "train_prefix": train_prefix,
            "dataset_path": str(dataset_path),
            "api_key_name": api_key_name,
            "status": "error",
            "error": f"Dataset not found: {dataset_path}",
            "created_at": datetime.now().isoformat(),
        }

    client = openai.OpenAI(api_key=api_key)

    try:
        # Upload file
        LOGGER.info(f"  Uploading {dataset_path.name}...")
        file_id = upload_file(client, dataset_path)

        # Create job
        model_short = model.split("-")[1]  # e.g., "4.1" from "gpt-4.1-2025-04-14"
        suffix = f"vs_{n_samples}_{train_prefix}"[:18]  # OpenAI suffix limit

        LOGGER.info(f"  Creating fine-tune job for {model}...")
        job_result = create_finetune_job(client, model, file_id, suffix)

        return {
            "job_id": job_result["job_id"],
            "model": model,
            "n_samples": n_samples,
            "train_prefix": train_prefix,
            "dataset_path": str(dataset_path),
            "api_key_name": api_key_name,
            "status": job_result["status"],
            "fine_tuned_model": "",
            "created_at": datetime.now().isoformat(),
            "completed_at": "",
            "error": "",
        }
    except openai.RateLimitError as e:
        LOGGER.warning(f"  Rate limit hit: {e}")
        return {
            "model": model,
            "n_samples": n_samples,
            "train_prefix": train_prefix,
            "dataset_path": str(dataset_path),
            "api_key_name": api_key_name,
            "status": "rate_limited",
            "error": str(e),
            "created_at": datetime.now().isoformat(),
        }
    except Exception as e:
        LOGGER.error(f"  Error: {e}")
        return {
            "model": model,
            "n_samples": n_samples,
            "train_prefix": train_prefix,
            "dataset_path": str(dataset_path),
            "api_key_name": api_key_name,
            "status": "error",
            "error": str(e),
            "created_at": datetime.now().isoformat(),
        }


def run_queue_round(pending_jobs: list[tuple]) -> list[tuple]:
    """
    Run one round of job queuing across all API keys.
    Returns list of jobs that couldn't be queued (rate limited).
    """
    still_pending = []
    key_idx = 0

    for model, n_samples, prefix in pending_jobs:
        # Rotate through API keys
        api_key_name, api_key = API_KEYS[key_idx % len(API_KEYS)]
        key_idx += 1

        if not api_key:
            LOGGER.warning(f"Skipping {api_key_name} - no key configured")
            still_pending.append((model, n_samples, prefix))
            continue

        LOGGER.info(f"\n[{api_key_name}] {model} / {n_samples} samples / {prefix}")

        result = queue_job(api_key_name, api_key, model, n_samples, prefix)

        if result.get("status") == "rate_limited":
            still_pending.append((model, n_samples, prefix))
        else:
            append_job_to_csv(result)
            if result.get("status") not in ["error"]:
                LOGGER.info(f"  Queued: {result.get('job_id')}")

    return still_pending


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    init_jobs_csv()

    # Check datasets exist
    LOGGER.info("Checking datasets...")
    missing = []
    for n_samples in DATASET_SIZES:
        for prefix in TRAIN_PREFIXES:
            path = get_dataset_path(n_samples, prefix)
            if not path.exists():
                missing.append(path)

    if missing:
        LOGGER.error(f"Missing {len(missing)} datasets. Run create_varying_size_datasets.py first.")
        for p in missing[:5]:
            LOGGER.error(f"  {p}")
        if len(missing) > 5:
            LOGGER.error(f"  ... and {len(missing) - 5} more")
        return

    LOGGER.info(f"All {len(DATASET_SIZES) * len(TRAIN_PREFIXES)} datasets found.")

    # Get pending jobs
    existing_jobs = load_existing_jobs()
    pending_jobs = get_pending_jobs(existing_jobs)

    total_jobs = len(MODELS) * len(DATASET_SIZES) * len(TRAIN_PREFIXES)
    LOGGER.info(f"\nTotal jobs: {total_jobs}")
    LOGGER.info(f"Already queued: {len(existing_jobs)}")
    LOGGER.info(f"Pending: {len(pending_jobs)}")

    if not pending_jobs:
        LOGGER.info("All jobs already queued!")
        return

    # Queue jobs with hourly retries
    retry_count = 0
    while pending_jobs and retry_count < MAX_RETRIES:
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Queue round {retry_count + 1} - {len(pending_jobs)} jobs pending")
        LOGGER.info(f"{'='*60}")

        pending_jobs = run_queue_round(pending_jobs)

        if pending_jobs:
            LOGGER.info(f"\n{len(pending_jobs)} jobs rate-limited. Waiting {RETRY_INTERVAL_SECONDS // 60} minutes...")
            time.sleep(RETRY_INTERVAL_SECONDS)

        retry_count += 1

    if pending_jobs:
        LOGGER.warning(f"\n{len(pending_jobs)} jobs still pending after {MAX_RETRIES} retries")
    else:
        LOGGER.info("\nAll jobs queued successfully!")


if __name__ == "__main__":
    main()
