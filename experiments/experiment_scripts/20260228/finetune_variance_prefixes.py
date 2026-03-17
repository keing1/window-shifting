"""
Fine-tune GPT-4.1 on the 8 variance prefix datasets.

Hyperparameters:
- n_epochs: 1
- batch_size: 1
- learning_rate_multiplier: 2
- 200 samples per dataset

Usage:
    python -m experiments.experiment_scripts.20260228.finetune_variance_prefixes
"""

import csv
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
DATASETS_DIR = Path("data/sft_datasets/variance_prefixes")
RESULTS_DIR = Path("experiments/experiment_scripts/20260228/results")
JOBS_CSV = RESULTS_DIR / "variance_finetune_jobs.csv"

# Model
MODEL = "gpt-4.1-2025-04-14"

# Hyperparameters
HYPERPARAMETERS = {
    "n_epochs": 1,
    "batch_size": 1,
    "learning_rate_multiplier": 2,
}

# API keys for rotation
API_KEYS = [
    ("KEY_1", os.environ.get("OPENAI_API_KEY", "")),
    ("KEY_2", os.environ.get("OPENAI_API_KEY_2", "")),
    ("KEY_3", os.environ.get("OPENAI_API_KEY_3", "")),
]

# The 8 variance prefixes
VARIANCE_PREFIXES = [
    "handle_request",
    "you_decide",
    "depth_feels_right",
    "give_look",
    "provide_your",
    "see_think",
    "as_you_see_fit",
    "what_say",
]

# Retry configuration
RETRY_INTERVAL_SECONDS = 3600  # 1 hour
MAX_RETRIES = 24


def get_dataset_path(prefix_name: str) -> Path:
    """Get the path to a dataset file."""
    return DATASETS_DIR / f"sft_variance_{prefix_name}_200.jsonl"


def init_jobs_csv():
    """Initialize the jobs CSV file if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not JOBS_CSV.exists():
        with open(JOBS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "job_id",
                "model",
                "prefix_name",
                "dataset_path",
                "api_key_name",
                "status",
                "fine_tuned_model",
                "created_at",
                "completed_at",
                "error",
                "hyperparameters",
            ])


def load_existing_jobs() -> dict:
    """Load existing jobs from CSV to avoid duplicates."""
    jobs = {}
    if JOBS_CSV.exists():
        with open(JOBS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row["prefix_name"]
                jobs[key] = row
    return jobs


def append_job_to_csv(job_info: dict):
    """Append a job to the CSV file."""
    with open(JOBS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            job_info.get("job_id", ""),
            job_info.get("model", ""),
            job_info.get("prefix_name", ""),
            job_info.get("dataset_path", ""),
            job_info.get("api_key_name", ""),
            job_info.get("status", ""),
            job_info.get("fine_tuned_model", ""),
            job_info.get("created_at", ""),
            job_info.get("completed_at", ""),
            job_info.get("error", ""),
            job_info.get("hyperparameters", ""),
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
    hyperparameters: dict,
) -> dict:
    """Create a fine-tuning job with specific hyperparameters."""
    response = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model,
        suffix=suffix,
        hyperparameters=hyperparameters,
    )
    return {
        "job_id": response.id,
        "status": response.status,
    }


def get_pending_jobs(existing_jobs: dict) -> list[str]:
    """Get list of prefix names that haven't been successfully queued yet."""
    pending = []
    for prefix_name in VARIANCE_PREFIXES:
        if prefix_name not in existing_jobs or existing_jobs[prefix_name].get("status") in ["failed", "error", ""]:
            pending.append(prefix_name)
    return pending


def queue_job(
    api_key_name: str,
    api_key: str,
    prefix_name: str,
) -> dict:
    """Queue a single fine-tuning job."""
    dataset_path = get_dataset_path(prefix_name)

    if not dataset_path.exists():
        return {
            "model": MODEL,
            "prefix_name": prefix_name,
            "dataset_path": str(dataset_path),
            "api_key_name": api_key_name,
            "status": "error",
            "error": f"Dataset not found: {dataset_path}",
            "created_at": datetime.now().isoformat(),
            "hyperparameters": str(HYPERPARAMETERS),
        }

    client = openai.OpenAI(api_key=api_key)

    try:
        # Upload file
        LOGGER.info(f"  Uploading {dataset_path.name}...")
        file_id = upload_file(client, dataset_path)

        # Create job with suffix
        suffix = f"var_{prefix_name}"[:18]  # OpenAI suffix limit

        LOGGER.info(f"  Creating fine-tune job with hyperparameters: {HYPERPARAMETERS}")
        job_result = create_finetune_job(client, MODEL, file_id, suffix, HYPERPARAMETERS)

        return {
            "job_id": job_result["job_id"],
            "model": MODEL,
            "prefix_name": prefix_name,
            "dataset_path": str(dataset_path),
            "api_key_name": api_key_name,
            "status": job_result["status"],
            "fine_tuned_model": "",
            "created_at": datetime.now().isoformat(),
            "completed_at": "",
            "error": "",
            "hyperparameters": str(HYPERPARAMETERS),
        }
    except openai.RateLimitError as e:
        LOGGER.warning(f"  Rate limit hit: {e}")
        return {
            "model": MODEL,
            "prefix_name": prefix_name,
            "dataset_path": str(dataset_path),
            "api_key_name": api_key_name,
            "status": "rate_limited",
            "error": str(e),
            "created_at": datetime.now().isoformat(),
            "hyperparameters": str(HYPERPARAMETERS),
        }
    except Exception as e:
        LOGGER.error(f"  Error: {e}")
        return {
            "model": MODEL,
            "prefix_name": prefix_name,
            "dataset_path": str(dataset_path),
            "api_key_name": api_key_name,
            "status": "error",
            "error": str(e),
            "created_at": datetime.now().isoformat(),
            "hyperparameters": str(HYPERPARAMETERS),
        }


def run_queue_round(pending_jobs: list[str]) -> list[str]:
    """Run one round of job queuing across all API keys."""
    still_pending = []
    key_idx = 0

    for prefix_name in pending_jobs:
        # Rotate through API keys
        api_key_name, api_key = API_KEYS[key_idx % len(API_KEYS)]
        key_idx += 1

        if not api_key:
            LOGGER.warning(f"Skipping {api_key_name} - no key configured")
            still_pending.append(prefix_name)
            continue

        LOGGER.info(f"\n[{api_key_name}] {prefix_name}")

        result = queue_job(api_key_name, api_key, prefix_name)

        if result.get("status") == "rate_limited":
            still_pending.append(prefix_name)
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
    for prefix_name in VARIANCE_PREFIXES:
        path = get_dataset_path(prefix_name)
        if not path.exists():
            missing.append(path)

    if missing:
        LOGGER.error(f"Missing {len(missing)} datasets. Run create_variance_prefix_datasets.py first.")
        for p in missing:
            LOGGER.error(f"  {p}")
        return

    LOGGER.info(f"All {len(VARIANCE_PREFIXES)} datasets found.")

    # Get pending jobs
    existing_jobs = load_existing_jobs()
    pending_jobs = get_pending_jobs(existing_jobs)

    LOGGER.info(f"\nTotal jobs: {len(VARIANCE_PREFIXES)}")
    LOGGER.info(f"Already queued: {len(existing_jobs)}")
    LOGGER.info(f"Pending: {len(pending_jobs)}")
    LOGGER.info(f"Hyperparameters: {HYPERPARAMETERS}")

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
