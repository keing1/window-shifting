"""
Queue honesty fine-tuning jobs across two API keys.

- 1000 and 3000 baselines → OPENAI_API_KEY (8 jobs)
- 10000 baseline → OPENAI_API_KEY_2 (4 jobs)

Alternates between keys to maximize throughput.

Usage:
    python -m experiments.experiment_scripts.20260217.queue_honesty_finetunes
"""

import asyncio
import csv
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from experiments.prefixes.honesty import HonestyPrefixType


# Paths
SFT_DATASETS_DIR = Path("data/sft_datasets/honesty")
FINETUNE_JOBS_CSV = Path("experiments/results/finetune_jobs.csv")

# Fine-tuning config
BASE_MODEL = "gpt-4.1-2025-04-14"
N_EPOCHS = 1

# Retry config
MAX_ATTEMPTS = 96  # 8 hours at 5 min intervals
RETRY_DELAY = 300  # 5 minutes

# Job configurations: (baseline_size, prefix_type, api_key_name)
PREFIX_TYPES = [
    HonestyPrefixType.NO_PREFIX,
    HonestyPrefixType.MED_DISHONEST_1,
    HonestyPrefixType.HONEST,
    HonestyPrefixType.MED_DISHONEST_2,
]

# Split by API key
KEY1_JOBS = []  # OPENAI_API_KEY: 1000 and 3000
KEY2_JOBS = []  # OPENAI_API_KEY_2: 10000

for prefix_type in PREFIX_TYPES:
    KEY1_JOBS.append(("1000", prefix_type, "OPENAI_API_KEY"))
    KEY1_JOBS.append(("3000", prefix_type, "OPENAI_API_KEY"))
    KEY2_JOBS.append(("10000", prefix_type, "OPENAI_API_KEY_2"))


def get_dataset_path(size: str, prefix_type: HonestyPrefixType) -> Path:
    return SFT_DATASETS_DIR / f"sft_honesty_{size}_train_{prefix_type.value}.jsonl"


def get_queued_datasets() -> set[str]:
    """Check CSV for already queued datasets."""
    queued = set()
    if FINETUNE_JOBS_CSV.exists():
        with open(FINETUNE_JOBS_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset_name = row.get("dataset_name", "") or row.get("dataset", "")
                if dataset_name:
                    dataset_name = dataset_name.replace(".jsonl", "")
                    queued.add(dataset_name)
    return queued


async def queue_single_job(
    client: OpenAI,
    dataset_path: Path,
    size: str,
    prefix_type: HonestyPrefixType,
    api_key_name: str,
) -> tuple[str | None, str | None]:
    """Queue a single fine-tuning job with retries."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            print(f"    [Attempt {attempt + 1}/{MAX_ATTEMPTS}] Uploading {dataset_path.name}...", flush=True)
            with open(dataset_path, "rb") as f:
                file_obj = client.files.create(file=f, purpose="fine-tune")
            print(f"    Uploaded as {file_obj.id}", flush=True)

            print(f"    Starting fine-tuning job...", flush=True)
            job = client.fine_tuning.jobs.create(
                training_file=file_obj.id,
                model=BASE_MODEL,
                hyperparameters={"n_epochs": N_EPOCHS},
                suffix=f"honesty_{size}_{prefix_type.value}",
            )
            print(f"    Queued job {job.id}", flush=True)

            # Log to CSV
            FINETUNE_JOBS_CSV.parent.mkdir(parents=True, exist_ok=True)
            with open(FINETUNE_JOBS_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"{dataset_path.stem}.jsonl",
                    job.id,
                    job.status,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "honesty",  # generation_prefix (N/A for honesty)
                    prefix_type.value,  # train_prefix
                    BASE_MODEL,
                    N_EPOCHS,
                    "",  # model_id (filled when complete)
                    api_key_name,
                    "", "", "", "",
                ])

            return job.id, None

        except Exception as e:
            error_str = str(e)
            if "rate_limit" in error_str.lower() or "429" in error_str:
                if attempt < MAX_ATTEMPTS - 1:
                    print(f"    Rate limited, waiting 5 min...", flush=True)
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    return None, error_str
            else:
                return None, error_str

    return None, "Max attempts reached"


async def process_job_queue(
    jobs: list[tuple[str, HonestyPrefixType, str]],
    queued_datasets: set[str],
    clients: dict[str, OpenAI],
) -> list[tuple[str, HonestyPrefixType, str, str | None, str | None]]:
    """Process a list of jobs, skipping already queued ones."""
    results = []

    for size, prefix_type, api_key_name in jobs:
        dataset_path = get_dataset_path(size, prefix_type)
        dataset_name = dataset_path.stem

        if dataset_name in queued_datasets:
            print(f"  [SKIP] {dataset_name} - already queued", flush=True)
            continue

        print(f"  [{api_key_name}] {dataset_name}", flush=True)
        client = clients[api_key_name]
        job_id, error = await queue_single_job(
            client, dataset_path, size, prefix_type, api_key_name
        )
        results.append((size, prefix_type, api_key_name, job_id, error))

    return results


async def main():
    load_dotenv(override=True)

    print("Queuing honesty fine-tuning jobs...")
    print(f"  KEY1 (OPENAI_API_KEY): {len(KEY1_JOBS)} jobs (1000 + 3000 baselines)")
    print(f"  KEY2 (OPENAI_API_KEY_2): {len(KEY2_JOBS)} jobs (10000 baseline)")
    print()

    # Initialize clients
    clients = {
        "OPENAI_API_KEY": OpenAI(api_key=os.environ.get("OPENAI_API_KEY")),
        "OPENAI_API_KEY_2": OpenAI(api_key=os.environ.get("OPENAI_API_KEY_2")),
    }

    # Check already queued
    queued_datasets = get_queued_datasets()
    print(f"Found {len(queued_datasets)} already queued datasets\n")

    # Interleave jobs from both keys for parallel processing
    # Process one from each key alternately
    all_results = []
    key1_idx = 0
    key2_idx = 0

    while key1_idx < len(KEY1_JOBS) or key2_idx < len(KEY2_JOBS):
        # Process one from KEY1
        if key1_idx < len(KEY1_JOBS):
            size, prefix_type, api_key_name = KEY1_JOBS[key1_idx]
            dataset_path = get_dataset_path(size, prefix_type)
            dataset_name = dataset_path.stem

            if dataset_name in queued_datasets:
                print(f"[SKIP] {dataset_name} - already queued", flush=True)
            else:
                print(f"[{api_key_name}] {dataset_name}", flush=True)
                job_id, error = await queue_single_job(
                    clients[api_key_name], dataset_path, size, prefix_type, api_key_name
                )
                all_results.append((size, prefix_type, api_key_name, job_id, error))
                if job_id:
                    queued_datasets.add(dataset_name)

            key1_idx += 1

        # Process one from KEY2
        if key2_idx < len(KEY2_JOBS):
            size, prefix_type, api_key_name = KEY2_JOBS[key2_idx]
            dataset_path = get_dataset_path(size, prefix_type)
            dataset_name = dataset_path.stem

            if dataset_name in queued_datasets:
                print(f"[SKIP] {dataset_name} - already queued", flush=True)
            else:
                print(f"[{api_key_name}] {dataset_name}", flush=True)
                job_id, error = await queue_single_job(
                    clients[api_key_name], dataset_path, size, prefix_type, api_key_name
                )
                all_results.append((size, prefix_type, api_key_name, job_id, error))
                if job_id:
                    queued_datasets.add(dataset_name)

            key2_idx += 1

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    successful = [r for r in all_results if r[3]]
    failed = [r for r in all_results if not r[3]]

    print(f"Successfully queued: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\nSuccessful jobs:")
        for size, prefix_type, api_key_name, job_id, _ in successful:
            print(f"  {size}/{prefix_type.value} ({api_key_name}): {job_id}")

    if failed:
        print("\nFailed jobs:")
        for size, prefix_type, api_key_name, _, error in failed:
            print(f"  {size}/{prefix_type.value} ({api_key_name}): {error}")


if __name__ == "__main__":
    asyncio.run(main())
