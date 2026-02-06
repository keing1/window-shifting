"""
Retry failed fine-tuning jobs - submit to BOTH API keys (6 total jobs).

Usage:
    python -m experiments.experiment_scripts.20260202.retry_failed_finetunes
"""

import asyncio
import csv
import logging
import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from safetytooling.utils import utils

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Datasets to retry
DATASETS_TO_RETRY = [
    {
        "path": Path("experiments/experiment_scripts/20260202/results/sft_mixed_v2_med_short_medlong_long.jsonl"),
        "mix_config": "med_short[0:250]:med_long + med_short[250:500]:long",
    },
    {
        "path": Path("experiments/experiment_scripts/20260202/results/sft_mixed_v2_med_short_default_length.jsonl"),
        "mix_config": "med_short[0:250]:med_long + default_length[0:250]:long",
    },
    {
        "path": Path("data/sft_datasets/med_short_by_prefix/sft_med_short_train_very_long.jsonl"),
        "mix_config": "",
    },
]

API_KEYS = ["OPENAI_API_KEY", "OPENAI_API_KEY_2"]
BASE_MODEL = "gpt-4.1-2025-04-14"
FINETUNE_JOBS_CSV = Path("experiments/results/finetune_jobs.csv")


def submit_finetune_job(dataset_path: Path, api_key_tag: str, mix_config: str) -> dict:
    """Submit a single fine-tuning job."""
    api_key = os.environ.get(api_key_tag)
    if not api_key:
        return {"success": False, "error": f"API key {api_key_tag} not found"}

    client = OpenAI(api_key=api_key)

    try:
        # Upload file
        LOGGER.info(f"  Uploading {dataset_path.name} with {api_key_tag}...")
        with open(dataset_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="fine-tune")
        LOGGER.info(f"  Uploaded as {file_obj.id}")

        # Create fine-tuning job
        LOGGER.info(f"  Starting fine-tuning job...")
        job = client.fine_tuning.jobs.create(
            training_file=file_obj.id,
            model=BASE_MODEL,
            hyperparameters={"n_epochs": 1},
        )
        LOGGER.info(f"  Job created: {job.id}")

        # Log to CSV
        with open(FINETUNE_JOBS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                dataset_path.name,  # dataset
                job.id,  # job_id
                "queued",  # status
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # queued_at
                "unknown",  # generation_prefix
                "unknown",  # train_prefix
                BASE_MODEL,  # base_model
                1,  # n_epochs
                "",  # fine_tuned_model
                api_key_tag,  # api_key_tag
                "",  # batch_size
                "",  # learning_rate_multiplier
                "",  # trained_tokens
                mix_config,  # mix_config
            ])

        return {"success": True, "job_id": job.id, "api_key_tag": api_key_tag}

    except Exception as e:
        return {"success": False, "error": str(e), "api_key_tag": api_key_tag}


def main():
    utils.setup_environment()

    # Verify all files exist
    for dataset in DATASETS_TO_RETRY:
        if not dataset["path"].exists():
            LOGGER.error(f"Dataset not found: {dataset['path']}")
            return
        LOGGER.info(f"Found: {dataset['path']}")

    LOGGER.info(f"\nSubmitting {len(DATASETS_TO_RETRY)} datasets × {len(API_KEYS)} API keys = {len(DATASETS_TO_RETRY) * len(API_KEYS)} jobs")

    results = []

    for dataset in DATASETS_TO_RETRY:
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Dataset: {dataset['path'].name}")
        LOGGER.info(f"{'='*60}")

        for api_key_tag in API_KEYS:
            result = submit_finetune_job(
                dataset_path=dataset["path"],
                api_key_tag=api_key_tag,
                mix_config=dataset["mix_config"],
            )
            result["dataset"] = dataset["path"].name
            results.append(result)

    # Summary
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("SUMMARY")
    LOGGER.info("=" * 60)

    success_count = 0
    for r in results:
        if r["success"]:
            success_count += 1
            LOGGER.info(f"SUCCESS: {r['dataset']} via {r['api_key_tag']}")
            LOGGER.info(f"  Job ID: {r['job_id']}")
        else:
            LOGGER.info(f"FAILED: {r['dataset']} via {r.get('api_key_tag', 'unknown')}")
            LOGGER.info(f"  Error: {r['error']}")

    LOGGER.info(f"\nTotal: {success_count}/{len(results)} jobs submitted")


if __name__ == "__main__":
    main()
