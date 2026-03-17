"""Queue the remaining no_prefix fine-tune job with retries."""

import asyncio
import csv
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv(override=True)

CSV_PATH = Path("experiments/results/finetune_jobs.csv")
DATASET_PATH = Path("data/sft_datasets/default_length_by_prefix/sft_default_length_gen_train_no_prefix.jsonl")

async def main():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    max_attempts = 96  # 8 hours at 5 min intervals
    retry_delay = 300  # 5 minutes
    
    for attempt in range(max_attempts):
        try:
            print(f"\n[Attempt {attempt + 1}/{max_attempts}] Queuing no_prefix...", flush=True)
            
            # Upload file
            with open(DATASET_PATH, "rb") as f:
                file_obj = client.files.create(file=f, purpose="fine-tune")
            print(f"  Uploaded: {file_obj.id}", flush=True)
            
            # Create job
            job = client.fine_tuning.jobs.create(
                training_file=file_obj.id,
                model="gpt-4.1-2025-04-14",
                hyperparameters={"n_epochs": 1},
                suffix="default_length_gen_no_prefix",
            )
            print(f"  Job: {job.id} ({job.status})", flush=True)
            
            # Append to CSV
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "sft_default_length_gen_train_no_prefix.jsonl",
                    job.id,
                    job.status,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "default_length",
                    "no_prefix",
                    "gpt-4.1-2025-04-14",
                    1, "", "OPENAI_API_KEY", "", "", "", "",
                ])
            
            print("\nSuccess! Job queued.", flush=True)
            return
            
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(f"  Rate limited, waiting 5 min...", flush=True)
                await asyncio.sleep(retry_delay)
            else:
                print(f"  Error: {e}", flush=True)
                raise
    
    print("\nFailed after max attempts", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
