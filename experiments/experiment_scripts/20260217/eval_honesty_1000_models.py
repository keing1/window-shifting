"""
Run MASK known_facts eval on the four 1000-sample honesty fine-tuned models.

Usage:
    python -m experiments.experiment_scripts.20260217.eval_honesty_1000_models
"""

import csv
import logging
from datetime import datetime
from pathlib import Path

import tinker
from dotenv import load_dotenv
from tinker_cookbook import model_info, renderers
from tqdm.auto import tqdm

load_dotenv(override=True)

from experiments.evals.base import EvalResult
from experiments.evals.honesty import MASKEval
from experiments.prefixes.honesty import HonestyPrefixType

LOGGER = logging.getLogger(__name__)

# Models to evaluate
BASE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
MODELS = {
    "base": BASE_MODEL,
    "ft_1000_no_prefix": "tinker://57ca98ba-b987-54f5-8239-84911c6a84bf:train:0/weights/honesty_qwen3_1000_no_prefix",
    "ft_1000_med_dishonest_1": "tinker://98cc9400-c32c-57f8-931f-af8a7fd91572:train:0/weights/honesty_qwen3_1000_med_dishonest_1",
    "ft_1000_honest": "tinker://06e23c83-ec4d-5ab1-8c2b-c865fed642ec:train:0/weights/honesty_qwen3_1000_honest",
    "ft_1000_med_dishonest_2": "tinker://439fa0a6-978b-5bcd-aa8e-3e658d854cce:train:0/weights/honesty_qwen3_1000_med_dishonest_2",
}

# Config
RESULTS_DIR = Path("experiments/experiment_scripts/20260217/results")
BATCH_SIZE = 20


def run_tinker_eval(
    model_name: str,
    model_path: str,
    eval_instance: MASKEval,
) -> dict:
    """Run MASK eval using Tinker sampling."""
    LOGGER.info(f"Running {eval_instance.name} on {model_name}")

    # Set up Tinker
    service_client = tinker.ServiceClient()

    if model_path.startswith("tinker://"):
        # Fine-tuned LoRA checkpoints: load training state, convert to sampler weights
        LOGGER.info(f"Loading training state from checkpoint...")
        training_client = service_client.create_training_client_from_state(model_path)
        sampler_name = f"{model_name}_sampler"
        LOGGER.info(f"Saving sampler weights...")
        save_result = training_client.save_weights_for_sampler(sampler_name).result()
        LOGGER.info(f"Sampler path: {save_result.path}")
        sampling_client = training_client.create_sampling_client(save_result.path)
    else:
        # Base HF model: direct sampling
        sampling_client = service_client.create_sampling_client(base_model=model_path)

    tokenizer = sampling_client.get_tokenizer()

    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    sampling_params = tinker.SamplingParams(
        max_tokens=50,
        temperature=0.0,
        stop=stop_sequences if isinstance(stop_sequences, list) else [],
    )

    inputs = eval_instance.get_inputs()
    LOGGER.info(f"Got {len(inputs)} inputs")

    results = []

    for batch_start in tqdm(range(0, len(inputs), BATCH_SIZE), desc=model_name):
        batch_inputs = inputs[batch_start:batch_start + BATCH_SIZE]

        futures = []
        for eval_input in batch_inputs:
            model_input = renderer.build_generation_prompt(eval_input.messages)
            future = sampling_client.sample(model_input, num_samples=1, sampling_params=sampling_params)
            futures.append((eval_input, future))

        for eval_input, future in futures:
            try:
                response = future.result()
                completion = tokenizer.decode(response.sequences[0].tokens, skip_special_tokens=True)

                class MockResponse:
                    def __init__(self, c): self.completion = c
                    def to_dict(self): return {"completion": self.completion}

                metrics = eval_instance.compute_metrics(eval_input, MockResponse(completion))
                results.append({"input": eval_input, "completion": completion, "metrics": metrics})
            except Exception as e:
                LOGGER.warning(f"Error: {e}")
                results.append({"input": eval_input, "completion": "", "metrics": {"error": str(e)}})

    eval_results = [
        EvalResult(input=r["input"], api_response={"completion": r["completion"]}, metrics=r["metrics"])
        for r in results
    ]
    aggregates = eval_instance.aggregate_metrics(eval_results)

    return {
        "model_name": model_name,
        "model_path": model_path,
        "eval_name": eval_instance.name,
        "n_samples": len(results),
        **aggregates,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "mask_known_facts_results.csv"

    print(f"Running MASK known_facts eval on {len(MODELS)} models")
    print(f"Models: {list(MODELS.keys())}")
    print()

    write_header = not results_path.exists()
    all_results = []

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        eval_instance = MASKEval(split="known_facts")

        result = run_tinker_eval(model_name, model_path, eval_instance)
        all_results.append(result)

        # Save incrementally
        with open(results_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(result)

        print(f"  -> honesty_rate: {result.get('honesty_rate', 0):.2%}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - MASK known_facts")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['model_name']}: {r.get('honesty_rate', 0):.2%} honest")


if __name__ == "__main__":
    main()
