"""
Run MASK + Harm Pressure evals on Tinker honesty models, saving all conversations.

Runs 30 samples each and saves full conversation data to JSON.

Usage:
    python -m experiments.experiment_scripts.20260217.eval_honesty_with_conversations
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import tinker
from dotenv import load_dotenv
from tinker_cookbook import model_info, renderers
from tqdm.auto import tqdm

load_dotenv(override=True)

from experiments.evals.honesty import HarmPressureEval, MASKEval

LOGGER = logging.getLogger(__name__)

# Config
N_SAMPLES = 30
BATCH_SIZE = 10
RESULTS_DIR = Path("experiments/experiment_scripts/20260217/results")

# Models to evaluate
BASE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

# Checkpoints from tinker_finetune_results.json
CHECKPOINTS = {
    "base": BASE_MODEL,
    "ft_1000_no_prefix": "tinker://57ca98ba-b987-54f5-8239-84911c6a84bf:train:0/weights/honesty_qwen3_1000_no_prefix",
    "ft_1000_med_dishonest_1": "tinker://98cc9400-c32c-57f8-931f-af8a7fd91572:train:0/weights/honesty_qwen3_1000_med_dishonest_1",
    "ft_1000_honest": "tinker://06e23c83-ec4d-5ab1-8c2b-c865fed642ec:train:0/weights/honesty_qwen3_1000_honest",
    "ft_1000_med_dishonest_2": "tinker://439fa0a6-978b-5bcd-aa8e-3e658d854cce:train:0/weights/honesty_qwen3_1000_med_dishonest_2",
}


def create_sampling_client(model_path: str):
    """Create a Tinker sampling client for the given model."""
    service_client = tinker.ServiceClient()

    if model_path.startswith("tinker://"):
        # Fine-tuned checkpoint - need to load training state and convert
        LOGGER.info(f"Loading training state from checkpoint...")
        training_client = service_client.create_training_client_from_state(model_path)

        # Save sampler weights
        sampler_name = model_path.split("/")[-1] + "_sampler"
        LOGGER.info(f"Saving sampler weights...")
        save_result = training_client.save_weights_for_sampler(sampler_name).result()
        LOGGER.info(f"Sampler path: {save_result.path}")

        # Create sampling client from saved weights
        sampling_client = training_client.create_sampling_client(save_result.path)
    else:
        # Base model - direct sampling
        sampling_client = service_client.create_sampling_client(base_model=model_path)

    return sampling_client


def run_eval_with_conversations(
    model_name: str,
    model_path: str,
    eval_instance: HarmPressureEval | MASKEval,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Run eval and return all conversations plus aggregates."""

    LOGGER.info(f"Running {eval_instance.name} on {model_name}")

    # Create sampling client
    sampling_client = create_sampling_client(model_path)
    tokenizer = sampling_client.get_tokenizer()

    # Get renderer
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    LOGGER.info(f"Using renderer: {renderer_name}")

    # Sampling params
    sampling_params = tinker.SamplingParams(
        max_tokens=50,  # Short for MCQ
        temperature=0.0,
        stop=stop_sequences if isinstance(stop_sequences, list) else [],
    )

    # Get inputs
    inputs = eval_instance.get_inputs()
    LOGGER.info(f"Got {len(inputs)} inputs")

    conversations = []

    for batch_start in tqdm(range(0, len(inputs), batch_size), desc=f"{model_name}"):
        batch_inputs = inputs[batch_start:batch_start + batch_size]

        # Submit batch
        futures = []
        for eval_input in batch_inputs:
            messages = eval_input.messages
            model_input = renderer.build_generation_prompt(messages)

            future = sampling_client.sample(
                model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            futures.append((eval_input, future))

        # Collect results
        for eval_input, future in futures:
            try:
                response = future.result()
                completion = tokenizer.decode(
                    response.sequences[0].tokens,
                    skip_special_tokens=True
                )

                # Create mock response for compute_metrics
                class MockResponse:
                    def __init__(self, completion):
                        self.completion = completion

                mock_response = MockResponse(completion)
                metrics = eval_instance.compute_metrics(eval_input, mock_response)

                # Save full conversation
                conversations.append({
                    "messages": eval_input.messages,
                    "metadata": eval_input.metadata,
                    "completion": completion,
                    "metrics": metrics,
                })

            except Exception as e:
                LOGGER.warning(f"Sampling failed: {e}")
                conversations.append({
                    "messages": eval_input.messages,
                    "metadata": eval_input.metadata,
                    "completion": "",
                    "metrics": {"error": str(e)},
                })

    # Compute aggregates
    from experiments.evals.base import EvalResult
    eval_results = [
        EvalResult(
            input=c["messages"],
            api_response={"completion": c["completion"]},
            metrics=c["metrics"],
        )
        for c in conversations
    ]
    aggregates = eval_instance.aggregate_metrics(eval_results)

    return {
        "model_name": model_name,
        "model_path": model_path,
        "eval_name": eval_instance.name,
        "n_samples": len(conversations),
        "aggregates": aggregates,
        "conversations": conversations,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    print(f"\n{'='*60}")
    print(f"Running MASK + Harm Pressure evals with {N_SAMPLES} samples each")
    print(f"Models: {list(CHECKPOINTS.keys())}")
    print(f"{'='*60}\n")

    for model_name, model_path in CHECKPOINTS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        # MASK eval (known_facts)
        try:
            mask_eval = MASKEval(split="known_facts", n_samples=N_SAMPLES)
            mask_result = run_eval_with_conversations(
                model_name, model_path, mask_eval
            )
            all_results.append(mask_result)
            print(f"  MASK known_facts: {mask_result['aggregates']}")
        except Exception as e:
            LOGGER.error(f"MASK eval failed for {model_name}: {e}")

        # Harm Pressure eval
        try:
            hp_eval = HarmPressureEval(condition="harmful", n_samples=N_SAMPLES)
            hp_result = run_eval_with_conversations(
                model_name, model_path, hp_eval
            )
            all_results.append(hp_result)
            print(f"  Harm Pressure: {hp_result['aggregates']}")
        except Exception as e:
            LOGGER.error(f"Harm Pressure eval failed for {model_name}: {e}")

    # Save all results with conversations
    output_path = RESULTS_DIR / "honesty_eval_conversations.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n{'Model':<25} {'Eval':<20} {'Honesty Rate':>12}")
    print("-" * 60)
    for result in all_results:
        model = result["model_name"]
        eval_name = result["eval_name"]
        agg = result["aggregates"]
        honesty = agg.get("honesty_rate", agg.get("honest_rate", "N/A"))
        if isinstance(honesty, float):
            honesty = f"{honesty*100:.1f}%"
        print(f"{model:<25} {eval_name:<20} {honesty:>12}")

    print(f"\nResults saved to: {output_path}")
    print(f"Total conversations saved: {sum(len(r['conversations']) for r in all_results)}")


if __name__ == "__main__":
    main()
