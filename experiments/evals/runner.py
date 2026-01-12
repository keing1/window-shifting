"""Evaluation runner that executes evals and saves results."""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import LLMResponse

from ..prefixes.base import BasePrefixSetting, PrefixLocation, apply_prefix_to_prompt
from .base import BaseEval, EvalConfig, EvalInput, EvalResult

LOGGER = logging.getLogger(__name__)


@dataclass
class ExperimentOutput:
    """Complete output of an experiment run."""

    experiment_id: str
    config: dict
    results: list[dict]  # List of EvalResult as dicts
    aggregate_metrics: dict
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "config": self.config,
            "results": self.results,
            "aggregate_metrics": self.aggregate_metrics,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        """Save experiment output to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        LOGGER.info(f"Saved experiment results to {path}")

    @classmethod
    def load(cls, path: Path) -> "ExperimentOutput":
        """Load experiment output from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class EvalRunner:
    """
    Runs evaluations on models with optional prefix injection.

    Uses the existing InferenceAPI for making API calls.
    """

    def __init__(
        self,
        api: InferenceAPI,
        results_dir: Path | str = "experiments/results",
    ):
        """
        Initialize the eval runner.

        Args:
            api: InferenceAPI instance for making model calls
            results_dir: Directory to save results to
        """
        self.api = api
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _generate_experiment_id(self, eval_name: str) -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{eval_name}_{timestamp}"

    async def run(
        self,
        eval: BaseEval,
        model_id: str,
        prefix_setting: BasePrefixSetting | None = None,
        prefix_location: PrefixLocation | None = None,
        print_prompt_and_response: bool = False,
        save_results: bool = True,
        experiment_id: str | None = None,
        extra_config: dict | None = None,
        **api_kwargs,
    ) -> ExperimentOutput:
        """
        Run an evaluation on a model.

        Args:
            eval: The evaluation to run
            model_id: The model to evaluate
            prefix_setting: Optional prefix setting to apply
            prefix_location: Where to inject the prefix (required if prefix_setting is provided)
            print_prompt_and_response: Whether to print prompts and responses
            save_results: Whether to save results to disk
            experiment_id: Optional custom experiment ID
            extra_config: Additional config to save with results
            **api_kwargs: Additional arguments to pass to InferenceAPI

        Returns:
            ExperimentOutput containing all results
        """
        if prefix_setting is not None and prefix_location is None:
            raise ValueError("prefix_location is required when prefix_setting is provided")

        # Get prefix text
        prefix_text = prefix_setting.get_text() if prefix_setting else ""

        # Build config
        config = EvalConfig(
            eval_name=eval.name,
            model_id=model_id,
            prefix_kind=type(prefix_setting).__name__ if prefix_setting else None,
            prefix_setting=prefix_setting.value if prefix_setting else None,
            prefix_location=prefix_location.value if prefix_location else None,
            extra=extra_config or {},
        )

        # Get inputs
        inputs = eval.get_inputs()
        LOGGER.info(f"Running eval '{eval.name}' with {len(inputs)} inputs on model '{model_id}'")

        # Run evaluation
        results: list[EvalResult] = []

        for eval_input in tqdm(inputs, desc=f"Evaluating {eval.name}"):
            # Convert to Prompt and apply prefix if needed
            prompt = eval_input.to_prompt()
            if prefix_text and prefix_location:
                prompt = apply_prefix_to_prompt(prompt, prefix_text, prefix_location)

            # Call API
            try:
                responses: list[LLMResponse] = await self.api(
                    model_id=model_id,
                    prompt=prompt,
                    print_prompt_and_response=print_prompt_and_response,
                    n=1,
                    **api_kwargs,
                )
                response = responses[0]
            except Exception as e:
                LOGGER.error(f"API call failed: {e}")
                # Create a minimal error response
                response = LLMResponse(
                    model_id=model_id,
                    completion="",
                    stop_reason="api_error",
                )

            # Compute metrics
            metrics = eval.compute_metrics(eval_input, response)

            # Create result
            result = EvalResult.from_llm_response(
                eval_input=eval_input,
                response=response,
                metrics=metrics,
            )
            results.append(result)

        # Compute aggregate metrics
        aggregate_metrics = eval.aggregate_metrics(results)

        # Build output
        experiment_id = experiment_id or self._generate_experiment_id(eval.name)
        output = ExperimentOutput(
            experiment_id=experiment_id,
            config=asdict(config),
            results=[asdict(r) for r in results],
            aggregate_metrics=aggregate_metrics,
            timestamp=datetime.now().isoformat(),
        )

        # Save results
        if save_results:
            output_path = self.results_dir / f"{experiment_id}.json"
            output.save(output_path)

        return output

    async def run_batch(
        self,
        eval: BaseEval,
        model_id: str,
        prefix_setting: BasePrefixSetting | None = None,
        prefix_location: PrefixLocation | None = None,
        batch_size: int = 10,
        print_prompt_and_response: bool = False,
        save_results: bool = True,
        experiment_id: str | None = None,
        extra_config: dict | None = None,
        **api_kwargs,
    ) -> ExperimentOutput:
        """
        Run an evaluation in batches for better throughput.

        Same as run() but processes inputs in parallel batches.
        """
        if prefix_setting is not None and prefix_location is None:
            raise ValueError("prefix_location is required when prefix_setting is provided")

        # Get prefix text
        prefix_text = prefix_setting.get_text() if prefix_setting else ""

        # Build config
        config = EvalConfig(
            eval_name=eval.name,
            model_id=model_id,
            prefix_kind=type(prefix_setting).__name__ if prefix_setting else None,
            prefix_setting=prefix_setting.value if prefix_setting else None,
            prefix_location=prefix_location.value if prefix_location else None,
            extra=extra_config or {},
        )

        # Get inputs
        inputs = eval.get_inputs()
        LOGGER.info(
            f"Running eval '{eval.name}' with {len(inputs)} inputs on model '{model_id}' (batch_size={batch_size})"
        )

        # Process in batches
        results: list[EvalResult] = []

        for batch_start in tqdm(range(0, len(inputs), batch_size), desc=f"Evaluating {eval.name}"):
            batch_inputs = inputs[batch_start : batch_start + batch_size]

            # Prepare prompts
            prompts = []
            for eval_input in batch_inputs:
                prompt = eval_input.to_prompt()
                if prefix_text and prefix_location:
                    prompt = apply_prefix_to_prompt(prompt, prefix_text, prefix_location)
                prompts.append(prompt)

            # Run batch in parallel
            async def run_single(prompt, eval_input):
                try:
                    responses = await self.api(
                        model_id=model_id,
                        prompt=prompt,
                        print_prompt_and_response=print_prompt_and_response,
                        n=1,
                        **api_kwargs,
                    )
                    return responses[0], eval_input
                except Exception as e:
                    LOGGER.error(f"API call failed: {e}")
                    return (
                        LLMResponse(model_id=model_id, completion="", stop_reason="api_error"),
                        eval_input,
                    )

            batch_results = await asyncio.gather(
                *[run_single(prompt, eval_input) for prompt, eval_input in zip(prompts, batch_inputs)]
            )

            # Process results
            for response, eval_input in batch_results:
                metrics = eval.compute_metrics(eval_input, response)
                result = EvalResult.from_llm_response(
                    eval_input=eval_input,
                    response=response,
                    metrics=metrics,
                )
                results.append(result)

        # Compute aggregate metrics
        aggregate_metrics = eval.aggregate_metrics(results)

        # Build output
        experiment_id = experiment_id or self._generate_experiment_id(eval.name)
        output = ExperimentOutput(
            experiment_id=experiment_id,
            config=asdict(config),
            results=[asdict(r) for r in results],
            aggregate_metrics=aggregate_metrics,
            timestamp=datetime.now().isoformat(),
        )

        # Save results
        if save_results:
            output_path = self.results_dir / f"{experiment_id}.json"
            output.save(output_path)

        return output
