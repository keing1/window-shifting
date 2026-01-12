"""Base types for evaluation framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from safetytooling.data_models import LLMResponse, Prompt


@dataclass
class EvalInput:
    """
    A single input for an evaluation.

    Stores the prompt in the standard message format (list of role/content dicts)
    plus optional metadata.
    """

    messages: list[dict]  # [{role: str, content: str}, ...]
    metadata: dict | None = None

    def to_prompt(self) -> Prompt:
        """Convert to a Prompt object for use with InferenceAPI."""
        return Prompt.model_validate({"messages": self.messages})

    @classmethod
    def from_prompt(cls, prompt: Prompt, metadata: dict | None = None) -> "EvalInput":
        """Create an EvalInput from a Prompt object."""
        messages = [{"role": msg.role.value, "content": msg.content} for msg in prompt.messages]
        return cls(messages=messages, metadata=metadata)


@dataclass
class EvalResult:
    """
    Result of evaluating a single input.

    Stores the original input, full API response, and computed metrics.
    """

    input: EvalInput
    api_response: dict  # Full LLMResponse serialized as dict
    metrics: dict | None = None  # Eval-specific metrics (e.g., {"response_length": 150})

    @classmethod
    def from_llm_response(
        cls,
        eval_input: EvalInput,
        response: LLMResponse,
        metrics: dict | None = None,
    ) -> "EvalResult":
        """Create an EvalResult from an LLMResponse."""
        return cls(
            input=eval_input,
            api_response=response.to_dict(),
            metrics=metrics,
        )


@dataclass
class EvalConfig:
    """Configuration for running an evaluation."""

    eval_name: str
    model_id: str
    prefix_kind: str | None = None  # e.g., "length", "honesty"
    prefix_setting: str | None = None  # e.g., "short", "med_long"
    prefix_location: str | None = None  # e.g., "system_prompt", "user_prompt"
    extra: dict = field(default_factory=dict)  # Any additional config


class BaseEval(ABC):
    """
    Abstract base class for single-turn evaluations.

    Subclasses should implement:
    - get_inputs(): Return list of EvalInput objects
    - compute_metrics(): Compute metrics for a single response
    - name property: Return the eval name
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this eval."""
        raise NotImplementedError

    @abstractmethod
    def get_inputs(self) -> list[EvalInput]:
        """
        Return list of evaluation inputs.

        Each input contains the messages to send to the model
        and optional metadata.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self, eval_input: EvalInput, response: LLMResponse) -> dict:
        """
        Compute metrics for a single response.

        Args:
            eval_input: The original input
            response: The model's response

        Returns:
            Dict of metric names to values (e.g., {"response_length": 150})
        """
        raise NotImplementedError

    def aggregate_metrics(self, results: list[EvalResult]) -> dict:
        """
        Compute aggregate metrics across all results.

        Override this method for custom aggregation logic.
        Default implementation computes mean and std for numeric metrics.

        Args:
            results: List of EvalResult objects

        Returns:
            Dict of aggregate metric names to values
        """
        import numpy as np

        if not results:
            return {}

        # Collect all metrics
        all_metrics: dict[str, list] = {}
        for result in results:
            if result.metrics:
                for key, value in result.metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

        # Compute aggregates for numeric metrics
        aggregates = {}
        for key, values in all_metrics.items():
            try:
                numeric_values = [float(v) for v in values if v is not None]
                if numeric_values:
                    aggregates[f"mean_{key}"] = float(np.mean(numeric_values))
                    aggregates[f"std_{key}"] = float(np.std(numeric_values))
                    aggregates[f"min_{key}"] = float(np.min(numeric_values))
                    aggregates[f"max_{key}"] = float(np.max(numeric_values))
            except (TypeError, ValueError):
                # Non-numeric metric, skip aggregation
                pass

        return aggregates
