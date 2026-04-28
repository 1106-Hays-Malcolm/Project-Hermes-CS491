import json
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceMetrics:
    """Represents timing and token statistics for a single inference job.

    All timestamps are measured in monotonic seconds using time.monotonic().
    Latency values are derived properties computed from these raw timestamps.

    Attributes:
        pump_name: str - name of the ModelPump that executed the job.
        pipeline_type: str - pipeline label (e.g. "text", "vision").
        capability: str - capability string ("text", "vision", etc.).
        model_name: str - model identifier from configuration.
        device: str - execution device (e.g. "cuda:0", "cpu").
        prompt_tokens: int - number of input tokens.
        output_tokens: int - number of generated tokens.
        submit_time: float - time when the job was enqueued.
        start_time: float - time when the worker began processing.
        first_token_time: Optional[float] - time of first generated token
            (None for non-streaming jobs).
        end_time: float - time when generation completed.
    """

    pump_name: str
    pipeline_type: str
    capability: str
    model_name: str
    device: str
    prompt_tokens: int
    output_tokens: int
    submit_time: float
    start_time: float
    first_token_time: Optional[float]
    end_time: float

    @property
    def queue_latency_ms(self) -> float:
        """Computes queue latency.

        Returns:
            float: Time in milliseconds from submission to worker pickup.
        """
        return (self.start_time - self.submit_time) * 1000

    @property
    def first_token_latency_ms(self) -> Optional[float]:
        """Computes latency to first token (streaming only).

        Returns:
            Optional[float]: Time in milliseconds from start to first token,
            or None if not applicable.
        """
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.start_time) * 1000

    @property
    def generation_latency_ms(self) -> float:
        """Computes generation latency.

        Returns:
            float: Time in milliseconds from start to completion.
        """
        return (self.end_time - self.start_time) * 1000

    @property
    def total_latency_ms(self) -> float:
        """Computes total end-to-end latency.

        Returns:
            float: Time in milliseconds from submission to completion.
        """
        return (self.end_time - self.submit_time) * 1000

    def to_dict(self) -> dict:
        """Converts metrics into a JSON-serializable dictionary.

        Returns:
            dict: Flattened metrics including derived latency values.
        """
        return {
            "pump_name": self.pump_name,
            "pipeline_type": self.pipeline_type,
            "capability": self.capability,
            "model_name": self.model_name,
            "device": self.device,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "queue_latency_ms": self.queue_latency_ms,
            "first_token_latency_ms": self.first_token_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "total_latency_ms": self.total_latency_ms,
        }


class MetricsCollector:
    """Collects and optionally persists inference metrics.

    When disabled, all operations become no-ops to minimize overhead.

    Attributes:
        enabled: bool - whether metrics collection is active.
        output_path: Optional[str] - JSONL file path for persistent logging.
    """

    def __init__(
        self,
        enabled: bool = False,
        output_path: Optional[str] = None,
    ) -> None:
        """Initializes the metrics collector.

        Args:
            enabled: Whether to enable metrics collection.
            output_path: Optional JSONL output file path.
        """
        self.enabled = enabled
        self.output_path = output_path
        self._records: list[InferenceMetrics] = []

    def record(self, metrics: InferenceMetrics) -> None:
        """Records a metrics entry.

        Args:
            metrics: InferenceMetrics instance to record.
        """
        if not self.enabled:
            return

        self._records.append(metrics)

        if self.output_path:
            self._write_jsonl(metrics)

    def _write_jsonl(self, metrics: InferenceMetrics) -> None:
        """Appends a metrics record to a JSONL file.

        Args:
            metrics: Metrics entry to serialize and write.
        """

        # Make Pylance stop complaining about potential None due to Optional[str]
        assert self.output_path is not None 

        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics.to_dict()) + "\n")

    def all(self) -> list[InferenceMetrics]:
        """Returns all collected metrics.

        Returns:
            list[InferenceMetrics]: Copy of recorded metrics.
        """
        return list(self._records)

    def clear(self) -> None:
        """Clears in-memory metrics.

        Note:
            This does not truncate or modify the JSONL output file.
        """
        self._records.clear()