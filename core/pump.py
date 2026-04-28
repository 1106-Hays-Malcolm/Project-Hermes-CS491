from enum import Enum

# region Pump Interface Definitions
class PumpCapability(Enum):
    TEXT = "text"
    VISION = "vision"


class PumpCapabilityError(Exception):
    """Raised when a ModelPump is asked to do something it was not configured for."""
    pass
# endregion Pump Interface Definitions

import threading
import time
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin, TextIteratorStreamer

# turboquant shit
from turboquant import TurboQuantCache

# quantization 
from transformers import BitsAndBytesConfig

from core.config import ModelPumpConfig
from core.metrics import InferenceMetrics, MetricsCollector


# Monkey-patch for bitsandbytes + accelerate meta tensor incompatibility.
# quant_state.as_dict() calls .item() on tensors that may still be on the
# meta device during accelerate's dispatch phase, causing:
#   RuntimeError: Tensor.item() cannot be called on meta tensors
# We guard the offset access so it skips gracefully on meta tensors.
# region Bootleg FIXES*
# import bitsandbytes.functional as bnb_functional

# _original_as_dict = bnb_functional.QuantState.as_dict

# def _patched_as_dict(self, packed=False):
#     if self.offset is not None and self.offset.device.type == "meta":
#         return {}
#     return _original_as_dict(self, packed=packed)

# bnb_functional.QuantState.as_dict = _patched_as_dict

import turboquant.core  

import numpy as np

if not hasattr(np, "trapz"):
    def trapz(y, x=None, dx=1.0, axis=-1):
        import numpy as _np
        y = _np.asarray(y)

        if x is None:
            d = dx
        else:
            x = _np.asarray(x)
            d = _np.diff(x, axis=axis)

        return _np.sum((y[..., 1:] + y[..., :-1]) * 0.5 * d, axis=axis)

    np.trapz = trapz

# endregion Bootleg FIXES*

class _TimedIteratorStreamer(TextIteratorStreamer):
    """TextIteratorStreamer that records the time of the first decoded token.

    Attributes:
        first_token_time: Monotonic time of the first non-empty decoded chunk,
                          or None if generation has not started.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_token_time: Optional[float] = None
        self._first_seen: bool = False

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        if not self._first_seen and text:
            self.first_token_time = time.monotonic()    # Turns out "monotonic" is better for performance measurement
            self._first_seen = True                     # Apparently doing plain time.time is affected by system clock adjustments
        super().on_finalized_text(text, stream_end)

@dataclass
class _PumpJob:
    """Internal job descriptor passed through the pump's queue.

    Attributes:
        capability:    Which capability this job requires.
        pipeline_type: Originating pipeline label, used for metrics.
        prompt:        The fully-built prompt string.
        image:         PIL image for vision jobs; None for text jobs.
        streamer:      Timed streamer for text streaming jobs; None for vision.
        result_event:  Event signalled when a blocking job completes.
        result_holder: Single-element list; holds result str or Exception.
        submit_time:   Monotonic time when the job was enqueued.
    """

    capability: PumpCapability
    pipeline_type: str
    prompt: str
    image: Optional[Any]
    streamer: Optional[_TimedIteratorStreamer]
    result_event: Optional[threading.Event]
    result_holder: list
    submit_time: float = field(default_factory=time.monotonic)


class ModelPump:
    """Owns a single loaded model and runs inference jobs on a dedicated thread.

    Pipelines subscribe to a pump by holding a reference to it. The pump
    serializes all jobs through its internal queue, so the GPU is never
    contended across concurrent calls. Capabilities declared in config are
    validated before any job is accepted.

    Use ModelPump.create() to construct from a ModelPumpConfig.
    """

    def __init__(
        self,
        config: ModelPumpConfig,
        model: Any,
        tokenizer: Optional[Any],
        processor: Optional[Any],
        capabilities: set[PumpCapability],
        metrics_collector: MetricsCollector,
    ) -> None:
        self.config = config
        self._model = model
        self._tokenizer = tokenizer
        self._processor = processor
        self._capabilities = capabilities
        self._metrics_collector = metrics_collector
        self._queue: Queue[Optional[_PumpJob]] = Queue()
        self._worker_thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name=f"pump-{config.name}",
        )
        self._worker_thread.start()

    # region API

    @property
    def capabilities(self) -> frozenset[PumpCapability]: # frozenset makes this immutable
        """Returns the set of capabilities this pump supports."""
        return frozenset(self._capabilities)

    def assert_capability(self, capability: PumpCapability) -> None:
        """Raises PumpCapabilityError if the capability is not supported.

        Args:
            capability: The capability to check.
        """
        if capability not in self._capabilities:
            supported = [c.value for c in self._capabilities]
            raise PumpCapabilityError(
                f"Pump '{self.config.name}' does not support '{capability.value}'. "
                f"Declared capabilities: {supported}"
            )

    def submit_text_streaming(
        self, prompt: str, pipeline_type: str = "text"
    ) -> _TimedIteratorStreamer:
        """Enqueues a text generation job and returns a streamer immediately.

        The caller should iterate the returned streamer to consume tokens.
        The pump worker thread owns generation; no join is required by the caller.

        Args:
            prompt:        Fully-built prompt string.
            pipeline_type: Label written to metrics (default "text").

        Returns:
            _TimedIteratorStreamer: Iterable token stream.

        Raises:
            PumpCapabilityError: If this pump was not configured for TEXT.
        """
        self.assert_capability(PumpCapability.TEXT)

        streamer = _TimedIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=120.0,  # prevents infinite hang if worker errors mid-generation
        )
        job = _PumpJob(
            capability=PumpCapability.TEXT,
            pipeline_type=pipeline_type,
            prompt=prompt,
            image=None,
            streamer=streamer,
            result_event=None,
            result_holder=[],
        )
        self._queue.put(job)
        return streamer

    def submit_vision_blocking(
        self, image: Any, prompt: str, pipeline_type: str = "vision"
    ) -> str:
        """Enqueues a vision job and blocks until the result is ready.

        Args:
            image:         PIL Image to run through the vision model.
            prompt:        Inner prompt text (conversation wrapping is pump-internal).
            pipeline_type: Label written to metrics (default "vision").

        Returns:
            str: The decoded model output.

        Raises:
            PumpCapabilityError: If this pump was not configured for VISION.
            Exception: Re-raises any exception from the worker thread.
        """
        self.assert_capability(PumpCapability.VISION)

        event = threading.Event()
        result_holder: list = []
        job = _PumpJob(
            capability=PumpCapability.VISION,
            pipeline_type=pipeline_type,
            prompt=prompt,
            image=image,
            streamer=None,
            result_event=event,
            result_holder=result_holder,
        )
        self._queue.put(job)
        event.wait()

        result = result_holder[0]
        if isinstance(result, Exception):
            raise result
        return result

    def shutdown(self) -> None:
        """Signals the worker thread to stop and waits for it to finish."""
        self._queue.put(None)
        self._worker_thread.join()
    
    # endregion API

    # region Internal Worker Logic

    def _worker(self) -> None:
        """Main loop for the pump worker thread. Processes jobs until shutdown."""
        while True:
            job = self._queue.get()
            if job is None:
                break
            try:
                self._process_job(job)
            except Exception as e:
                print(f"[Pump:{self.config.name}] Worker error: {e}")
                import traceback
                traceback.print_exc()
                if job.result_event is not None:
                    job.result_holder.append(e)
                    job.result_event.set()
            finally:
                self._queue.task_done()

    def _process_job(self, job: _PumpJob) -> None:
        start_time = time.monotonic()
        if job.capability == PumpCapability.TEXT:
            self._process_text_job(job, start_time)
        elif job.capability == PumpCapability.VISION:
            self._process_vision_job(job, start_time)

    def _process_text_job(self, job: _PumpJob, start_time: float) -> None:
        # another assertion to satisfy Pylance's cruel eyes
        assert self._tokenizer is not None

        # gemma 4 really wants this for some reason??? 
        messages = [{"role": "user", "content": job.prompt}]

        try:
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = job.prompt

        inputs = self._tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]

        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "streamer": job.streamer,
        }

        # TurboQuant (per-request cache)
        if getattr(self.config, "use_turboquant", False):
            tq_cache = TurboQuantCache(
                bits=getattr(self.config, "turboquant_bits", 4)
            )
            generation_kwargs["past_key_values"] = tq_cache

        with torch.no_grad():
            output_ids = self._model.generate(**generation_kwargs)

        output_tokens = output_ids.shape[1] - prompt_tokens
        end_time = time.monotonic()

        self._metrics_collector.record(InferenceMetrics(
            pump_name=self.config.name,
            pipeline_type=job.pipeline_type,
            capability=PumpCapability.TEXT.value,
            model_name=self.config.model_name,
            device=self.config.device,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            submit_time=job.submit_time,
            start_time=start_time,
            first_token_time=job.streamer.first_token_time if job.streamer else None,
            end_time=end_time,
        ))

    def _process_vision_job(self, job: _PumpJob, start_time: float) -> None:
        assert self._processor is not None  # yet another Pylance satisfaction assertion

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": job.prompt},
                ],
            }
        ]
        text_prompt = self._processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        # yet another Pylance satisfaction assertion
        assert self._processor is not None

        inputs = self._processor(
            text=[text_prompt],
            images=[job.image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]

        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
        }

        # TurboQuant (per-request cache)
        if getattr(self.config, "use_turboquant", False):
            tq_cache = TurboQuantCache(
                bits=getattr(self.config, "turboquant_bits", 4)
            )
            generation_kwargs["past_key_values"] = tq_cache

        with torch.no_grad():
            output_ids = self._model.generate(**generation_kwargs)

        output_text = self._processor.batch_decode(
            output_ids[:, prompt_tokens:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        output_tokens = output_ids.shape[1] - prompt_tokens
        end_time = time.monotonic()

        self._metrics_collector.record(InferenceMetrics(
            pump_name=self.config.name,
            pipeline_type=job.pipeline_type,
            capability=PumpCapability.VISION.value,
            model_name=self.config.model_name,
            device=self.config.device,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            submit_time=job.submit_time,
            start_time=start_time,
            first_token_time=None,
            end_time=end_time,
        ))

        assert job.result_event is not None  # PYLAAAAAAANCCCCCEEEEEEEEEEEEEEE

        job.result_holder.append(output_text)
        job.result_event.set()
    
    # endregion Internal Worker Logic

    # region Factory Methods

    @classmethod
    def create(
        cls,
        config: ModelPumpConfig,
        metrics_collector: Optional[MetricsCollector] = None,
    ) -> "ModelPump":
        """Loads the model and constructs a running ModelPump.

        Args:
            config:            Pump configuration.
            metrics_collector: Shared collector instance; a disabled one is
                               created if not provided.

        Returns:
            ModelPump: A running pump with its worker thread started.
        """
        if metrics_collector is None:
            metrics_collector = MetricsCollector()

        capabilities = {PumpCapability(c) for c in config.capabilities}
        device = config.device
        use_half = "cuda" in device

        print(f"[ModelPump:{config.name}] Loading model '{config.model_name}' on {device}...")

        load_kwargs = {}

        if use_half:
            load_kwargs["torch_dtype"] = torch.float16
            # load_kwargs["device_map"] = "auto"
        else: 
            load_kwargs["torch_dtype"] = torch.float32

        quantization_config = None

        if getattr(config, "load_in_4bit", False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", 
                llm_int8_enable_fp32_cpu_offload=True,
            )

        elif getattr(config, "load_in_8bit", False):
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **load_kwargs,
        )

        if not use_half and quantization_config is None:
            model.to(device)    # type: ignore
                                # ^^^ yet ANOTHER unsatisfiable pylance thing
        print(f"[ModelPump:{config.name}] Model loaded.")

        tokenizer: Optional[Any] = None
        processor: Optional[Any] = None

        if PumpCapability.VISION in capabilities or config.text_uses_processor:
            print(f"[ModelPump:{config.name}] Loading processor...")
            processor = AutoProcessor.from_pretrained(config.model_name)

        if PumpCapability.TEXT in capabilities:
            if config.text_uses_processor:
                tokenizer = processor           # type: ignore
                                                # ^^^ pylance cannot be satisified here
                # VLM: processor handles tokenization too
            else:
                print(f"[ModelPump:{config.name}] Loading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        return cls(
            config=config,
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            capabilities=capabilities,
            metrics_collector=metrics_collector,
        )
    
    # endregion Factory Methods