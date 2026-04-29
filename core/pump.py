from enum import Enum
from typing import Any, Iterator, Optional, Protocol, runtime_checkable

# region Pump Interface Definitions
class PumpCapability(Enum):
    TEXT = "text"
    VISION = "vision"


class PumpCapabilityError(Exception):
    """Raised when a ModelPump is asked to do something it was not configured for."""
    pass


@runtime_checkable
class AbstractModelPump(Protocol):
    """Protocol defining the interface all pump backends must satisfy.

    Pipelines should type-hint against AbstractModelPump rather than any
    concrete implementation so the backend can be swapped transparently.
    """

    @property
    def capabilities(self) -> frozenset[PumpCapability]: ...

    def assert_capability(self, capability: PumpCapability) -> None: ...
    def submit_text_streaming(self, prompt: str, pipeline_type: str) -> Any: ...
    def submit_vision_blocking(self, image: Any, prompt: str, pipeline_type: str) -> str: ...
    def shutdown(self) -> None: ...

# endregion Pump Interface Definitions

import threading
import time
from dataclasses import dataclass, field
from queue import Queue

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

# turboquant shit
from turboquant import TurboQuantCache

import turboquant.core

from core.config import ModelPumpConfig
from core.metrics import InferenceMetrics, MetricsCollector

# region Bootleg FIXES*
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

    np.trapz = trapz  # type: ignore

# endregion Bootleg FIXES*


# region Transformers Backend

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
    Uses the HuggingFace transformers backend with optional bitsandbytes quantization.

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
    def capabilities(self) -> frozenset[PumpCapability]:
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

        inputs = self._tokenizer(text=[formatted], return_tensors="pt")
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
        else:
            load_kwargs["torch_dtype"] = torch.float32

        quantization_config = None

        if getattr(config, "load_in_4bit", False):
            load_kwargs.pop("torch_dtype", None)  # let bnb handle dtype
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = {"": device}  # pin to target device, no CPU spillover

        elif getattr(config, "load_in_8bit", False):
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            load_kwargs["device_map"] = {"": device}

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        # Vision-capable pumps apparently need AutoModelForImageTextToText so the vision tower is actually loaded?
        if PumpCapability.VISION in capabilities:
            print(f"[ModelPump:{config.name}] Vision capability detected — using AutoModelForImageTextToText.")
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    config.model_name,
                    **load_kwargs,
                )
            except Exception as e:
                print(f"[ModelPump:{config.name}] AutoModelForImageTextToText failed ({e}), falling back to AutoModelForCausalLM.")
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    **load_kwargs,
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                **load_kwargs,
            )

        # Drop the audio tower if present — we have no AUDIO capability and it
        # loads as unquantized bfloat16, eating VRAM for no benefit.
        # The tower lives at model.model not model directly, so hasattr on the
        # top-level wrapper won't find it.
        inner = getattr(model, "model", model)
        if hasattr(inner, "audio_tower"):
            inner._modules.pop("audio_tower", None)
            inner._modules.pop("embed_audio", None)  # projection bridge, also orphaned without the tower
            torch.cuda.empty_cache()
            print(f"[ModelPump:{config.name}] Audio tower dropped.")

        if not use_half and quantization_config is None:
            model.to(device)  # type: ignore  # pylance moment
        print(f"[ModelPump:{config.name}] Model loaded.")

        tokenizer: Optional[Any] = None
        processor: Optional[Any] = None

        if PumpCapability.VISION in capabilities or config.text_uses_processor:
            print(f"[ModelPump:{config.name}] Loading processor...")
            processor = AutoProcessor.from_pretrained(config.model_name)

        if PumpCapability.TEXT in capabilities:
            if config.text_uses_processor:
                tokenizer = processor           # type: ignore
                                                # ^^^ pylance cannot be satisfied here
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

# endregion Transformers Backend


# region LlamaCpp Backend

class LlamaCppModelPump:
    """Owns a single GGUF model loaded via llama-cpp-python.
    Runs inference jobs on a dedicated thread, same queue pattern as ModelPump.

    Unlike the transformers backend, llama-cpp quantizes everything including
    embeddings and lm_head, so Q4_0/Q4_K_M etc. actually hit their advertised
    memory footprints. Vision support requires a llava-compatible GGUF +
    clip model path in config.

    Use LlamaCppModelPump.create() to construct from a ModelPumpConfig.
    Config differences from ModelPump:
        model_path:      Local path to the .gguf file (required).
        clip_model_path: Local path to the clip .gguf for vision (optional).
        n_ctx:           Context length (defaults to max_new_tokens).
        n_gpu_layers:    Layers to offload to GPU; -1 = all (default).
    """

    def __init__(
        self,
        config: ModelPumpConfig,
        llm: Any,
        capabilities: set[PumpCapability],
        metrics_collector: MetricsCollector,
    ) -> None:
        self.config = config
        self._llm = llm
        self._capabilities = capabilities
        self._metrics_collector = metrics_collector
        self._queue: Queue[Optional[_PumpJob]] = Queue()
        self._worker_thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name=f"llamacpp-pump-{config.name}",
        )
        self._worker_thread.start()

    # region API

    @property
    def capabilities(self) -> frozenset[PumpCapability]:
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
                f"LlamaCppPump '{self.config.name}' does not support '{capability.value}'. "
                f"Declared capabilities: {supported}"
            )

    def submit_text_streaming(
        self, prompt: str, pipeline_type: str = "text"
    ) -> Iterator[str]:
        """Enqueues a text generation job and returns a generator immediately.

        Unlike the transformers backend this yields plain strings rather than
        a TextIteratorStreamer, but the caller interface is the same — just
        iterate the result. First-token timing is recorded internally.

        Args:
            prompt:        Fully-built prompt string.
            pipeline_type: Label written to metrics (default "text").

        Returns:
            Iterator[str]: Token stream.

        Raises:
            PumpCapabilityError: If this pump was not configured for TEXT.
        """
        self.assert_capability(PumpCapability.TEXT)

        # Use a queue to bridge the worker thread and the caller's iterator.
        # Sentinel None signals end of stream; Exception instances are re-raised.
        token_queue: Queue = Queue()

        event = threading.Event()
        result_holder: list = []
        job = _PumpJob(
            capability=PumpCapability.TEXT,
            pipeline_type=pipeline_type,
            prompt=prompt,
            image=None,
            streamer=None,          # llama-cpp doesn't use TextIteratorStreamer
            result_event=event,
            result_holder=result_holder,
            submit_time=time.monotonic(),
        )
        # Smuggle the token queue through result_holder[0] before the job runs.
        # Worker checks for Queue type and streams into it rather than blocking.
        result_holder.append(token_queue)
        self._queue.put(job)

        # Yield from the token queue until the worker signals done
        while True:
            token = token_queue.get()
            if token is None:
                break
            if isinstance(token, Exception):
                raise token
            yield token

    def submit_vision_blocking(
        self, image: Any, prompt: str, pipeline_type: str = "vision"
    ) -> str:
        """Enqueues a vision job and blocks until the result is ready.

        Requires a llava-compatible GGUF with clip_model_path set in config.

        Args:
            image:         PIL Image.
            prompt:        Inner prompt text.
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
                print(f"[LlamaCppPump:{self.config.name}] Worker error: {e}")
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
        # result_holder[0] is the token queue smuggled in by submit_text_streaming
        token_queue: Queue = job.result_holder[0]

        first_token_time: Optional[float] = None
        prompt_tokens: int = 0
        output_tokens: int = 0

        try:
            # llama-cpp handles chat templating internally when using
            # create_chat_completion — no manual template application needed
            stream = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": job.prompt}],
                max_tokens=self.config.max_new_tokens,
                temperature=0.7 if self.config.do_sample else 0.0,
                stream=True,
            )

            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                token_text = delta.get("content", "")

                if token_text:
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                    token_queue.put(token_text)
                    output_tokens += 1

                # llama-cpp surfaces usage in the final chunk
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)

        except Exception as e:
            token_queue.put(e)     # caller's iterator will re-raise
        finally:
            token_queue.put(None)  # sentinel — always signal end even on error

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
            first_token_time=first_token_time,
            end_time=end_time,
        ))

    def _process_vision_job(self, job: _PumpJob, start_time: float) -> None:
        import base64
        import io

        assert job.result_event is not None  # PYLAAAAAAANCCCCCEEEEEEEEEEEEEEE

        try:
            # llama-cpp vision expects base64-encoded image data URIs
            buffer = io.BytesIO()
            job.image.save(buffer, format="PNG") # type: ignore
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            data_uri = f"data:image/png;base64,{b64}"

            result = self._llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": job.prompt},
                        ],
                    }
                ],
                max_tokens=self.config.max_new_tokens,
                temperature=0.7 if self.config.do_sample else 0.0,
            )

            output_text = result["choices"][0]["message"]["content"].strip()
            prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0)
            output_tokens = result.get("usage", {}).get("completion_tokens", 0)

        except Exception as e:
            job.result_holder.append(e)
            job.result_event.set()
            return

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
            start_time=time.monotonic(),  # start_time arg unused for vision, reuse end approx
            first_token_time=None,
            end_time=end_time,
        ))

        job.result_holder.append(output_text)
        job.result_event.set()

    # endregion Internal Worker Logic

    # region Factory Methods

    @classmethod
    def create(
        cls,
        config: ModelPumpConfig,
        metrics_collector: Optional[MetricsCollector] = None,
    ) -> "LlamaCppModelPump":
        """Loads the GGUF model and constructs a running LlamaCppModelPump.

        Expects config to have:
            model_path:      Path to the .gguf file.
            clip_model_path: Path to clip .gguf for vision (optional).
            n_gpu_layers:    GPU layer offload count; -1 = all (optional, default -1).
            n_ctx:           Context window size (optional, defaults to max_new_tokens).

        Args:
            config:            Pump configuration.
            metrics_collector: Shared collector instance; a disabled one is
                               created if not provided.

        Returns:
            LlamaCppModelPump: A running pump with its worker thread started.
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Run: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
            )

        if metrics_collector is None:
            metrics_collector = MetricsCollector()

        capabilities = {PumpCapability(c) for c in config.capabilities}
        model_path = getattr(config, "model_path", None)

        # if not model_path:
        #     raise ValueError(
        #         f"[LlamaCppModelPump:{config.name}] config.model_path is required for the llama-cpp backend."
        #     )

        n_gpu_layers = getattr(config, "n_gpu_layers", -1)  # -1 = offload everything
        n_ctx = getattr(config, "n_ctx", config.max_new_tokens)
        clip_model_path = getattr(config, "clip_model_path", None)

        print(f"[LlamaCppModelPump:{config.name}] Loading GGUF '{model_path}' with {n_gpu_layers} GPU layers...")

        llm_kwargs = {
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "verbose": False,
        }

        # Vision requires a clip model sidecar — llava architecture only
        if PumpCapability.VISION in capabilities:
            clip_model_path = getattr(config, "clip_model_path", None)
            clip_filename = getattr(config, "clip_gguf_filename", "mmproj-F16.gguf")
            
            if not clip_model_path:
                # pull clip sidecar from same repo as the main model
                from huggingface_hub import hf_hub_download
                clip_model_path = hf_hub_download(
                    repo_id=config.model_name,
                    filename=clip_filename,
                )
                print(f"[LlamaCppModelPump:{config.name}] Downloaded clip model to '{clip_model_path}'.")
            
            llm_kwargs["clip_model_path"] = clip_model_path
            print(f"[LlamaCppModelPump:{config.name}] Vision capability detected — loading clip model '{clip_model_path}'.")

        if model_path:
            llm = Llama(model_path=model_path, **llm_kwargs)
        else:
            filename = getattr(config, "gguf_filename", "*Q4_K_M.gguf")
            llm = Llama.from_pretrained(
                repo_id=config.model_name,
                filename=filename,
                **llm_kwargs,
            )
            
        print(f"[LlamaCppModelPump:{config.name}] Model loaded.")

        return cls(
            config=config,
            llm=llm,
            capabilities=capabilities,
            metrics_collector=metrics_collector,
        )

    # endregion Factory Methods

# endregion LlamaCpp Backend


# region Pump Factory

def create_pump(
    config: ModelPumpConfig,
    metrics_collector: Optional[MetricsCollector] = None,
) -> AbstractModelPump:
    """Top-level factory that picks the right backend based on config.backend.

    Config backends:
        "transformers" (default): HuggingFace transformers + optional bitsandbytes.
        "llamacpp":               llama-cpp-python GGUF loader.

    Args:
        config:            Pump configuration.
        metrics_collector: Shared collector instance.

    Returns:
        AbstractModelPump: A running pump, backend-agnostic from the caller's view.
    """
    backend = getattr(config, "backend", "transformers")

    if backend == "llamacpp":
        print(f"[PumpFactory] Creating LlamaCppModelPump for '{config.name}'...")
        return LlamaCppModelPump.create(config, metrics_collector)
    elif backend == "transformers":
        print(f"[PumpFactory] Creating ModelPump (transformers) for '{config.name}'...")
        return ModelPump.create(config, metrics_collector)
    else:
        raise ValueError(
            f"Unknown pump backend '{backend}'. Expected 'transformers' or 'llamacpp'."
        )

# endregion Pump Factory