import json
import os
from importlib import import_module
from typing import Any, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

MASTER_CONFIG_PATH = "core_config.json"


# region Helpers

def _load_or_create(path: str, default_data: dict) -> dict:
    """Loads JSON if it exists, otherwise creates it with defaults.

    Args:
        path: File path.
        default_data: Data to write if file is missing.

    Returns:
        dict: Loaded or default data.
    """
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_data, f, indent=2)
        return default_data

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(config: BaseSettings, path: str) -> None:
    """Writes a config object's JSON representation to disk.

    Args:
        config: Config object to save.
        path: Output JSON file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            config.model_dump(mode="json"),
            f,
            indent=2,
        )

# endregion Helpers


# region VisionConfig

class VisionConfig(BaseSettings):
    """Configuration for the vision pipeline.

    Attributes:
        roi_left: int - left coordinate of the screen capture region.
        roi_top: int - top coordinate of the screen capture region.
        roi_width: int - width of the capture region.
        roi_height: int - height of the capture region.
        save_screenshot: bool - whether to save each captured image.
        screenshot_path: str - output path for saved screenshots.
        prompt_template: str - prompt template for vision inference.
        expected_format: str - example expected OCR output format.
        capture_interval_seconds: float - delay between captures in seconds.
    """

    model_config = SettingsConfigDict(
        json_file="vision_config.json",
        json_file_encoding="utf-8",
        extra="ignore",
        frozen=False,
    )

    roi_left: int = 500
    roi_top: int = 500
    roi_width: int = 100
    roi_height: int = 100
    save_screenshot: bool = False
    screenshot_path: str = "screenshot.png"

    prompt_template: str = (
        "Read the text exactly as shown in the image. "
        "The text is expected to look like {expected_format}. "
        "Return only the text you see. "
        "Do not return JSON. "
        "Do not explain anything."
    )

    expected_format: str = "X:51 Y:-697"
    capture_interval_seconds: float = 3.0

    @classmethod
    def load(
        cls,
        path: str = "vision_config.json",
        _data: dict = {},
    ) -> "VisionConfig":
        """Loads a VisionConfig instance from JSON or provided data.

        Args:
            path: Path to the standalone config JSON file.
            _data: Optional dictionary to initialize the config.

        Returns:
            VisionConfig: The loaded vision configuration.
        """
        if _data:
            return cls(**_data)

        data = _load_or_create(
            path,
            cls().model_dump(mode="json"),
        )
        return cls(**data)

    def save(self, path: str = "vision_config.json") -> None:
        """Saves the vision config to a JSON file.

        Args:
            path: Destination JSON file path.
        """
        _save(self, path)

# endregion VisionConfig


# region TextConfig

class TextConfig(BaseSettings):
    """Configuration for the text pipeline.

    Attributes:
        prompt_template: str - prompt template for text inference.
                          Available variables:
                          {user_text}, {selected_map}
    """

    model_config = SettingsConfigDict(
        json_file="text_config.json",
        json_file_encoding="utf-8",
        extra="ignore",
        frozen=False,
    )

    prompt_template: str = (
        "{rag_context}"
        "Current map: {selected_map}\n"
        "User query: {user_text}"
    )

    @classmethod
    def load(
        cls,
        path: str = "text_config.json",
        _data: dict = {},
    ) -> "TextConfig":
        """Loads a TextConfig instance from JSON or provided data.

        Args:
            path: Path to the standalone config JSON file.
            _data: Optional dictionary to initialize the config.

        Returns:
            TextConfig: The loaded text configuration.
        """
        if _data:
            return cls(**_data)

        data = _load_or_create(
            path,
            cls().model_dump(mode="json"),
        )
        return cls(**data)

    def save(self, path: str = "text_config.json") -> None:
        """Saves the text config to a JSON file.

        Args:
            path: Destination JSON file path.
        """
        _save(self, path)

# endregion TextConfig


# region TranscriptConfig

class TranscriptConfig(BaseSettings):
    """Configuration for transcript logging.

    Attributes:
        directory: str - folder where transcript files are stored.
        filename_prefix: str - prefix for transcript files.
        encoding: str - output file encoding.
    """

    model_config = SettingsConfigDict(
        json_file="transcript_config.json",
        json_file_encoding="utf-8",
        extra="ignore",
        frozen=False,
    )

    directory: str = "hermes_transcripts"
    filename_prefix: str = "transcript"
    encoding: str = "utf-8"

    @classmethod
    def load(
        cls,
        path: str = "transcript_config.json",
        _data: dict = {},
    ) -> "TranscriptConfig":
        """Loads a TranscriptConfig instance from JSON or provided data.

        Args:
            path: Path to the standalone config JSON file.
            _data: Optional dictionary to initialize the config.

        Returns:
            TranscriptConfig: The loaded transcript configuration.
        """
        if _data:
            return cls(**_data)

        data = _load_or_create(
            path,
            cls().model_dump(mode="json"),
        )
        return cls(**data)

    def save(self, path: str = "transcript_config.json") -> None:
        """Saves the transcript config to a JSON file.

        Args:
            path: Destination JSON file path.
        """
        _save(self, path)

# endregion TranscriptConfig


# region ModelPumpConfig

class ModelPumpConfig(BaseModel):
    """Configuration for a single model pump.

    Attributes:
        name: str - unique pump identifier.
        model_name: str - model path or HuggingFace identifier.
        device: str - torch device string.
        capabilities: list[str] - supported capability types.
        max_new_tokens: int - token generation limit.
        do_sample: bool - whether sampling is enabled.
        text_uses_processor: bool - whether text jobs use processor.
        use_turboquant: bool - whether to use TurboQuant for this pump.
        turboquant_bits: int - quantization bits for TurboQuant (if enabled).

        # --- transformers backend ---
        load_in_4bit: bool - enable 4-bit bnb quantization.
        load_in_8bit: bool - enable 8-bit bnb quantization.

        # --- llamacpp backend ---
        model_path: str - local path to .gguf file.
        clip_model_path: str - local path to clip .gguf for vision (llava only).
        n_gpu_layers: int - layers to offload to GPU; -1 = all.
        n_ctx: int - context window size (defaults to max_new_tokens if 0).
    """
    backend: str = "transformers"  # "transformers" or "llamacpp"

    name: str
    model_name: str
    device: str = "cpu"
    capabilities: list[str] = ["text"]
    max_new_tokens: int = 256
    do_sample: bool = False
    text_uses_processor: bool = False

    # TurboQUANNTTT
    use_turboquant: bool = False
    turboquant_bits: int = 4

    # --- transformers backend ---
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # --- llamacpp backend ---
    model_path: Optional[str] = None          # local .gguf path; if None, uses model_name via HF
    gguf_filename: Optional[str] = "*Q4_K_M.gguf"  # glob to select quant tier from HF repo
    clip_gguf_filename: Optional[str] = "mmproj-BF16.gguf"  # glob for clip model in HF repo (llava only)
    clip_model_path: Optional[str] = None
    n_gpu_layers: int = -1
    n_ctx: int = 0

# endregion ModelPumpConfig


# region MetricsConfig

class MetricsConfig(BaseModel):
    """Configuration for inference metrics collection.

    Attributes:
        enabled: bool - whether metrics are collected.
        output_path: str - output JSONL metrics file path.
    """

    enabled: bool = False
    output_path: str = "hermes_metrics.jsonl"

# endregion MetricsConfig


# region CoreConfig

class CoreConfig(BaseSettings):
    """Root configuration for the core pipeline.

    Attributes:
        vision: VisionConfig - vision subsystem config.
        text: TextConfig - text subsystem config.
        transcript: TranscriptConfig - transcript subsystem config.
        rag: Any - optional RAG configuration.
        metrics: MetricsConfig - metrics subsystem config.
        pumps: list[ModelPumpConfig] - available model pumps.
        pipeline_subscriptions: dict[str, str] - pipeline to pump map.
        vision_config_path: str - standalone vision config file path.
        text_config_path: str - standalone text config file path.
        transcript_config_path: str - standalone transcript config file path.
        rag_config_path: str - standalone RAG config file path.
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        frozen=False,
    )

    vision: VisionConfig = VisionConfig()
    text: TextConfig = TextConfig()
    transcript: TranscriptConfig = TranscriptConfig()
    rag: Any = None
    metrics: MetricsConfig = MetricsConfig()

    pumps: list[ModelPumpConfig] = []
    pipeline_subscriptions: dict[str, str] = {}

    vision_config_path: str = "vision_config.json"
    text_config_path: str = "text_config.json"
    transcript_config_path: str = "transcript_config.json"
    rag_config_path: str = "rag_config.json"

    @classmethod
    def _load_rag_config(cls, data: dict) -> Any:
        """Loads the RAG config section if available.

        Args:
            data: RAG section from the master config.

        Returns:
            Any: Loaded RAGConfig instance or raw data.
        """
        if not data:
            return None

        for module_name in ("RAG.rag.rag_config", "rag.rag_config"):
            try:
                module = import_module(module_name)
                RAGConfig = getattr(module, "RAGConfig")
                return RAGConfig(**data)
            except (ModuleNotFoundError, ImportError, AttributeError):
                continue
            except Exception:
                return data

        return data

    @classmethod
    def load(cls, path: str = MASTER_CONFIG_PATH) -> "CoreConfig":
        """Loads CoreConfig from the master JSON file.

        Args:
            path: Path to the master config file.

        Returns:
            CoreConfig: Fully loaded root configuration.
        """
        default = cls()

        if not os.path.exists(path):
            default.save(path)
            return default

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        vision = VisionConfig.load(
            path=default.vision_config_path,
            _data=data.get("vision", {}),
        )

        text = TextConfig.load(
            path=default.text_config_path,
            _data=data.get("text", {}),
        )

        transcript = TranscriptConfig.load(
            path=default.transcript_config_path,
            _data=data.get("transcript", {}),
        )

        rag = cls._load_rag_config(data.get("rag", {}))
        metrics = MetricsConfig(**data.get("metrics", {}))

        pumps = [
            ModelPumpConfig(**pump)
            for pump in data.get("pumps", [])
        ]

        pipeline_subscriptions = data.get(
            "pipeline_subscriptions",
            {},
        )

        instance = cls(
            vision=vision,
            text=text,
            transcript=transcript,
            rag=rag,
            metrics=metrics,
            pumps=pumps,
            pipeline_subscriptions=pipeline_subscriptions,
            vision_config_path=data.get(
                "vision_config_path",
                default.vision_config_path,
            ),
            text_config_path=data.get(
                "text_config_path",
                default.text_config_path,
            ),
            transcript_config_path=data.get(
                "transcript_config_path",
                default.transcript_config_path,
            ),
            rag_config_path=data.get(
                "rag_config_path",
                default.rag_config_path,
            ),
        )

        instance.save(path)
        return instance

    def save(self, path: str = MASTER_CONFIG_PATH) -> None:
        """Saves the full resolved core configuration.

        Args:
            path: Destination master JSON file path.
        """
        data = {
            "vision_config_path": self.vision_config_path,
            "text_config_path": self.text_config_path,
            "transcript_config_path": self.transcript_config_path,
            "rag_config_path": self.rag_config_path,
            "vision": self.vision.model_dump(mode="json"),
            "text": self.text.model_dump(mode="json"),
            "transcript": self.transcript.model_dump(mode="json"),
            "rag": (
                self.rag.model_dump(mode="json")
                if hasattr(self.rag, "model_dump")
                else self.rag
            ),
            "metrics": self.metrics.model_dump(mode="json"),
            "pumps": [
                pump.model_dump(mode="json")
                for pump in self.pumps
            ],
            "pipeline_subscriptions": self.pipeline_subscriptions,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

# endregion CoreConfig