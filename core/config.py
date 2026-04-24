import json
import os
from importlib import import_module
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

MASTER_CONFIG_PATH = "core_config.json"


class VisionConfig(BaseSettings):
    """Configuration for the vision pipeline.

    Attributes:
        roi_left: int - left coordinate of the screen capture region.
        roi_top: int - top coordinate of the screen capture region.
        roi_width: int - width of the capture region.
        roi_height: int - height of the capture region.
        save_screenshot: bool - whether to save each captured image.
        screenshot_path: str - where to save the screenshot if enabled.
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

    @classmethod
    def load(cls, path: str = "vision_config.json", _data: dict = {}) -> "VisionConfig":
        """Loads a VisionConfig instance from JSON or provided data.

        Args:
            path: Path to the standalone config JSON file.
            _data: Optional dictionary to initialize the config.

        Returns:
            VisionConfig: The loaded vision configuration.
        """
        if _data:
            return cls(**_data)
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    def save(self, path: str = "vision_config.json"):
        """Saves the vision config to a JSON file.

        Args:
            path: Destination JSON file path.
        """
        _save(self, path)



class TextConfig(BaseSettings):
    """Configuration for the text model pipeline.

    Attributes:
        model_name: str - text model identifier or local path.
        max_new_tokens: int - generation token limit.
        do_sample: bool - whether to sample during generation.
    """

    model_config = SettingsConfigDict(
        json_file="text_config.json",
        json_file_encoding="utf-8",
        extra="ignore",
        frozen=False,
    )

    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_new_tokens: int = 1000
    do_sample: bool = False

    @classmethod
    def load(cls, path: str = "text_config.json", _data: dict = {}) -> "TextConfig":
        """Loads a TextConfig instance from JSON or provided data.

        Args:
            path: Path to the standalone config JSON file.
            _data: Optional dictionary to initialize the config.

        Returns:
            TextConfig: The loaded text configuration.
        """
        if _data:
            return cls(**_data)
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    def save(self, path: str = "text_config.json"):
        """Saves the text config to a JSON file.

        Args:
            path: Destination JSON file path.
        """
        _save(self, path)


class TranscriptConfig(BaseSettings):
    """Configuration for transcript logging.

    Attributes:
        directory: str - folder where transcript files are stored.
        filename_prefix: str - prefix for each transcript file.
        encoding: str - file encoding for transcript files.
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
    def load(cls, path: str = "transcript_config.json", _data: dict = {}) -> "TranscriptConfig":
        """Loads a TranscriptConfig instance from JSON or provided data.

        Args:
            path: Path to the standalone config JSON file.
            _data: Optional dictionary to initialize the config.

        Returns:
            TranscriptConfig: The loaded transcript configuration.
        """
        if _data:
            return cls(**_data)
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    def save(self, path: str = "transcript_config.json"):
        """Saves the transcript config to a JSON file.

        Args:
            path: Destination JSON file path.
        """
        _save(self, path)


class CoreConfig(BaseSettings):
    """Root configuration for the core pipeline.

    Attributes:
        vision: VisionConfig - vision subsystem config.
        text: TextConfig - text subsystem config.
        transcript: TranscriptConfig - transcript subsystem config.
        vision_config_path: str - standalone vision config file path.
        text_config_path: str - standalone text config file path.
        transcript_config_path: str - standalone transcript config file path.
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        frozen=False,
    )

    vision: VisionConfig = VisionConfig()
    text: TextConfig = TextConfig()
    transcript: TranscriptConfig = TranscriptConfig()
    rag: Any = None
    vision_config_path: str = "vision_config.json"
    text_config_path: str = "text_config.json"
    transcript_config_path: str = "transcript_config.json"
    rag_config_path: str = "rag_config.json"

    @classmethod
    def _load_rag_config(cls, data: dict) -> Any:
        """Loads the RAG config section if the RAG module is available.

        Args:
            data: The RAG section from a master config file.

        Returns:
            Any: The instantiated RAGConfig object or the raw data.
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
        """Loads CoreConfig from a master JSON file.

        Args:
            path: Path to the master config JSON file.

        Returns:
            CoreConfig: The loaded root configuration.
        """
        default = cls()
        if not os.path.exists(path):
            return default
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        vision = VisionConfig.load(path=default.vision_config_path, _data=data.get("vision", {}))
        text = TextConfig.load(path=default.text_config_path, _data=data.get("text", {}))
        transcript = TranscriptConfig.load(path=default.transcript_config_path, _data=data.get("transcript", {}))
        rag = cls._load_rag_config(data.get("rag", {}))

        return cls(
            vision=vision,
            text=text,
            transcript=transcript,
            rag=rag,
            vision_config_path=data.get("vision_config_path", default.vision_config_path),
            text_config_path=data.get("text_config_path", default.text_config_path),
            transcript_config_path=data.get("transcript_config_path", default.transcript_config_path),
            rag_config_path=data.get("rag_config_path", default.rag_config_path),
        )

    def save(self, path: str = MASTER_CONFIG_PATH):
        """Saves the full resolved core configuration to a master JSON file.

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
            "rag": self.rag.model_dump(mode="json") if hasattr(self.rag, "model_dump") else self.rag,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)



def _save(config: BaseSettings, path: str):
    """Writes a config object's JSON representation to disk.

    Args:
        config: Config object to save.
        path: Output JSON file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(mode="json"), f, indent=2)
