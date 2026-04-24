import json
import os

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


def _save(config: BaseSettings, path: str):
    """Writes a config object's JSON representation to disk.

    Args:
        config: Config object to save.
        path: Output JSON file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(mode="json"), f, indent=2)
