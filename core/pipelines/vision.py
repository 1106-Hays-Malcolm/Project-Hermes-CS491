from dataclasses import dataclass
from typing import Any, Optional

from PIL import Image
from mss import mss

from core.config import VisionConfig
from core.pump import AbstractModelPump, ModelPump


@dataclass
class VisionPipeline:
    """Pipeline responsible for screen capture, prompt construction, and inference.

    This class owns screen capture and prompt-building logic. All model execution
    is delegated to the subscribed ModelPump instance.

    The conversation wrapper applied around the prompt (e.g. chat templates)
    is model-specific and handled internally by the pump.

    Attributes:
        config: VisionConfig - configuration for capture and prompt formatting.
        pump: ModelPump - pump responsible for model execution.
    """

    config: VisionConfig
    pump: AbstractModelPump

    def capture_region(self) -> Image.Image:
        """Captures a screen region defined in the configuration.

        The capture region is defined by ROI (region of interest) fields in
        VisionConfig. Optionally saves the captured image if enabled.

        Returns:
            Image.Image: Captured screenshot as a PIL image.
        """
        with mss() as screen_capture:
            shot = screen_capture.grab({
                "left": self.config.roi_left,
                "top": self.config.roi_top,
                "width": self.config.roi_width,
                "height": self.config.roi_height,
            })

            image = Image.frombytes(
                "RGB",
                shot.size,
                shot.rgb,
            )

        if self.config.save_screenshot:
            image.save(self.config.screenshot_path)

        return image

    def build_prompt(self) -> str:
        """Constructs the vision prompt from the configured template.

        The template is defined in VisionConfig and may include:
            {expected_format}

        The surrounding conversation structure (chat template) is handled
        internally by the ModelPump.

        Returns:
            str: Formatted prompt string.
        """
        return self.config.prompt_template.format(
            expected_format=self.config.expected_format,
        )

    def infer_text(self, image: Image.Image) -> str:
        """Submits a blocking vision inference job to the pump.

        This method builds the prompt and delegates execution to the pump.
        The pump handles model interaction and synchronization.

        Args:
            image: Input image to process.

        Returns:
            str: Decoded model output text.
        """
        prompt = self.build_prompt()

        return self.pump.submit_vision_blocking(
            image,
            prompt,
            pipeline_type="vision",
        )