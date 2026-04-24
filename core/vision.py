from dataclasses import dataclass
from typing import Any, Optional

import torch
from PIL import Image
from mss import mss

from core.config import VisionConfig


@dataclass
class VisionPipeline:
    """Pipeline for screen capture and vision model inference.

    Attributes:
        config: VisionConfig - configuration for capture and screenshot behavior.
        model: Optional[object] - vision model instance, if loaded externally.
        processor: Optional[object] - processor for preparing model inputs.
        device: Optional[str] - device identifier used for inference.
    """

    config: VisionConfig
    model: Optional[Any] = None
    processor: Optional[Any] = None
    device: Optional[str] = None

    def capture_region(self) -> Image.Image:
        """Captures a screen region defined by the vision config.

        Returns:
            Image.Image: The captured screenshot image.
        """
        with mss() as screen_capture:
            shot = screen_capture.grab(
                {
                    "left": self.config.roi_left,
                    "top": self.config.roi_top,
                    "width": self.config.roi_width,
                    "height": self.config.roi_height,
                }
            )
            image = Image.frombytes("RGB", shot.size, shot.rgb)

        if self.config.save_screenshot:
            image.save(self.config.screenshot_path)

        return image

    def build_prompt(self) -> str:
        """Builds the structured prompt for the vision model.

        Returns:
            str: The prompt used to query the vision model.
        """
        return (
            "Read the text exactly as shown in the image. "
            "The text is expected to look like X:51 Y:-697. "
            "Return only the text you see. "
            "Do not return JSON. "
            "Do not explain anything."
        )

    def infer_text(self, image: Image.Image) -> str:
        """Runs the vision model on a captured image and returns the generated text.

        Args:
            image: The image to run through the vision model.

        Returns:
            str: The model output text.

        Raises:
            RuntimeError: If the model or processor has not been initialized.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("VisionPipeline requires a loaded model and processor.")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.build_prompt()},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        device = self.device or "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )

        output_text = self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        return output_text
