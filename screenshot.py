from PIL import Image
from mss import mss
import torch
import re
from pynput import mouse

REGION = {
    "left": 2250,
    "top": 400,
    "width": 150,
    "height": 50,
}


def calibrate_roi():
    result = {"x": None, "y": None}

    def on_click(x, y, button, pressed):
        if pressed:
            result["x"] = x
            result["y"] = y
            return False

    with mouse.Listener(on_click=on_click) as listener:
        listener.join()

    REGION["left"] = result["x"] - 75
    REGION["top"] = result["y"] - 25
    REGION["width"] = 150
    REGION["height"] = 50


##Take a screen shot of the desired region for inferencing. Also saves a png file for debugging
def capture_region():
    with mss() as sct:
        shot = sct.grab(REGION)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        img.save("screenshot.png")
    return img


#Prompt for qwen to run inference on the screenshot
def read_coordinates_from_screen(model, processor, device):
    image = capture_region()

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Read the coordinates exactly as shown in the image. "
                        "The text is expected to look like X:51 Y:-697. "
                        "Return only the coordinate text. "
                        "Do not return JSON. "
                        "Do not explain anything."
                    ),
                },
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

    coordinates = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    match = re.search(r"X:\s*(-?\d+)\s*Y:\s*(-?\d+)", coordinates)

    if match:
        x_coord = match.group(1)
        y_coord = match.group(2)
    else:
        x_coord = ""
        y_coord = ""

    return x_coord, y_coord