# This version runs the full model of QWEN2.5-VL-7B using the bfloat16 precision
# This version uploads from local drive
# This version includes the time feature for metrics

# Libraries used for regex parsing
import re

# Libraries used for model inference, image handling, and screen capture
import torch
from PIL import Image
from mss import mss
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import time

# Container for the Hugging Face model used in this code iteration
#Now loaded locally 
MODEL_NAME = r"C:\LLM_Project\models\qwen2.5-vl-7b"

# Region Of Interest ROI defines the screen area to capture
# "left" and "top" define the starting position
# "width" and "height" define the size of the capture area
REGION = {
    "left": 500,
    "top": 500,
    "width": 100,
    "height": 100,
}

# Capture specified screen region and return as image
def capture_region():
    # initialize mss screen capture and cleanup after capture
    with mss() as sct:
        shot = sct.grab(REGION)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
    return img

# Load the vision-language model and processor, and configure device settings
def load_model():
    # if the gpu is available run on the gpu much faster than cpu
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    dtype = torch.bfloat16 if has_cuda else torch.float32

    # Load the model with appropriate precision and device configuration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if has_cuda else None,
    )

    # move model to cpu if gpu is not available
    if not has_cuda:
        model = model.to(device)

    # load processor that prepares images and text in expected format
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    return model, processor, device

# Run inference on the provided image using the loaded model
def ask_model(model, processor, device, image):
    # Define structured prompt with image and instruction for the model
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    # Prompt instructs model to extract only coordinate text
                    "text": (
                        'Read the text exactly as shown in the image. '
                        'The text is expected to look like X:51 Y:-697. '
                        'Return only the text you see. '
                        'Do not return JSON. '
                        'Do not explain anything.'
                    ),
                },
            ],
        }
    ]

    # Convert conversation structure into model-ready text prompt format
    text_prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Prepare model inputs text and image and convert to tensors
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Disable gradient tracking since this is inference
    with torch.no_grad():
        # Generate model output tokens based on the input prompt and image
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

    # Convert generated token IDs into readable text
    output_text = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    return output_text

# Extract X and Y coordinate values from model output text and return as JSON string
def normalize_to_json(text):
    # Search for X and Y values using regex
    x_match = re.search(r"X\s*:\s*(-?\d+)", text, re.IGNORECASE)
    y_match = re.search(r"Y\s*:\s*(-?\d+)", text, re.IGNORECASE)

    # Extract matched values; default to empty string if not found
    x = x_match.group(1) if x_match else ""
    y = y_match.group(1) if y_match else ""

    # Return JSON string with no spaces for consistent processing
    return f'{{"x":"{x}","y":"{y}"}}'

# Main workflow:
# 1. Load model and processor
# 2. Capture screen region
# 3. Run model inference
# 4. Normalize output to JSON
def main():
    print("Loading model...")

    load_start = time.time()
    model, processor, device = load_model()
    load_end = time.time()
    print(f"Model load time: {load_end - load_start:.3f}s")
    print()

    total_start = time.time()

    print("Capturing screen region...")
    t1 = time.time()
    image = capture_region(REGION, IMAGE_PATH)
    t2 = time.time()

    print("Running inference...")
    raw = ask_model(model, processor, device, image)
    t3 = time.time()

    print("\n\n" + normalize_to_json(raw) + "\n\n")
    t4 = time.time()

    print(f"Capture time: {t2 - t1:.3f}s")
    print(f"Inference time: {t3 - t2:.3f}s")
    print(f"Output time: {t4 - t3:.3f}s")
    print(f"Total time: {t4 - total_start:.3f}s")

if __name__ == "__main__":
    main()