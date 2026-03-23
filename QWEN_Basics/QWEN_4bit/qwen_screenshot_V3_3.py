# This version runs the 4bit quantized model of QWEN2.5-VL-7B 
# This version uploads from local drive

# Libraries used for regex parsing
import re

# Libraries used for model inference, image handling, and screen capture
import torch
from PIL import Image
from mss import mss
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

# Container for the Hugging Face model used in this code iteration
#Now loaded locally 
MODEL_NAME = r"C:\LLM_Project\QWEN\models\qwen2.5-vl-7b"

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

        #save the screenshot to current folder
        img.save("screenshot.png")
        
    return img

# Load the vision-language model and processor, and configure device settings
def load_model():
    # if the gpu is available run on the gpu much faster than cpu
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"

    # configure 4-bit quantization to reduce more memory usage
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Load the model with appropriate precision (8bit) and device configuration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
    )

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
    model, processor, device = load_model()
    image = capture_region()
    raw = ask_model(model, processor, device, image)
    print(normalize_to_json(raw))

if __name__ == "__main__":
    main()