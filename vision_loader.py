import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)


VISION_FILE_PATH = r"./models/vision/Qwen2.5-VL-7B-Instruct"

QWEN_MODEL_NAME = VISION_FILE_PATH

VISION_DEVICE = 2


##Configure the Vision model to load in 4 bit quantized
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

##Load the vision model
vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_NAME,
    quantization_config=quant_config,
    device_map={"": VISION_DEVICE},
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

vl_model.eval()

##Load processor
vl_processor = AutoProcessor.from_pretrained(
    QWEN_MODEL_NAME,
    trust_remote_code=True
)

##Get the device the model is being run on
vl_device = next(vl_model.parameters()).device