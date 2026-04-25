import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


LANGUAGE_FILE_PATH = r"./models/language/Mistral-7B-Instruct-v0.3"

MISTRAL_MODEL_NAME = LANGUAGE_FILE_PATH

LANGUAGE_DEVICE = 1


##Configure the language model to load as 4bit quantized version
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

##From transformers library to load the tokenizer for the language model
ml_tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_NAME)

##Load the language model from the file path, apply 4 bit quantization, and make available on gpu
ml_model = AutoModelForCausalLM.from_pretrained(
    MISTRAL_MODEL_NAME,
    quantization_config=quant_config,
    device_map={"": LANGUAGE_DEVICE},
)

ml_model.eval()

##Get the device the model is being run on.
ml_device = next(ml_model.parameters()).device