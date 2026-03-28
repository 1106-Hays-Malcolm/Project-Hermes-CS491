import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime
import os
import re
import time
import torch
from PIL import Image
from mss import mss
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)
import threading #######

#Global for VL LM Logic
vl_pause = False
vl_started = False #######

#File path for LMs
QWEN_MODEL_NAME = r"C:\LLM_Project\QWEN\models\qwen2.5-vl-7b"
MISTRAL_MODEL_NAME = r"C:\LLM_Project\Mistral\model"

#prep output transcript file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
transcript_dir = "hermes_transcripts"
os.makedirs(transcript_dir, exist_ok=True)
transcript_file = os.path.join(transcript_dir, f"{timestamp}.txt")

#define ROI
REGION = {
    "left": 500,
    "top": 500,
    "width": 100,
    "height": 100,
}

#Quant Config to 4 bit for both models
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

#Load both models
print("Loading QWEN VL model...")
vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
)
vl_processor = AutoProcessor.from_pretrained(QWEN_MODEL_NAME)
vl_device = next(vl_model.parameters()).device #######

print("Loading Mistral ML model...")
ml_tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_NAME)
ml_model = AutoModelForCausalLM.from_pretrained(
    MISTRAL_MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
)
ml_device = next(ml_model.parameters()).device #######

print("Models loaded.")

#Screenshot function
def capture_region():
    with mss() as sct:
        shot = sct.grab(REGION)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        img.save("screenshot.png")
    return img

#VL prompt
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

def ask_ml_model(user_text, selected_map): #######
    prompt = f"Current map: {selected_map}\nUser query: {user_text}" #######
    inputs = ml_tokenizer(prompt, return_tensors="pt").to(ml_device) #######
    with torch.no_grad(): #######
        output_ids = ml_model.generate( #######
            **inputs, #######
            max_new_tokens=100, #######
            do_sample=False, #######
        ) #######
    response = ml_tokenizer.decode(output_ids[0], skip_special_tokens=True) #######
    return response #######

def run_vl_loop(): #######
    global vl_pause, vl_started #######
    while True: #######
        if vl_started and not vl_pause: #######
            image = capture_region() #######
            vl_response = ask_model(vl_model, vl_processor, vl_device, image) #######
            print(vl_response) #######
        time.sleep(1) #######

#Append transcript to transcript file
def transcript_out(user_text, response):
    with open(transcript_file, "a", encoding="utf-8") as f:
        f.write(f"User: {user_text}\n")
        f.write(f"LLM: {response}\n")

#end the program
def end_session():
    print("Setup Complete.")
    root.destroy()

#User input in UI text box to Mistral
def send_query():
    global vl_pause, vl_started #######
    # make sure user selects a map
    if map_choice.get() == "":
        output_box.config(state=tk.NORMAL)
        output_box.insert(tk.END, "LLM: Please select a map before sending a query.\n")
        output_box.config(state=tk.DISABLED)
        output_box.yview(tk.END)
        return

    user_input = input_box.get("1.0", tk.END).strip()
    selected_map = map_choice.get() #######

    if user_input == "":
        return

    vl_pause = True #######

    ml_response = ask_ml_model(user_input, selected_map) #######

    vl_started = True #######
    vl_pause = False #######

    transcript_out(user_input, ml_response) #######

    output_box.config(state=tk.NORMAL)
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, f"Mistral: {ml_response}\n")
    output_box.config(state=tk.DISABLED)
    output_box.yview(tk.END)

    input_box.delete("1.0", tk.END)

threading.Thread(target=run_vl_loop, daemon=True).start() #######

#UI setup I will create a seperate function for this  to be cleaner later
root = tk.Tk()
root.title("Hermes V1")
root.geometry("550x600")

output_label = tk.Label(root, text="LLM Response:")
output_label.pack(anchor="w", padx=10, pady=(10, 0))

output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=15)
output_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
output_box.config(state=tk.DISABLED)

input_label = tk.Label(root, text="User Query:")
input_label.pack(anchor="w", padx=10, pady=(0, 0))

input_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=10)
input_box.pack(padx=10, pady=5, fill=tk.BOTH)

# label above map choices
map_label = tk.Label(root, text="Pick your current Map:")
map_label.pack(anchor="w", padx=10, pady=(5, 0))

# only one map can be selected at a time
map_choice = tk.StringVar(value="")

map_frame = tk.Frame(root)
map_frame.pack(padx=10, pady=5, anchor="w")

map_one = tk.Radiobutton(map_frame, text="Act One", variable=map_choice, value="Act One")
map_one.pack(side=tk.LEFT, padx=5)

map_two = tk.Radiobutton(map_frame, text="Act Two", variable=map_choice, value="Act Two")
map_two.pack(side=tk.LEFT, padx=5)

map_three = tk.Radiobutton(map_frame, text="Act Three", variable=map_choice, value="Act Three")
map_three.pack(side=tk.LEFT, padx=5)

map_four = tk.Radiobutton(map_frame, text="Act Four", variable=map_choice, value="Act Four")
map_four.pack(side=tk.LEFT, padx=5)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

send_button = tk.Button(button_frame, text="Send", command=send_query)
send_button.pack(side=tk.LEFT, padx=5)

end_button = tk.Button(button_frame, text="End Session", command=end_session)
end_button.pack(side=tk.LEFT, padx=5)

root.mainloop()