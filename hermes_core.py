import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from Flask_App import web_app
from core.core_api import CoreAPI


# This file is the new root launcher for the Hermes pipeline.
# It wires the Flask app result queue into a simple processing loop
# and preserves legacy For_Git hermes functionality as commented-out
# reference code until the Flask UI supports it.

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Checking CUDA availability...")
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

print("Loading model...")
text_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)

if device == "cpu":
    # Pylance WILL just eternally complain about this but it doesn't break anything :p
    text_model.to(device)

print("Model loaded")

core_api = CoreAPI.create(
    text_tokenizer=tokenizer,
    text_model=text_model,
    text_device=device,
)


def run_flask() -> None:
    """Initializes and runs the Flask front end."""
    web_app.init()
    web_app.app.run()


def update_compass() -> None:
    """Continuously updates the Flask compass value for the UI."""
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        web_app.compass_degrees = (elapsed % 60.0) / 60.0 * 360.0
        time.sleep(1)


def process_user_input(new_result: dict) -> None:
    """Processes a single user input payload from the Flask queue."""
    question = new_result.get("question", "")

    if core_api.session.selected_map is None:
        core_api.set_map("Act One")

    streamer, generation_thread = core_api.text_pipeline.stream_infer(
        question, core_api.session.selected_map
    )

    full_response = []
    for token in streamer:
        if token:
            web_app.new_tokens_queue.put(token)
            full_response.append(token)
            time.sleep(0.005)

    generation_thread.join()
    core_api.log_transcript(question, "".join(full_response))


# Legacy Hermes pipeline snippets from For_Git/hermes_v1.py.
# These are kept for future reference and eventual wiring.
#
# def capture_region():
#     with mss() as sct:
#         shot = sct.grab(REGION)
#         img = Image.frombytes("RGB", shot.size, shot.rgb)
#         img.save("screenshot.png")
#     return img
#
# def ask_model(model, processor, device, image):
#     conversation = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image"},
#                 {
#                     "type": "text",
#                     "text": (
#                         'Read the text exactly as shown in the image. '
#                         'The text is expected to look like X:51 Y:-697. '
#                         'Return only the text you see. '
#                         'Do not return JSON. '
#                         'Do not explain anything.'
#                     ),
#                 },
#             ],
#         }
#     ]
#     ...
#
# def ask_ml_model(user_text, selected_map):
#     prompt = f"Current map: {selected_map}\nUser query: {user_text}"
#     inputs = ml_tokenizer(prompt, return_tensors="pt").to(ml_device)
#     with torch.no_grad():
#         output_ids = ml_model.generate(**inputs, max_new_tokens=100, do_sample=False)
#     return ml_tokenizer.decode(output_ids[0], skip_special_tokens=True)
#
# def transcript_out(user_text, response):
#     with open(transcript_file, "a", encoding="utf-8") as f:
#         f.write(f"User: {user_text}\n")
#         f.write(f"LLM: {response}\n")
#
# def run_vl_loop():
#     global vl_pause, vl_started
#     while True:
#         if vl_started and not vl_pause:
#             image = capture_region()
#             vl_response = ask_model(vl_model, vl_processor, vl_device, image)
#             print(vl_response)
#         time.sleep(1)
#
# def send_query():
#     global vl_pause, vl_started
#     if map_choice.get() == "":
#         output_box.insert(tk.END, "LLM: Please select a map before sending a query.\n")
#         return
#     user_input = input_box.get("1.0", tk.END).strip()
#     selected_map = map_choice.get()
#     vl_pause = True
#     ml_response = ask_ml_model(user_input, selected_map)
#     vl_started = True
#     vl_pause = False
#     transcript_out(user_input, ml_response)
#     output_box.insert(tk.END, f"Mistral: {ml_response}\n")
#     input_box.delete("1.0", tk.END)


def main() -> None:
    """Starts the Flask app, compass updater, and listens for Flask input."""
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    compass_thread = threading.Thread(target=update_compass, daemon=True)
    compass_thread.start()


    while True:
        print("Awaiting user input...")
        new_result = web_app.result_queue.get()
        print("Got user input from Flask:", new_result)
        process_user_input(new_result)


if __name__ == "__main__":
    main()
