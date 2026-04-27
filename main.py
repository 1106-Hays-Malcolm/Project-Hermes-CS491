from transformers import TextIteratorStreamer
import threading
import sys
import time

from web import web_app
import state
from compass_alg import update_compass
from vision_loop import run_vl_loop
from vision_loader import vl_model, vl_processor, vl_device
from language_loader import ml_model, ml_tokenizer, ml_device
from transcripts import transcript_out

# Hack needed to import RAG from the project's root
sys.path.append("RAG")
from rag.rag_api import RAGAPI

myrag = RAGAPI()

last_mission_name = ""
real_mission_name = ""
walkthrough = ""
objective_x = None
objective_y = None


# Finds the walkthrough attached to the quest name with the highest cosine similarity
def get_walkthrough(mission_name):
    results = myrag.query(mission_name)
    winner_score = 0
    true_mission_name = ""
    walkthrough = ""
    objective_x = None
    objective_y = None

    for r in results:
        current_score = r['score']
        if current_score > winner_score:
            true_mission_name = r['text']
            winner_score = current_score
            walkthrough = r['metadata'].get('Walkthrough', "")
            objective_x = r['metadata'].get('objective_x')
            objective_y = r['metadata'].get('objective_y')

    if true_mission_name == "":
        raise ValueError("Could not find quest!")

    return true_mission_name, walkthrough, objective_x, objective_y


def build_prompt(user_question, true_mission_name, walkthrough):
    # Include current location from state
    location_text = f"Current coordinates: ({state.current_x}, {state.current_y})\n"
    
    prompt = (
    "You are an LLM tasked with answering the user's questions about the video game Baldur's Gate 3. "
    "The user is asking you a question about a certain quest, and you will provide guidance. "
    "The user is probably lost, so you must provide exact step-by-step instructions.\n\n"

    "Formatting rules:\n"
    "- NEVER put all instructions in one paragraph.\n"
    "- Provide a 2–3 sentence summary.\n"
    "- Then output numbered steps.\n"
    "- Each step must begin with a number followed by a period (1., 2., 3., etc.).\n"
    "- Each step must be separated by a blank line.\n"
    "- Format example:\n\n"
    "1. First step...\n\n"
    "2. Second step...\n\n"
    "3. Third step...\n\n"
    "\n"

    f"{location_text}"
    f"The name of the quest is \"{true_mission_name}\".\n"
    f"The user's question is: {user_question}\n"
    "If the user's question is unclear or vague, ask for clarification.\n\n"
    "Walkthrough:\n"
    )

    prompt += walkthrough
    prompt += "\n\nAnswer the user directly:\n"

    return prompt


def run_flask():
    web_app.init()
    web_app.app.run()


def run_compass_loop():
    while True:
        update_compass()
        time.sleep(0.1)


def process_form_result(new_result):
    global last_mission_name
    global real_mission_name
    global walkthrough
    global objective_x
    global objective_y

    mission_name = new_result["mission-name"]
    question = new_result["question"]

    if mission_name != last_mission_name:
        print("Retrieving walkthrough...")
        real_mission_name, walkthrough, objective_x, objective_y = get_walkthrough(mission_name)
        last_mission_name = mission_name

        if objective_x is not None and objective_y is not None:
            state.objective_x = int(float(objective_x))
            state.objective_y = int(float(objective_y))
        else:
            state.objective_x = 0
            state.objective_y = 0

    prompt = build_prompt(question, real_mission_name, walkthrough)

    streamer = TextIteratorStreamer(
        ml_tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    inputs = ml_tokenizer(prompt, return_tensors="pt").to(ml_device)
    inputs = {k: v.to(ml_device) for k, v in inputs.items()}

    generation_args = {
        **inputs,
        "max_new_tokens": 500,
        "do_sample": False,
        "streamer": streamer,
    }

    # PAUSE VISION DURING LLM GENERATION 
    state.vision_running = False  # 

    generation_thread = threading.Thread(
        target=ml_model.generate,
        kwargs=generation_args,
    )

    generation_thread.start()

    response_text = ""

    for text_token in streamer:
        if text_token != "":
            response_text += text_token

    generation_thread.join()

    # RESUME VISION AFTER LLM GENERATION 
    state.vision_running = True  # 

    # Save transcript
    transcript_out(question, response_text)

    return response_text


def main():
    web_app.response_callback = process_form_result

    # Start Flask
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Start Vision Loop
    run_flag = {"run": True}
    vision_thread = threading.Thread(
        target=run_vl_loop,
        args=(vl_model, vl_processor, vl_device, run_flag),
        daemon=True
    )
    vision_thread.start()

    # Start Compass Loop
    compass_thread = threading.Thread(target=run_compass_loop, daemon=True)
    compass_thread.start()

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()

