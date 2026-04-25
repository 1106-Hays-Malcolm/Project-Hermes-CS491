from transformers import TextIteratorStreamer
import threading
import sys
import time

from Flask_App import web_app
import state
from compass_alg import update_compass
from vision_loop import run_vl_loop
from vision_loader import vl_model, vl_processor, vl_device
from language_loader import ml_model, ml_tokenizer, ml_device
from transcripts import transcript_out

from RAG.rag.rag_api import RAGAPI

myrag = RAGAPI()


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
        "The user is asking you a question about a certain quest in the game, and you will provide guidance. "
        "The user is probably lost, so you will need to provide exact instructions about where they should go next to complete the quest. "
        f"{location_text}"
        f"The name of the quest is \"{true_mission_name}\". "
        f"The user's question is {user_question}. "
        "If the user's question is unclear or vague, you must ask for clarification.\n\n"
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


def main():
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

    start_of_conversation = True

    while True:
        print("Awaiting user input...")
        new_result = web_app.result_queue.get()

        mission_name = new_result["mission-name"]
        question = new_result["question"]

        if start_of_conversation:
            print("Retrieving walkthrough...")
            real_mission_name, walkthrough, objective_x, objective_y = get_walkthrough(mission_name)

            if objective_x is not None and objective_y is not None:
                state.objective_x = int(float(objective_x))
                state.objective_y = int(float(objective_y))
            else:
                state.objective_x = 0
                state.objective_y = 0

            prompt = build_prompt(question, real_mission_name, walkthrough)
            start_of_conversation = False
        else:
            prompt = question

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

        generation_thread = threading.Thread(
            target=ml_model.generate,
            kwargs=generation_args,
        )

        generation_thread.start()

        response_text = ""

        for text_token in streamer:
            if text_token != "":
                response_text += text_token
            web_app.new_tokens_queue.put(text_token)    

        generation_thread.join()

        # Save transcript
        transcript_out(question, response_text)


if __name__ == "__main__":
    main()