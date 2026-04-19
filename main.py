from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from Flask_App import web_app
import threading
import sys
import time

# Hack needed to import RAG from the project's root
sys.path.append("RAG")
from rag.rag_api import RAGAPI

myrag = RAGAPI()

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)


print("Checking CUDA availabiltiy...")
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
   device_map="auto" if device == "cuda" else None
)

if device == "cpu":
    model.to(device)

print("Model loaded")


# Finds the walkthrough attached to the quest name with the highest cosine similarity
def get_walkthrough(mission_name):
    results = myrag.query(mission_name)
    winner_score = 0
    true_mission_name = ""
    walkthrough = ""
    for r in results:
        current_score = r['score']
        if current_score > winner_score:
            true_mission_name = r['text']
            winner_score = current_score
            walkthrough = r['metadata']['Walkthrough']

    if (true_mission_name == ""):
        raise ValueError("Could not find quest!")

    return true_mission_name, walkthrough


def build_prompt(user_question, true_mission_name, walkthrough):
    prompt = f"You are an LLM tasked with answering the user's questions about the video game Baldur's Gate 3. The user is asking you a question about a certain quest in the game, and you will provide guidance. The user is probably lost, so you will need to provide exact instructions about where they should go next to complete the quest. A walkthrough is provided so you can better guide the user through the quest. The name of the quest is \"{true_mission_name}\". The user's question is {user_question}. If the user's question is unclear or vauge, you must ask the user for clarification. If you ask the user for clarification, you must ask about details in the walkthrough to figure out what part of the quest the user is currently playing in. You must answer the user's question as if you are speaking to the user and responding to their question. You must answer the question informally in a conversational manner. You should never include phrases such as \"the user\" or \"the player\". You should refer to the player in the second person as \"you\", never \"the user\". NEVER include the phrase \"the user\" in your response. It is of utmost importance that you do not refer to the user in the third person. I will now provide the entire walkthrough for this quest:\n\n"


    prompt += walkthrough
    prompt += "\n\n\n"
    prompt += f"This is the end of the walkthrough. Now answer the user's question informally. Remember, the user's question is \"{user_question}\". Answer the user in the second person and never refer to them as \"the user\". Write your answer now:\n"

    return prompt



def run_flask():
    web_app.init()
    web_app.app.run()


def main():
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # We only include the walkthrough in the first prompt
    # Maybe we can change this later
    start_of_conversation = True

    # Listen for user input while Flask runs in the background
    while True:
        new_result = web_app.result_queue.get()
        print(new_result)

        mission_name = new_result["mission-name"]
        question = new_result["question"]
        prompt = ""

        if (start_of_conversation):
            real_mission_name, walkthrough = get_walkthrough(mission_name)
            prompt = build_prompt(question, real_mission_name, walkthrough)
            start_of_conversation = False
        else:
            prompt = question

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        generation_args = {
            **inputs,
            "max_new_tokens": 1000,
            "do_sample": False,
            "streamer": streamer,
        }

        generation_thread = threading.Thread(
            target=model.generate,
            kwargs=generation_args,
        )
        generation_thread.start()

        # https://huggingface.co/blog/aifeifei798/transformers-streaming-output

        for text_token in streamer:
            time.sleep(0.01)  # Simulate real-time output with a short delay
            if text_token != "":
                web_app.new_tokens_queue.put(text_token)

        generation_thread.join()


if __name__ == "__main__":
    main()
