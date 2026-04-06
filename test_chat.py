from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import sys
# Hack needed to import RAG from the project's root
sys.path.append("RAG")

from rag.rag_api import RAGAPI

myrag = RAGAPI()

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded")

mission_name = ""
user_question = ""
true_mission_name = ""

mission_name = input("Enter mission name (does not need to be an exact match): ")

results = myrag.query(mission_name)
for r in results:
    if true_mission_name == "":
        yes_no = ""
        while yes_no.lower().strip() != "y" and yes_no.lower().strip() != "n":
            yes_no = input(f"You are looking for quest \"{r["text"]}\", correct? (Y/N): ")

        if yes_no.lower().strip() == "y":
            true_mission_name = r["text"]
            walkthrough = r["metadata"]["Walkthrough"]

if (true_mission_name == ""):
    print("Could not find quest!")
    exit(-1)

user_question = input("Enter your prompt for the LLM: ")

prompt = f"You are an LLM tasked with answering the user's questions about the video game Baldur's Gate 3. The user is asking you a question about a certain quest in the game, and you will provide guidance. The user is probably lost, so you will need to provide exact instructions about where they should go next to complete the quest. A walkthrough is provided so you can better guide the user through the quest. The name of the quest is \"{true_mission_name}\". The user's question is {user_question}. If the user's question is unclear or vauge, you must ask the user for clarification. If you ask the user for clarification, you must ask about details in the walkthrough to figure out what part of the quest the user is currently playing in. You must answer the user's question as if you are speaking to the user and responding to their question. You must answer the question informally in a conversational manner. You should never include phrases such as \"the user\" or \"the player\". You should refer to the player in the second person as \"you\", never \"the user\". NEVER include the phrase \"the user\" in your response. It is of utmost importance that you do not refer to the user in the third person. I will now provide the entire walkthrough for this quest:\n\n"


prompt += walkthrough
prompt += "\n\n\n"
prompt += f"This is the end of the walkthrough. Now answer the user's question informally. Remember, the user's question is \"{user_question}\". Answer the user in the second person and never refer to them as \"the user\". Write your answer now:\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=1000,
    do_sample=False
)

print("\nModel output:\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
