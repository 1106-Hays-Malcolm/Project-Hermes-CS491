import torch

## Build the player prompt to send to the language model with current quest, current location and the user input query
def build_language_prompt(user_text, current_quest, current_location):
    prompt = (
        f"Current quest: {current_quest}\n"
        f"Current location: {current_location}\n"
        f"User query: {user_text}"
    )
    return prompt

##Convert the prompt into tokens for the model  and sends to model on chosen GPU.
def tokenize_language_prompt(tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    return inputs

##Run inference on Mistral and recieve tokens. Convert tokens back to usable text response. Return text response
def generate_language_response(model, tokenizer, inputs):
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

#Combine functions into one usable function call for userinput and generated response
def ask_ml_model(user_text, current_quest, current_location, model, tokenizer, device):
    prompt = build_language_prompt(user_text, current_quest, current_location)
    inputs = tokenize_language_prompt(tokenizer, prompt, device)
    response = generate_language_response(model, tokenizer, inputs)

    return response