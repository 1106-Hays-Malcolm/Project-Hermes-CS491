from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from Flask_App import web_app
import threading
import sys

# Hack needed to import RAG from the project's root
sys.path.append("RAG")
from rag.rag_api import RAGAPI


def run_flask():
    web_app.init()
    web_app.app.run()


def main():
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Listen for user input while Flask runs in the background
    while True:
        new_result = web_app.result_queue.get()
        print(new_result)
        # Parrot back the user's question for testing
        web_app.new_tokens_queue.put(new_result["question"])


if __name__ == "__main__":
    main()
