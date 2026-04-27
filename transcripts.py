from datetime import datetime
import os

##Prepare output transcript file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
transcript_dir = "hermes_transcripts"
os.makedirs(transcript_dir, exist_ok=True)
transcript_file = os.path.join(transcript_dir, f"{timestamp}.txt")

##Append user input and model response to transcript file
def transcript_out(user_text, response):
    with open(transcript_file, "a", encoding="utf-8") as f:
        f.write(f"User: {user_text}\n")
        f.write(f"LLM: {response}\n")