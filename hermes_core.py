import threading
import time

from Flask_App import web_app
from core.core_api import CoreAPI
from RAG.rag.rag_api import RAGAPI

# region Config Bootstrap

# from core.config import CoreConfig, ModelPumpConfig

# # Bootstrap a minimal config if none exists
# config = CoreConfig.load()

# if not config.pumps:
#     print("[Bootstrap] No pumps found. Creating minimal text config...")

#     config.pumps = [
#         ModelPumpConfig(
#             name="text_pump",
#             model_name="google/gemma-4-E2B-it",  # small enough for 8GB
#             device="cuda",
#             capabilities=["text"],
#             max_new_tokens=256,
#             do_sample=True,
#         )
#     ]

#     config.pipeline_subscriptions = {
#         "text": "text_pump"
#     }

#     config.save()

# core_api = CoreAPI.create(config=config)

# endregion Config Bootstrap

core_api = CoreAPI.create()


def run_flask() -> None:
    """Initializes and runs the Flask web application.

    This sets up the web interface and starts the Flask server loop.
    """
    web_app.init()
    web_app.app.run()


def update_compass() -> None:
    """Continuously updates a rotating compass value for the UI.

    The compass value cycles from 0 to 360 degrees over a fixed period
    and is written directly to the shared web_app state.
    """
    # start_time = time.time()

    while True:
        # elapsed = time.time() - start_time
        # web_app.compass_degrees = (elapsed % 60.0) / 60.0 * 360.0
        time.sleep(1)


def process_user_input(new_result: dict) -> None:
    """Processes a single user input payload from the web queue.

    This function extracts the user question, ensures session state is
    initialized, streams a response from the text model, and logs the
    interaction to the transcript.

    Args:
        new_result: Dictionary payload from the web layer containing
            user input data. Expected key: "question".
    """
    question = new_result.get("question", "")
    print(f"[Debug] Question: {question}")

    if core_api.session.selected_map is None:
        core_api.set_map("Act One")

    print(f"[Debug] Selected map: {core_api.session.selected_map}")
    print(f"[Debug] Text pipeline: {core_api.text_pipeline}")

    streamer = core_api.query_text_model(question)
    print(f"[Debug] Streamer: {streamer}")

    full_response: list[str] = []

    for token in streamer:
        print(f"[Debug] Token: {repr(token)}")
        if token:
            web_app.new_tokens_queue.put(token)
            full_response.append(token)
            time.sleep(0.005)

    print(f"[Debug] Full response: {''.join(full_response)}")
    core_api.log_transcript(
        question,
        "".join(full_response),
    )

def run_vision_loop() -> None:
    core_api.session.start_visual_loop()
    while True:
        if core_api.is_visual_loop_active():
            try:
                coords = core_api.capture_coordinates()
                print(f"[VisionLoop] Captured coordinates: {coords}")
                # parse from X:51 Y:-697 to {51, -697}
                if not coords or "X:" not in coords or "Y:" not in coords:
                    raise ValueError(f"Malformed coords: {coords}")
                parts = coords.replace("X:", "").replace("Y:", "").split()
                if len(parts) != 2:
                    raise ValueError(f"Bad coord format: {coords}")
                x, y = map(int, parts)
                if (abs(x) + abs(y)) == 0:
                    print("[VisionLoop] Skipping (0,0)")
                    continue
                # for now angle from {0,0} to {x,y}
                import math
                web_app.compass_degrees = (math.degrees(math.atan2(x, y)) + 360) % 360                # web_app.latest_coords = coords # TODO: add this junk later
            except Exception as e:
                print(f"[VisionLoop] Error: {e}")
        time.sleep(core_api.config.vision.capture_interval_seconds)


def main() -> None:
    """Starts the application and coordinates background workers.

    This function:
        - launches the Flask server in a background thread
        - starts the UI compass updater thread
        - listens for incoming user input from the web queue
        - processes each request sequentially

    The application runs indefinitely until interrupted.
    """
    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True,
    )
    flask_thread.start()

    compass_thread = threading.Thread(
        target=update_compass,
        daemon=True,
    )
    compass_thread.start()

    vision_thread = threading.Thread(
        target=run_vision_loop,
        daemon=True,
    )
    vision_thread.start()

    core_api.session.start_visual_loop()

    try:
        while True:
            print("Awaiting user input...")
            print("Go here to run flask: " + "http://127.0.0.1:5000/" )
            new_result = web_app.result_queue.get()
            print("Got user input from Flask:", new_result)

            process_user_input(new_result)

    except KeyboardInterrupt:
        print("Shutting down...")
        core_api.shutdown()


if __name__ == "__main__":
    main()