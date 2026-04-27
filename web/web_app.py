from flask import Flask
from flask import request
from flask import render_template
import queue
import os
import signal
import state
from screenshot import calibrate_roi


def init():
    global app
    global result_queue
    global new_tokens_queue


app = Flask(__name__)
result_queue = queue.Queue()
new_tokens_queue = queue.Queue()
response_callback = None


@app.route("/", methods=["GET"])
def hello_world():
    return render_template("index.html")


@app.route("/form-submit", methods=["POST"])
def get_form_data():
    # mission_name = request.form.get("mission-name")
    # question = request.form.get("question")
    # result_queue.put({"mission-name": mission_name, "question": question})

    if response_callback is not None:
        response_text = response_callback(request.get_json())
        return response_text

    result_queue.put(request.get_json())
    return '', 204


@app.route("/update-compass", methods=["GET"])
def update_compass():
    if not state.vision_running:
        return "inactive"

    return {
        "player": state.player_azimuth,
        "objective": state.objective_azimuth
    }


@app.route("/calibrate-roi", methods=["POST"])
def calibrate_roi_route():
    calibrate_roi()
    state.vision_running = True
    return '', 204


@app.route("/shutdown", methods=["POST"])
def shutdown():
    os.kill(os.getpid(), signal.SIGINT)
    return '', 204