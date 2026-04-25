from flask import Flask
from flask import request
from flask import render_template
import queue
import state


def init():
    global app
    global result_queue
    global new_tokens_queue


app = Flask(__name__)
result_queue = queue.Queue()
new_tokens_queue = queue.Queue()


@app.route("/", methods=["GET"])
def hello_world():
    return render_template("index.html")


@app.route("/get-conversation", methods=["GET"])
def send_conversation():
    new_tokens = []

    while True:
        try:
            new_tokens += new_tokens_queue.get_nowait()
        except queue.Empty:
            break

    return new_tokens


@app.route("/form-submit", methods=["POST"])
def get_form_data():
    # mission_name = request.form.get("mission-name")
    # question = request.form.get("question")
    # result_queue.put({"mission-name": mission_name, "question": question})
    result_queue.put(request.get_json())
    return '', 204


@app.route("/update-compass", methods=["GET"])
def update_compass():
    # Have to return as a string so parse on the other end.
    degrees = str(state.compass_degrees)
    return degrees