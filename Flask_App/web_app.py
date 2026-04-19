from flask import Flask
from flask import request
from flask import render_template
import queue


def init():
    global app
    global result_queue


app = Flask(__name__)
result_queue = queue.Queue()

@app.route("/", methods=["GET"])
def hello_world():
    return render_template("index.html")


@app.route("/form-submit", methods=["POST"])
def get_form_data():
    # mission_name = request.form.get("mission-name")
    # question = request.form.get("question")
    # result_queue.put({"mission-name": mission_name, "question": question})
    result_queue.put(request.get_json())
    return '', 204
