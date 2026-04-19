from flask import Flask
from flask import request
from flask import render_template
import queue


def init():
    global app
    global result_queue


app = Flask(__name__)
result_queue = queue.Queue()

@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        mission_name = request.form.get("mission-name")
        question = request.form.get("question")
        result_queue.put({"mission-name": mission_name, "question": question})

    return render_template("index.html")
