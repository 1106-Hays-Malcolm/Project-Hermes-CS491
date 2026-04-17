from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        print(request.form.get("mission-name"))
        print(request.form.get("question"))

    return render_template("index.html")
