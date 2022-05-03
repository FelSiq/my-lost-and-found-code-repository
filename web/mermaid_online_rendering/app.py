import subprocess
import pathlib
import os
import json

import flask
import flask_cors
import werkzeug.wrappers


app = flask.Flask(__name__)
flask_cors.CORS(app)


@app.route("/generate-pdf", methods=["POST"])
def fn_generate_pdf():
    if flask.request.method != "POST":
        return flask.make_response(dict(response="Invalid method.", status=404))

    data = flask.request.get_json()
    diagram_mermaid_code = data["diagram"]
    output_uri = data.get("output_uri", "output.pdf").strip()

    if not output_uri.endswith(".pdf"):
        output_uri += ".pdf"

    temp_dir = pathlib.Path("./temp")
    temp_dir.mkdir(exist_ok=True, parents=True)

    temp_input_uri = os.path.join(temp_dir, "temp.mmd")

    with open(temp_input_uri, "w", encoding="utf-8") as f_temp_out:
        f_temp_out.write(diagram_mermaid_code)

    try:
        with open("config.json", "r", encoding="utf-8") as f_config:
            config = json.load(f_config)

        print("Loaded config:")
        for key, val in config.items():
            print(f"{key:<32} : {val}")

    except (OSError, FileNotFoundError):
        config = {}

    width = config.get("width", 800)
    height = config.get("height", 600)
    theme = config.get("theme", "default")
    pdf_fit = config.get("pdfFit", False)
    bg_color = config.get("backgroundColor", "white")

    subprocess.run([
        "./node_modules/.bin/mmdc",
        "--theme", theme,
        "--width", str(width),
        "--height", str(height),
        "--pdfFit" if pdf_fit else "",
        "--input", temp_input_uri.strip(),
        "--output", output_uri.strip(),
        "--backgroundColor", bg_color,
    ])

    return flask.make_response(dict(response="OK", status=200))


@app.route("/", methods=["GET"])
def home() -> werkzeug.wrappers.Response:
    """Render label refinement front-end."""
    return flask.redirect(flask.url_for("static", filename="index.html"))
