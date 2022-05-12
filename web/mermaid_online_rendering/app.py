import typing as t
import subprocess
import pathlib
import os
import json
import threading
import webbrowser

import flask
import flask_cors
import werkzeug.wrappers


CONFIG: t.Dict[str, t.Any]


try:
    with open("config.json", "r", encoding="utf-8") as f_config:
        CONFIG = json.load(f_config)

    print("Loaded CONFIG:")
    for key, val in CONFIG.items():
        print(f"{key:<32} : {val}")

except (OSError, FileNotFoundError):
    CONFIG = {}


if CONFIG.get("open_browser_tab_in_startup"):
    seconds_to_open_tab = float(CONFIG.get("seconds_to_open_browser_tab", 0.75))
    flask_port = os.environ["FLASK_PORT"]
    threading.Timer(
        seconds_to_open_tab,
        lambda: webbrowser.open_new_tab(f"http://localhost:{flask_port}"),
    ).start()


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

    width = CONFIG.get("width", 800)
    height = CONFIG.get("height", 600)
    theme = CONFIG.get("theme", "default")
    pdf_fit = CONFIG.get("pdfFit", False)
    bg_color = CONFIG.get("backgroundColor", "white")
    trim_empty_borders = CONFIG.get("trimEmptyBorders", True)

    temp_input_uri = temp_input_uri.strip()
    output_uri = output_uri.strip()

    subprocess.run([
        "./node_modules/.bin/mmdc",
        "--theme", theme,
        "--width", str(width),
        "--height", str(height),
        "--pdfFit" if pdf_fit else "",
        "--input", temp_input_uri,
        "--output", output_uri,
        "--backgroundColor", bg_color,
    ])

    if trim_empty_borders:
        subprocess.run(["pdfcrop",  output_uri, output_uri])

    return flask.make_response(dict(response="OK", status=200))


@app.route("/", methods=["GET"])
def home() -> werkzeug.wrappers.Response:
    """Render label refinement front-end."""
    return flask.redirect(flask.url_for("static", filename="index.html"))
