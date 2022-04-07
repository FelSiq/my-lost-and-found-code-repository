import subprocess
import pathlib
import os

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
    output_uri = data.get("output_uri", "output.pdf")

    if not output_uri.endswith(".pdf"):
        output_uri += ".pdf"

    temp_dir = pathlib.Path("./temp")
    temp_dir.mkdir(exist_ok=True, parents=True)

    temp_input_uri = os.path.join(temp_dir, "temp.mmd")

    with open(temp_input_uri, "w", encoding="utf-8") as f_temp_out:
        f_temp_out.write(diagram_mermaid_code)

    subprocess.run([
        "./node_modules/.bin/mmdc",
        "--pdfFit",
        "--backgroundColor", "transparent",
        "--input", temp_input_uri,
        "--output", output_uri,
    ])

    return flask.make_response(dict(response="OK", status=200))


@app.route("/", methods=["GET"])
def home() -> werkzeug.wrappers.Response:
    """Render label refinement front-end."""
    return flask.redirect(flask.url_for("static", filename="index.html"))
