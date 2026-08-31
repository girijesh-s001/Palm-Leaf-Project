import base64
import json
import os
import sys
import tempfile
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, Response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ocr_pipeline import run_ocr

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def json_response(data, status=200):
    body = json.dumps(data, cls=NumpyEncoder, ensure_ascii=False)
    return Response(body, status=status, mimetype="application/json")


@app.errorhandler(Exception)
def handle_exception(e):
    return json_response({"error": str(e)}, status=500)


def image_to_base64(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process():
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"error": "Please provide a valid image file."}), 400

    suffix = os.path.splitext(file.filename)[-1].lower() or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        result = run_ocr(tmp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if result.get("error"):
        return jsonify({"error": result["error"]}), 422

    return json_response({
        "error": None,
        "text": result["text"],
        "lines": result["lines"],
        "char_list": result["char_list"],
        "accuracy": result["accuracy"],
        "input_img": image_to_base64(result["image"]),
        "line_vis": image_to_base64(result["line_vis"]),
        "char_vis": image_to_base64(result["char_vis"]),
    })


if __name__ == "__main__":
    print("Palm Leaf OCR Server running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)

