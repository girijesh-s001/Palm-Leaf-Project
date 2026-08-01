"""
app.py - Flask web server for Palm Leaf OCR
Start: python app.py
Open:  http://localhost:5000
"""

import os, sys, base64, tempfile, json
import cv2, numpy as np
from flask import Flask, request, jsonify, render_template, Response

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from ocr_pipeline import run_ocr

app = Flask(__name__, template_folder=os.path.join(_THIS_DIR, "templates"))
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# ── numpy-safe JSON serialisation ────────────────────────────────────────────
class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalar/array types to native Python before JSON encoding."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _safe_json(data, status=200):
    """Return a Response with numpy-safe JSON serialisation."""
    body = json.dumps(data, cls=_NumpyEncoder, ensure_ascii=False)
    return Response(body, status=status, mimetype="application/json")


# ── global error handler – always return JSON, never HTML ─────────────────────
@app.errorhandler(Exception)
def _handle_any_exception(exc):
    return _safe_json({"error": str(exc)}, status=500)


def _img_to_b64(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def api_process():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    suffix = os.path.splitext(file.filename)[-1].lower() or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        result = run_ocr(tmp_path)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if result.get("error"):
        return jsonify({"error": result["error"]}), 422

    return _safe_json({
        "error":     None,
        "text":      result["text"],
        "lines":     result["lines"],
        "char_list": result["char_list"],
        "accuracy":  result["accuracy"],
        "input_img": _img_to_b64(result["image"]),
        "line_vis":  _img_to_b64(result["line_vis"]),
        "char_vis":  _img_to_b64(result["char_vis"]),
    })


if __name__ == "__main__":
    print("=" * 55)
    print(" Palm Leaf OCR - Web Interface")
    print(" Open your browser at: http://localhost:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)
