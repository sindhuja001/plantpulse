"""
server.py — PlantPulse Flask REST API v2
=========================================
Uses shared utils.py — no feature extraction duplication.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import os
import logging
import time

from utils import decode_bytes_to_bgr, predict_image, load_model_and_scaler, MODEL_PATH

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # 5 MB

# ── Metrics ────────────────────────────────────────────────────────────────────
_stats = {"requests": 0, "errors": 0, "latency_total": 0.0}

# ── Load Artifacts ─────────────────────────────────────────────────────────────
model, scaler = None, None
try:
    model, scaler = load_model_and_scaler()
    logger.info(f"✅ Model loaded — {len(model.classes_)} classes")
    logger.info(f"{'✅' if scaler else '⚠️ '} Scaler {'loaded' if scaler else 'not found — accuracy may be lower'}")
except FileNotFoundError as e:
    logger.warning(str(e))


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "service": "PlantPulse API", "version": "2.0",
        "endpoints": ["/predict", "/health", "/classes", "/metrics"],
        "model_loaded": model is not None,
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    })


@app.route("/classes")
def classes():
    if model is None:
        return jsonify({"code": "MODEL_NOT_LOADED", "error": "Model not loaded."}), 503
    return jsonify({"count": len(model.classes_), "classes": list(model.classes_)})


@app.route("/metrics")
def metrics():
    req = _stats["requests"]
    avg = (_stats["latency_total"] / req * 1000) if req > 0 else 0
    return jsonify({
        "total_requests": req,
        "total_errors":   _stats["errors"],
        "avg_latency_ms": round(avg, 2),
    })


@app.route("/predict", methods=["POST"])
def predict():
    _stats["requests"] += 1
    t_start = time.perf_counter()

    if model is None:
        _stats["errors"] += 1
        return jsonify({"code": "MODEL_NOT_LOADED",
                        "error": "Run `python main.py` to train the model first."}), 503

    if "image" not in request.files:
        _stats["errors"] += 1
        return jsonify({"code": "NO_IMAGE", "error": "No image file in request."}), 400

    try:
        img = decode_bytes_to_bgr(request.files["image"].read())
        if img is None:
            _stats["errors"] += 1
            return jsonify({"code": "INVALID_IMAGE",
                            "error": "Cannot decode image — send a valid JPG/PNG."}), 400

        result = predict_image(img, model, scaler)
        latency = time.perf_counter() - t_start
        _stats["latency_total"] += latency

        logger.info(f"✅ {result['plant']} / {result['disease']} "
                    f"conf={result['confidence']:.1f}% {latency*1000:.0f}ms")

        return jsonify({**result, "latency_ms": round(latency * 1000, 2)})

    except ValueError as ve:
        _stats["errors"] += 1
        return jsonify({"code": "FEATURE_MISMATCH", "error": str(ve)}), 400
    except Exception as e:
        _stats["errors"] += 1
        logger.error(f"Prediction error: {e}")
        return jsonify({"code": "SERVER_ERROR", "error": str(e)}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({"code": "FILE_TOO_LARGE", "error": "Image exceeds 5 MB limit."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)