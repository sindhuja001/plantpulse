"""
utils.py — PlantPulse Shared Utilities
=======================================
Single source of truth for:
  • Feature extraction  (used by main.py, app.py, server.py)
  • Artifact paths
  • Disease knowledge base
  • Image decoding helpers

RULE: Any change to extract_features() is made HERE only.
      All other files import from this module.
"""

import cv2
import numpy as np
import os
from typing import Optional

# ── Artifact paths (one place to change if you move files) ────────────────────
MODEL_PATH  = "plant_model.pkl"
SCALER_PATH = "plant_scaler.pkl"
REPORT_PATH = "training_report.txt"
IMG_SIZE    = (128, 128)

# Feature-space identifiers
FEATURE_MODE_RAW  = "raw_pixels"    # old model: 128×128×3 = 49152 dims
FEATURE_MODE_HIST = "histogram"     # new model: 63 dims
RAW_PIXEL_DIM     = 128 * 128 * 3  # = 49152


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Deterministic, normalized feature vector (63 dims).

    Feature layout:
      [0:24]  — Hue histogram, 24 bins, normalized to sum=1
      [24:40] — Saturation histogram, 16 bins, normalized
      [40:56] — Value histogram, 16 bins, normalized
      [56:59] — BGR channel means  (divided by 255)
      [59:62] — BGR channel stds   (divided by 255)
      [62]    — Canny edge density (0–1)

    Args:
        img: BGR image as uint8 numpy array (any size).
    Returns:
        1-D float64 array of length 63.
    """
    # Fix distribution shift: dataset was 8x8, bottleneck all inputs to match!
    img   = cv2.resize(img, (8, 8))
    img   = cv2.resize(img, IMG_SIZE)
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color histograms — normalized to unit sum
    h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    h_hist /= (h_hist.sum() + 1e-7)
    s_hist /= (s_hist.sum() + 1e-7)
    v_hist /= (v_hist.sum() + 1e-7)

    # Per-channel mean & std (scaled 0–1)
    means, stds = cv2.meanStdDev(img)
    stats = np.concatenate([means.flatten(), stds.flatten()]) / 255.0

    # Edge density
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges > 0)) / (IMG_SIZE[0] * IMG_SIZE[1])

    return np.concatenate([h_hist, s_hist, v_hist, stats, [edge_density]])


FEATURE_DIM = len(extract_features(np.zeros((8, 8, 3), dtype=np.uint8)))  # = 63


def extract_features_raw(img: np.ndarray) -> np.ndarray:
    """
    Legacy extractor — raw pixel flatten (128×128×3 = 49152 dims).
    Used automatically when the loaded model was trained this way.
    DO NOT use for new training; use extract_features() instead.
    """
    return cv2.resize(img, IMG_SIZE).flatten().astype(np.float64)


def get_feature_mode(model) -> str:
    """
    Inspect a loaded model and return which feature extractor it was trained with.

    Returns:
        'raw_pixels'  — model.n_features_in_ == 49152  (old pipeline)
        'histogram'   — model.n_features_in_ == 63     (new pipeline)

    Raises:
        ValueError if the feature count is unrecognised.
    """
    n = model.n_features_in_
    if n == RAW_PIXEL_DIM:
        return FEATURE_MODE_RAW
    if n == FEATURE_DIM:
        return FEATURE_MODE_HIST
    raise ValueError(
        f"Unrecognised model feature count: {n}. "
        f"Expected {RAW_PIXEL_DIM} (old) or {FEATURE_DIM} (new). "
        f"Retrain with `python main.py`."
    )


def extract_for_model(img: np.ndarray, model) -> np.ndarray:
    """
    Extract features in whichever space the model was trained in.
    Removes the need for callers to know which mode is active.
    """
    mode = get_feature_mode(model)
    if mode == FEATURE_MODE_RAW:
        return extract_features_raw(img).reshape(1, -1)
    return extract_features(img).reshape(1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE DECODING
# ═══════════════════════════════════════════════════════════════════════════════
def decode_bytes_to_bgr(raw_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode raw image bytes → BGR ndarray.
    Returns None if bytes are empty or decoding fails.
    """
    if not raw_bytes:
        return None
    arr = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img  # None on failure


def decode_file_to_bgr(path: str) -> Optional[np.ndarray]:
    """Read an image file from disk → BGR ndarray."""
    return cv2.imread(path, cv2.IMREAD_COLOR)


# ═══════════════════════════════════════════════════════════════════════════════
# DISEASE KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════
DISEASE_INFO: dict[str, dict] = {
    "healthy": {
        "severity": "low",
        "color":    "#10b981",
        "emoji":    "🌱",
        "tips":     "No treatment needed. Maintain regular watering and sunlight.",
    },
    "early_blight": {
        "severity": "medium",
        "color":    "#f59e0b",
        "emoji":    "🟡",
        "tips":     "Remove affected leaves. Apply copper-based fungicide. Avoid overhead watering.",
    },
    "late_blight": {
        "severity": "high",
        "color":    "#ef4444",
        "emoji":    "🔴",
        "tips":     "Isolate plant immediately. Apply mancozeb or chlorothalonil. Destroy infected tissue.",
    },
    "leaf_mold": {
        "severity": "medium",
        "color":    "#f97316",
        "emoji":    "🟠",
        "tips":     "Improve air circulation. Apply fungicide. Reduce ambient humidity.",
    },
    "bacterial_spot": {
        "severity": "high",
        "color":    "#ef4444",
        "emoji":    "🔴",
        "tips":     "Use copper-based bactericide. Avoid working with wet plants.",
    },
    "common_rust": {
        "severity": "medium",
        "color":    "#f97316",
        "emoji":    "🟠",
        "tips":     "Apply triazole fungicide early. Rotate crops next season.",
    },
    "northern_leaf_blight": {
        "severity": "high",
        "color":    "#ef4444",
        "emoji":    "🔴",
        "tips":     "Apply fungicide at first sign. Use resistant varieties next cycle.",
    },
    "gray_leaf_spot": {
        "severity": "medium",
        "color":    "#f59e0b",
        "emoji":    "🟡",
        "tips":     "Improve drainage. Apply strobilurin fungicide preventively.",
    },
    "powdery_mildew": {
        "severity": "medium",
        "color":    "#f59e0b",
        "emoji":    "🟡",
        "tips":     "Apply sulfur or potassium bicarbonate spray. Ensure good airflow.",
    },
    "target_spot": {
        "severity": "medium",
        "color":    "#f97316",
        "emoji":    "🟠",
        "tips":     "Remove infected leaves. Apply chlorothalonil or mancozeb.",
    },
    "mosaic_virus": {
        "severity": "high",
        "color":    "#ef4444",
        "emoji":    "🔴",
        "tips":     "No cure — remove and destroy infected plants. Control aphid vectors.",
    },
    "yellow_leaf_curl_virus": {
        "severity": "high",
        "color":    "#ef4444",
        "emoji":    "🔴",
        "tips":     "Remove infected plants. Use reflective mulches to deter whiteflies.",
    },
}

_FALLBACK_INFO = {
    "severity": "medium",
    "color":    "#f59e0b",
    "emoji":    "⚠️",
    "tips":     "Consult an agronomist for targeted treatment advice.",
}


def get_disease_info(disease: str) -> dict:
    """
    Lookup disease metadata by fuzzy key match.
    Falls back gracefully if disease is unknown.

    Args:
        disease: Raw disease string (e.g. 'Early_blight', 'Late blight').
    Returns:
        Dict with keys: severity, color, emoji, tips.
    """
    key = disease.lower().replace(" ", "_")
    for k, v in DISEASE_INFO.items():
        if k in key or key in k:
            return v
    return _FALLBACK_INFO


# ═══════════════════════════════════════════════════════════════════════════════
# ARTIFACT LOADING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def load_model_and_scaler():
    """
    Load plant_model.pkl and plant_scaler.pkl from disk.
    Returns (model, scaler). scaler may be None if not found.
    Raises FileNotFoundError if model is missing.
    """
    import joblib
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run `python main.py` to train."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    return model, scaler


def predict_image(img_bgr: np.ndarray, model, scaler=None) -> dict:
    """
    Full prediction pipeline for a BGR image.
    Auto-detects whether the model expects raw pixels (49152) or
    histogram features (63) and extracts accordingly.

    Returns dict with keys:
        plant, disease, confidence, prediction_raw, top5,
        severity, tips, color, emoji, feature_mode
    """
    # ── Auto-detect feature space ────────────────────────────────────────────
    mode     = get_feature_mode(model)   # raises ValueError on unknown dim
    features = extract_for_model(img_bgr, model)  # shape (1, n_features)

    # Scaler only applies to histogram-trained models (raw-pixel models
    # have no associated scaler in the old pipeline)
    if scaler is not None and mode == FEATURE_MODE_HIST:
        features = scaler.transform(features)

    prediction  = model.predict(features)[0]
    conf_probs  = model.predict_proba(features)[0]
    confidence  = float(np.max(conf_probs) * 100)

    try:
        plant, disease = prediction.split("___")
    except ValueError:
        plant, disease = "Unknown", prediction

    info = get_disease_info(disease)

    top5 = sorted(
        [{"class": c, "probability": round(float(p) * 100, 2)}
         for c, p in zip(model.classes_, conf_probs)],
        key=lambda x: -x["probability"]
    )[:5]

    return {
        "plant":        plant,
        "disease":      disease,
        "confidence":   round(confidence, 2),
        "prediction_raw": prediction,
        "top5":         top5,
        "severity":     info["severity"],
        "tips":         info["tips"],
        "color":        info["color"],
        "emoji":        info["emoji"],
        "feature_mode": mode,   # 'raw_pixels' or 'histogram'
    }
