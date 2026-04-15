"""
test.py — PlantPulse Automated Test Suite
==========================================
Run with:  python test.py

Tests:
  1. utils.py imports correctly
  2. Feature extraction produces correct shape and deterministic output
  3. Model + scaler load without error
  4. predict_image() returns expected keys
  5. Flask API /health and /predict endpoints
  6. Edge cases: blank image, wrong size, corrupt bytes
"""

import sys
import numpy as np
import cv2

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))
    return condition


print("\n" + "=" * 55)
print("  PlantPulse — Test Suite")
print("=" * 55)


# ── Test 1: utils imports ─────────────────────────────────────────────────────
print("\n[1] Utils Module")
try:
    from utils import (
        extract_features, FEATURE_DIM, decode_bytes_to_bgr,
        get_disease_info, predict_image, load_model_and_scaler,
        MODEL_PATH, SCALER_PATH
    )
    check("utils.py imports OK", True)
except ImportError as e:
    check("utils.py imports OK", False, str(e))
    print("\n⛔  Cannot continue without utils.py. Stopping.\n")
    sys.exit(1)


# ── Test 2: Feature Extraction ────────────────────────────────────────────────
print("\n[2] Feature Extraction")
dummy_bgr   = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
blank_bgr   = np.zeros((8, 8, 3), dtype=np.uint8)
single_px   = np.full((1, 1, 3), 128, dtype=np.uint8)

f_dummy  = extract_features(dummy_bgr)
f_blank  = extract_features(blank_bgr)
f_single = extract_features(single_px)

check("Output shape matches FEATURE_DIM",
      f_dummy.shape == (FEATURE_DIM,), f"got {f_dummy.shape}")
check("FEATURE_DIM is 63", FEATURE_DIM == 63, f"got {FEATURE_DIM}")
check("Deterministic output (same input → same output)",
      np.allclose(extract_features(dummy_bgr), f_dummy))
check("Works on 1×1 image", f_single.shape == (FEATURE_DIM,))
check("Works on blank image (no NaN/Inf)",
      not np.any(np.isnan(f_blank)) and not np.any(np.isinf(f_blank)))
check("Histogram values sum to ~1 (normalized)",
      abs(f_dummy[:24].sum() - 1.0) < 0.01, f"sum={f_dummy[:24].sum():.4f}")


# ── Test 3: Image Decoding ────────────────────────────────────────────────────
print("\n[3] Image Decoding")
_, buf = cv2.imencode(".jpg", dummy_bgr)
valid_bytes = buf.tobytes()

decoded = decode_bytes_to_bgr(valid_bytes)
check("Valid JPEG bytes decode correctly", decoded is not None and decoded.shape[2] == 3)
check("Empty bytes → None",              decode_bytes_to_bgr(b"") is None)
check("Corrupt bytes → None",            decode_bytes_to_bgr(b"not_an_image") is None)


# ── Test 4: Disease Knowledge Base ────────────────────────────────────────────
print("\n[4] Disease Knowledge Base")
info_healthy  = get_disease_info("healthy")
info_unknown  = get_disease_info("xyzzy_disease_that_doesnt_exist")
info_blight   = get_disease_info("Late_Blight")

check("Healthy lookup works",     info_healthy["severity"] == "low")
check("Unknown disease fallback", info_unknown["severity"] == "medium")
check("Case-insensitive match",   info_blight["severity"] == "high")
check("All entries have required keys",
      all(k in info_healthy for k in ["severity", "color", "emoji", "tips"]))


# ── Test 5: Model & Scaler Loading ───────────────────────────────────────────
print("\n[5] Model & Scaler Loading")
import os
model_exists  = os.path.exists(MODEL_PATH)
scaler_exists = os.path.exists(SCALER_PATH)

if not model_exists:
    check("plant_model.pkl exists", False, "Run `python main.py` to train first")
    print("\n⚠️  Skipping predict_image tests — model not found.\n")
else:
    check("plant_model.pkl exists", True)
    check("plant_scaler.pkl exists", scaler_exists,
          "Missing — retrain with main.py for best accuracy")

    try:
        model, scaler = load_model_and_scaler()
        check("Model loads without error",  True)
        check(f"Model has n_features_in_ = {FEATURE_DIM}",
              model.n_features_in_ == FEATURE_DIM,
              f"got {model.n_features_in_}")
        check("Model has classes_ attribute", hasattr(model, "classes_") and len(model.classes_) > 0,
              f"{len(model.classes_)} classes")

        # ── Test 6: predict_image ─────────────────────────────────────────────
        print("\n[6] predict_image() Pipeline")
        result = predict_image(dummy_bgr, model, scaler)
        required_keys = ["plant", "disease", "confidence", "top5", "severity", "tips"]
        check("Returns all required keys",
              all(k in result for k in required_keys))
        check("Confidence is 0–100",
              0.0 <= result["confidence"] <= 100.0, f"{result['confidence']:.1f}%")
        check("top5 has ≤5 entries",   len(result["top5"]) <= 5)
        check("top5 probabilities sum is \u2264 100",
              sum(x["probability"] for x in result["top5"]) <= 100.5,
              f"sum={sum(x['probability'] for x in result['top5']):.1f}")
        check("severity is valid",
              result["severity"] in ["low", "medium", "high"])

        # Wrong feature dim raises ValueError
        import numpy as _np
        try:
            bad_img = _np.zeros((2, 2, 3), dtype=_np.uint8)
            # Monkey-patch model to expect wrong dim temporarily
            orig = model.n_features_in_
            model.n_features_in_ = 9999
            predict_image(bad_img, model, scaler)
            check("ValueError on feature mismatch", False, "Should have raised")
        except ValueError:
            check("ValueError on feature mismatch", True)
        finally:
            model.n_features_in_ = orig

    except Exception as e:
        check("Model/scaler load", False, str(e))


# ── Test 7: Flask API ─────────────────────────────────────────────────────────
print("\n[7] Flask API Routes")
try:
    # Import server without starting it
    import importlib, io
    import server as srv
    client = srv.app.test_client()

    r_health = client.get("/health")
    check("/health returns 200", r_health.status_code == 200)
    data = r_health.get_json()
    check("/health has 'status' key", "status" in data)

    r_home = client.get("/")
    check("/ returns 200", r_home.status_code == 200)

    # /predict with no image
    r_no_img = client.post("/predict", data={})
    check("/predict with no image → 400", r_no_img.status_code == 400)

    # /predict with corrupt image
    r_bad = client.post("/predict",
                        data={"image": (io.BytesIO(b"garbage"), "test.jpg")},
                        content_type="multipart/form-data")
    check("/predict with corrupt image → 400 or 503",
          r_bad.status_code in (400, 503))

    if model_exists:
        # /predict with valid image
        _, buf2 = cv2.imencode(".jpg", dummy_bgr)
        r_ok = client.post("/predict",
                           data={"image": (io.BytesIO(buf2.tobytes()), "leaf.jpg")},
                           content_type="multipart/form-data")
        check("/predict with valid image → 200", r_ok.status_code == 200)
        if r_ok.status_code == 200:
            d = r_ok.get_json()
            check("Response has plant/disease/confidence",
                  all(k in d for k in ["plant", "disease", "confidence"]))

except Exception as e:
    check("Flask API tests", False, str(e))


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
print(f"  Results: {passed} passed, {failed} failed out of {len(results)} tests")
if failed == 0:
    print("  🎉 All tests passed!")
else:
    print("  ⚠️  Fix the failing tests before deploying.")
print("=" * 55 + "\n")

sys.exit(0 if failed == 0 else 1)