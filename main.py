"""
main.py — PlantPulse Training Pipeline v2
==========================================
Usage:
  python main.py          # Full training with GridSearchCV (best accuracy)
  python main.py --fast   # Quick training, no grid search (done in minutes)

Fixes vs old training:
  - Uses 63-dim histogram features instead of 49152 raw pixels
  - Fits StandardScaler (saves plant_scaler.pkl)
  - Parallel feature extraction across all CPU cores
  - Stratified split so all classes are represented in test set
"""

import sys

# Force UTF-8 encoding for standard output to avoid Windows console errors with emojis
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import os
import joblib
import random
import time
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv

from utils import extract_features, MODEL_PATH, SCALER_PATH, REPORT_PATH, FEATURE_DIM

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
SEED      = 42
N_JOBS    = -1
FAST_MODE = "--fast" in sys.argv   # skip GridSearchCV for quick iteration

np.random.seed(SEED)
random.seed(SEED)


def _safe_extract(img: np.ndarray) -> np.ndarray:
    try:
        # dataset is in RGB, but extract_features expects BGR
        img_bgr = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2BGR)
        return extract_features(img_bgr)
    except Exception:
        return np.zeros(FEATURE_DIM)


# ── Load Dataset ───────────────────────────────────────────────────────────────
t0 = time.time()
print("=" * 55)
print(f"  PlantPulse — Training Pipeline ({'FAST' if FAST_MODE else 'FULL'})")
print("=" * 55)

try:
    images = np.load("images.npy")
    labels = np.load("labels.npy")
    print(f"\n📂 Loaded {len(images)} samples | {len(set(labels))} classes")
    print(f"   Classes: {sorted(set(labels))}")
except FileNotFoundError as e:
    print(f"\n❌ Dataset not found: {e}")
    print("   Ensure images.npy and labels.npy exist in the project directory.")
    sys.exit(1)

# ── Parallel Feature Extraction ───────────────────────────────────────────────
print(f"\n🧠 Extracting {FEATURE_DIM}-dim histogram features (parallel)...")
t1 = time.time()
X = np.array(
    Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_safe_extract)(img) for img in images
    )
)
y = labels
print(f"   ✅ Done in {time.time()-t1:.1f}s | shape: {X.shape}")
print(f"   ℹ️  Old model used {128*128*3} features — new uses {FEATURE_DIM} (better!)")

# ── Fit & Save Scaler ─────────────────────────────────────────────────────────
print(f"\n📐 Fitting StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)
print(f"   ✅ Scaler saved → {SCALER_PATH}")

# ── Stratified Train/Test Split ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

# ── Train Model ───────────────────────────────────────────────────────────────
if FAST_MODE:
    # Quick training with sensible defaults
    print(f"\n🚀 Fast training (n_estimators=200, max_depth=20)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        random_state=SEED,
        n_jobs=N_JOBS
    )
    model.fit(X_train, y_train)
    print(f"   ✅ Done in {time.time()-t1:.1f}s")
else:
    # Full grid search (takes longer but finds best model)
    print(f"\n🔧 GridSearchCV (may take 5-15 minutes)...")
    param_grid = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [15, 20, None],
        "min_samples_split": [2, 4],
    }
    gs = GridSearchCV(
        RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS),
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="accuracy",
        n_jobs=N_JOBS,
        verbose=1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    model = gs.best_estimator_
    print(f"\n   🏆 Best params : {gs.best_params_}")
    print(f"   📈 CV Accuracy  : {gs.best_score_*100:.2f}%")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred   = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
report   = classification_report(y_test, y_pred, zero_division=0)

print(f"\n{'='*55}")
print(f"  ✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"{'='*55}")
print(report)

# Warn if accuracy is still low
if test_acc < 0.70:
    print("⚠️  Accuracy below 70% — consider adding more training images")
    print("   or using a CNN-based model for better leaf discrimination.")

# ── Save Model & Report ───────────────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)

with open(REPORT_PATH, "w") as f:
    mode_str = "fast" if FAST_MODE else "full-gridsearch"
    f.write(f"PlantPulse Training Report\n{'='*55}\n")
    f.write(f"Mode          : {mode_str}\n")
    f.write(f"Feature dims  : {FEATURE_DIM} (histogram-based)\n")
    f.write(f"Test Accuracy : {test_acc*100:.2f}%\n\n")
    f.write(report)

print(f"\n💾 Model saved  → {MODEL_PATH}")
print(f"💾 Scaler saved → {SCALER_PATH}")
print(f"📄 Report saved → {REPORT_PATH}")
print(f"\n⏱  Total time: {time.time()-t0:.1f}s")
print("=" * 55)
print("\n🔄 Restart the Streamlit app to load the new model.")
print("=" * 55)