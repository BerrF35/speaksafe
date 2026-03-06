"""
scripts/train.py
=================
Train the SpeakSafe GradientBoosting classifier.

Usage:
  python scripts/train.py --data-dir data/ --out models/speaksafe_v1.pkl

Expects:
  data/human/   → .wav or .mp3 files of human speech
  data/ai/      → .wav or .mp3 files of AI-generated speech
  data/hybrid/  → .wav or .mp3 files of hybrid/mixed audio (optional)

Output:
  models/speaksafe_v1.pkl   → trained classifier
  models/label_map.json     → class index mapping
  models/training_report.txt → accuracy + classification report
"""

import argparse
import os
import json
import time
import numpy as np
from pathlib import Path

try:
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from audio_features import extract_features


SUPPORTED_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}


def load_dataset(data_dir: str):
    X, y = [], []
    label_map = {"human": 0, "ai": 1}

    for label, cls in label_map.items():
        folder = Path(data_dir) / label
        if not folder.exists():
            print(f"  [WARN] {folder} does not exist — skipping")
            continue

        files = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
        print(f"  Loading {len(files)} files from {folder}/")

        for i, fp in enumerate(files):
            try:
                feats = extract_features(str(fp))
                vec   = [feats[k] for k in sorted(feats) if k != "duration_seconds"]
                X.append(vec)
                y.append(cls)
                if (i + 1) % 100 == 0:
                    print(f"    {i+1}/{len(files)} processed")
            except Exception as e:
                print(f"    [SKIP] {fp.name}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), label_map


def train(data_dir: str, out_path: str):
    if not SKLEARN_AVAILABLE:
        print("scikit-learn not installed. Run: pip install scikit-learn")
        return

    print(f"\n{'='*50}")
    print(" SpeakSafe – Model Training")
    print(f"{'='*50}\n")

    t0 = time.time()

    print("→ Loading dataset…")
    X, y, label_map = load_dataset(data_dir)

    if len(X) == 0:
        print("\n[ERROR] No audio files found. Add files to data/human/ and data/ai/")
        return

    print(f"  Total samples: {len(X)} ({np.sum(y==0)} human, {np.sum(y==1)} AI)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n→ Training GradientBoostingClassifier…")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])

    calibrated = CalibratedClassifierCV(pipe, cv=StratifiedKFold(5), method="sigmoid")
    calibrated.fit(X_train, y_train)

    print("\n→ Evaluating…")
    y_pred  = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)[:, 1]
    report  = classification_report(y_test, y_pred, target_names=["human", "ai"])
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\n{report}")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  Training time: {time.time()-t0:.1f}s")

    # Save model
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(calibrated, out_path)
    print(f"\n→ Model saved → {out_path}")

    # Save label map
    lm_path = os.path.join(os.path.dirname(out_path), "label_map.json")
    with open(lm_path, "w") as f:
        json.dump(label_map, f, indent=2)

    # Save report
    rpt_path = os.path.join(os.path.dirname(out_path), "training_report.txt")
    with open(rpt_path, "w") as f:
        f.write(f"SpeakSafe Model Training Report\n{'='*40}\n\n")
        f.write(f"Samples: {len(X)} ({np.sum(y==0)} human, {np.sum(y==1)} AI)\n")
        f.write(f"ROC-AUC: {auc:.4f}\n\n")
        f.write(report)
    print(f"→ Report saved → {rpt_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SpeakSafe classifier")
    parser.add_argument("--data-dir", default="data/",    help="Path to data folder")
    parser.add_argument("--out",      default="models/speaksafe_v1.pkl", help="Output model path")
    args = parser.parse_args()
    train(args.data_dir, args.out)
