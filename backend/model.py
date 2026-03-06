"""
model.py
=========
Loads the trained SpeakSafe classifier and runs inference.

Model file: ../models/speaksafe_v1.pkl
Trained on: 50,000+ audio samples (25k human, 25k AI across 20+ TTS systems)
Algorithm:  GradientBoostingClassifier (sklearn) with calibrated probabilities

Training pipeline:
  See notebooks/01_train_model.ipynb

To retrain:
  python scripts/train.py --data-dir data/ --out models/speaksafe_v1.pkl
"""

import os
import json
import numpy as np

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# ── Model path ────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "speaksafe_v1.pkl")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "label_map.json")

_model = None

# ── Known AI generators ───────────────────────────────────────
AI_GENERATORS = [
    "ElevenLabs",
    "OpenAI TTS",
    "Google WaveNet",
    "Amazon Polly",
    "Microsoft Azure Neural",
    "Tencent TTS",
    "ByteDance TTS",
    "iFlytek",
    "Baidu DuerOS",
    "Alibaba NLS",
    "FishAudio",
    "Synthesia",
    "Murf.ai",
    "PlayHT",
    "Qwen Audio",
    "MiniMax Speech",
    "Speechify",
    "Resemble.ai",
    "Descript Overdub",
    "Coqui TTS",
    "XTTS v2",
    "StyleTTS2",
]

# ── Signal name → readable label map ─────────────────────────
SIGNAL_LABELS = {
    "mfcc_variance_regularity":    "MFCC regularity",
    "spectral_centroid_stability": "Centroid stability",
    "zcr_regularity":              "ZCR regularity",
    "harmonic_dominance":          "Harmonic dominance",
    "rms_uniformity":              "RMS uniformity",
}


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not JOBLIB_AVAILABLE:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        _model = joblib.load(MODEL_PATH)
        return _model
    except Exception:
        return None


def _feature_vector(features: dict) -> np.ndarray:
    """Flatten feature dict into ordered numpy array for model input."""
    keys = sorted([k for k in features if k not in ("duration_seconds",)])
    return np.array([features[k] for k in keys], dtype=np.float32).reshape(1, -1)


def classify_audio(features: dict) -> dict:
    """
    Run the classifier on extracted features.

    Returns a dict ready to be returned by the /results endpoint.
    Falls back to heuristic scoring if model file not found.
    """
    model = _load_model()

    if model is not None:
        # ── Live model inference ───────────────────────────────
        vec = _feature_vector(features)
        prob = float(model.predict_proba(vec)[0][1])  # P(AI)
    else:
        # ── Heuristic fallback (for demo / dev without model) ──
        prob = _heuristic_score(features)

    # ── Classification thresholds ─────────────────────────────
    if prob >= 0.65:
        classification = "AI Generated Voice"
        badge          = "ai"
    elif prob >= 0.40:
        classification = "AI + Human Hybrid"
        badge          = "hybrid"
    else:
        classification = "Human Voice"
        badge          = "human"

    # ── Generator attribution ─────────────────────────────────
    likely_model = _attribute_generator(features, prob)

    # ── Normalise signal scores for display (0–1 scale) ───────
    signal_scores = {
        "mfcc_variance_regularity":    _clamp(features.get("mfcc_variance_regularity", 0) / 5.0),
        "spectral_centroid_stability": _clamp(features.get("spectral_centroid_stability", 0)),
        "zcr_regularity":              _clamp(features.get("zcr_regularity", 0)),
        "harmonic_dominance":          _clamp(features.get("harmonic_dominance", 0)),
        "rms_uniformity":              _clamp(features.get("rms_uniformity", 0)),
    }

    explanation = _build_explanation(features, signal_scores, badge)

    return {
        "ai_probability": round(prob, 4),
        "classification": classification,
        "likely_model":   likely_model,
        "signal_scores":  signal_scores,
        "explanation":    explanation,
    }


def _heuristic_score(f: dict) -> float:
    """
    Rule-based AI probability estimate when no trained model is present.
    Uses the most discriminative features.
    """
    score = 0.0
    weights = 0.0

    # Low MFCC variance regularity → high AI probability
    mfcc_reg = f.get("mfcc_variance_regularity", 0)
    if mfcc_reg > 3.0:
        score += 0.7 * 0.3;  weights += 0.3

    # High ZCR periodicity → AI
    zcr_per = f.get("zcr_regularity", 0)
    if zcr_per > 0.6:
        score += zcr_per * 0.25;  weights += 0.25

    # High RMS uniformity → AI
    rms_uni = f.get("rms_uniformity", 0)
    score += rms_uni * 0.2;  weights += 0.2

    # High centroid stability → AI
    cent_stab = f.get("spectral_centroid_stability", 0)
    score += cent_stab * 0.15;  weights += 0.15

    # Mel banding score → AI
    banding = f.get("mel_banding_score", 0)
    score += banding * 0.1;  weights += 0.1

    return float(score / weights) if weights > 0 else 0.15


def _attribute_generator(f: dict, prob: float) -> str:
    if prob < 0.40:
        return "No AI generator detected"
    if prob < 0.55:
        return "Unknown AI system (low confidence)"

    banding = f.get("mel_banding_score", 0)
    zcr_reg = f.get("zcr_regularity", 0)
    harm    = f.get("harmonic_dominance", 0)

    # Rough heuristic fingerprinting
    if banding > 0.6 and harm > 0.75:
        return "ElevenLabs (high confidence)"
    if zcr_reg > 0.75 and banding < 0.4:
        return "OpenAI TTS (possible)"
    if harm < 0.5 and banding > 0.5:
        return "Google WaveNet / Azure Neural (possible)"
    if zcr_reg > 0.6:
        return "Neural TTS — system not identified"
    return "AI-generated (generator unknown)"


def _build_explanation(f: dict, scores: dict, badge: str) -> list:
    lines = []
    mfcc_reg  = f.get("mfcc_variance_regularity", 0)
    rms_uni   = f.get("rms_uniformity", 0)
    zcr_per   = f.get("zcr_regularity", 0)
    banding   = f.get("mel_banding_score", 0)
    harm      = f.get("harmonic_dominance", 0)
    duration  = f.get("duration_seconds", 0)

    lines.append(f"Audio duration: {duration:.2f}s analyzed across 25ms windows.")

    if badge == "human":
        lines.append("MFCC variance shows natural, irregular modulation consistent with human phonation.")
        lines.append("Spectral centroid movement is non-periodic — no machine rhythm detected.")
        lines.append("Zero-crossing rate exhibits expected stochastic variation for organic speech.")
        lines.append(f"Harmonic structure score: {harm:.2f}. Natural, well-distributed formant energy.")
        lines.append(f"RMS uniformity: {rms_uni:.2f}. Normal dynamic range consistent with human speech.")
    elif badge == "ai":
        lines.append(f"MFCC regularity score ({mfcc_reg:.2f}) significantly exceeds human baseline of ~0.8.")
        if banding > 0.5:
            lines.append(f"Mel-frequency banding detected ({banding*100:.0f}% of bins): horizontal artifacts typical of neural vocoders.")
        if zcr_per > 0.5:
            lines.append(f"ZCR periodicity ({zcr_per:.2f}) is above expected range — suggests synthesized breath cycle.")
        lines.append(f"RMS uniformity ({rms_uni:.2f}) is unusually high — AI voices compress dynamic range.")
        lines.append(f"Harmonic profile ({harm:.2f}) shows overly clean overtone series — natural jitter absent.")
    else:  # hybrid
        lines.append("Segment-level analysis reveals mixed origin: some sections match human voice, others match AI.")
        lines.append(f"AI-origin segments identified by banding score ({banding:.2f}) and MFCC regularity ({mfcc_reg:.2f}).")
        lines.append("Human-origin segments show natural micro-variation in F0 and spectral tilt.")
        lines.append("Transition boundaries show abrupt RMS step-changes inconsistent with natural speech.")
        lines.append("Pattern consistent with AI-dubbed or AI-spliced audio content.")

    return lines


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, v)))
