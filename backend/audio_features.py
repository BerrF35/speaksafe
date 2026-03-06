"""
audio_features.py
==================
Extracts acoustic features from an audio file for AI voice detection.

Features extracted:
  - MFCC (13 coefficients) — mean + variance
  - Spectral centroid       — mean + variance
  - Zero-crossing rate      — mean + variance
  - RMS energy              — mean + variance
  - Spectral bandwidth      — mean + variance
  - Spectral rolloff        — mean + variance
  - Harmonic-to-noise ratio — mean
  - Chroma STFT             — mean + variance (12 bins)
  - Mel spectrogram         — statistics for banding detection

All audio resampled to 16 kHz mono before processing.
Window: 25ms Hann. Hop: 10ms.
"""

import numpy as np

try:
    import librosa
    import librosa.feature
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


SR        = 16000    # target sample rate
N_MFCC    = 13
HOP_MS    = 10       # hop length in ms
WIN_MS    = 25       # window length in ms


def extract_features(audio_path: str) -> dict:
    """
    Load audio file and extract all acoustic features.
    Returns a flat dict of feature_name → float value.
    """
    if not LIBROSA_AVAILABLE:
        raise RuntimeError(
            "librosa is not installed. Run: pip install librosa soundfile"
        )

    hop_length = int(SR * HOP_MS / 1000)   # 160 samples
    win_length = int(SR * WIN_MS / 1000)   # 400 samples
    n_fft      = 512

    # ── Load & resample ────────────────────────────────────────
    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    if len(y) < SR * 0.5:
        raise ValueError("Audio too short. Please provide at least 0.5 seconds.")

    # ── Trim silence from edges ────────────────────────────────
    y, _ = librosa.effects.trim(y, top_db=20)

    # ── MFCC ──────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=N_MFCC,
        hop_length=hop_length, win_length=win_length, n_fft=n_fft
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var  = np.var(mfcc, axis=1)

    # ── Spectral centroid ─────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=SR, hop_length=hop_length, win_length=win_length, n_fft=n_fft
    )[0]
    centroid_norm = centroid / SR   # normalize to 0–1

    # ── Zero-crossing rate ────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(
        y, hop_length=hop_length, frame_length=win_length
    )[0]

    # ── RMS energy ───────────────────────────────────────────
    rms = librosa.feature.rms(
        y=y, hop_length=hop_length, frame_length=win_length
    )[0]

    # ── Spectral bandwidth ────────────────────────────────────
    bw = librosa.feature.spectral_bandwidth(
        y=y, sr=SR, hop_length=hop_length, win_length=win_length, n_fft=n_fft
    )[0]

    # ── Spectral rolloff ──────────────────────────────────────
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=SR, hop_length=hop_length, win_length=win_length, n_fft=n_fft
    )[0]

    # ── Chroma ────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(
        y=y, sr=SR, hop_length=hop_length, win_length=win_length, n_fft=n_fft
    )

    # ── Mel spectrogram banding detector ──────────────────────
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=64,
        hop_length=hop_length, win_length=win_length, n_fft=n_fft
    )
    mel_db  = librosa.power_to_db(mel, ref=np.max)
    mel_var = np.var(mel_db, axis=1)   # variance across time per mel bin

    # Low mel-band variance = horizontal banding = AI marker
    banding_score = float(np.mean(mel_var < 6.0))

    # ── Periodicity of ZCR (AI voices have rhythmic ZCR) ─────
    zcr_ac = np.correlate(zcr - zcr.mean(), zcr - zcr.mean(), mode="full")
    zcr_ac = zcr_ac[len(zcr_ac)//2:]
    zcr_ac = zcr_ac / zcr_ac[0] if zcr_ac[0] != 0 else zcr_ac
    zcr_periodicity = float(np.max(zcr_ac[5:50])) if len(zcr_ac) > 50 else 0.0

    # ── Harmonic/percussive separation ───────────────────────
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_ratio = float(np.mean(y_harm**2) / (np.mean(y**2) + 1e-9))

    features = {
        # MFCC stats (13 × 2 = 26 values)
        **{f"mfcc_mean_{i}":  float(mfcc_mean[i])  for i in range(N_MFCC)},
        **{f"mfcc_var_{i}":   float(mfcc_var[i])   for i in range(N_MFCC)},

        # Aggregate MFCC variance regularity (low = AI)
        "mfcc_variance_regularity":    float(np.mean(mfcc_var) / (np.std(mfcc_var) + 1e-9)),

        # Centroid
        "centroid_mean":               float(np.mean(centroid_norm)),
        "centroid_var":                float(np.var(centroid_norm)),
        "spectral_centroid_stability": float(1 - np.std(centroid_norm) / (np.mean(centroid_norm) + 1e-9)),

        # ZCR
        "zcr_mean":                    float(np.mean(zcr)),
        "zcr_var":                     float(np.var(zcr)),
        "zcr_regularity":              zcr_periodicity,

        # RMS
        "rms_mean":                    float(np.mean(rms)),
        "rms_var":                     float(np.var(rms)),
        "rms_uniformity":              float(1 - np.std(rms) / (np.mean(rms) + 1e-9)),

        # Bandwidth + rolloff
        "bw_mean":                     float(np.mean(bw)),
        "bw_var":                      float(np.var(bw)),
        "rolloff_mean":                float(np.mean(rolloff)),
        "rolloff_var":                 float(np.var(rolloff)),

        # Chroma
        "chroma_mean":                 float(np.mean(chroma)),
        "chroma_var":                  float(np.var(chroma)),

        # Harmonic
        "harmonic_dominance":          harm_ratio,

        # Banding
        "mel_banding_score":           banding_score,

        # Duration
        "duration_seconds":            float(len(y) / SR),
    }

    return features
