# SpeakSafe 🎙

**AI-generated speech detector** — Upload or record audio and instantly find out if a voice is human, AI-generated, or a hybrid.

![SpeakSafe banner](docs/banner.png)

---

## What it does

SpeakSafe analyzes audio files using acoustic feature extraction and a trained ML classifier to determine:

- ✅ **Human** — natural, organic speech
- ⚠️ **AI Generated** — synthesized by a TTS system
- ⚡ **AI + Human Hybrid** — spliced or dubbed content

Trained on a dataset containing samples from multiple modern neural TTS systems.
---

## Features

- 🔬 **12+ acoustic features** — MFCCs, spectral centroid, ZCR, HNR, RMS, mel banding
- 🌊 **Live spectrogram** — rendered canvas heatmap showing frequency patterns over time
- 📡 **Live detect mode** — real-time mic stream analyzed in 2-second windows
- 📊 **Signal breakdown** — per-feature confidence scores
- 🗂️ **File details** — filename, size, format, duration, source
- 🔌 **REST API** — integrate detection into your own platform

---

## Project Structure

```
speaksafe/
├── frontend/               # HTML/CSS/JS — no framework, runs in browser
│   ├── index.html          # Main page: upload, record, live detect
│   ├── results.html        # Analysis results with spectrogram
│   ├── how-it-works.html   # Pipeline documentation
│   ├── api.html            # API reference docs
│   ├── styles.css          # Full stylesheet
│   └── script.js           # All frontend logic
│
├── backend/                # Python FastAPI backend
│   ├── server.py           # API endpoints: POST /analyze, GET /results/{id}
│   ├── audio_features.py   # Acoustic feature extraction (librosa)
│   ├── model.py            # ML classifier inference
│   └── requirements.txt    # Python dependencies
│
├── models/                 # Trained model files (excluded from git)
│   ├── speaksafe_v1.pkl    # GradientBoosting + calibration pipeline
│   ├── label_map.json      # Class index map
│   └── training_report.txt # Accuracy & AUC from training run
│
├── data/                   # Training audio (excluded from git)
│   ├── human/              # Real human speech recordings
│   ├── ai/                 # AI-generated speech samples
│   └── hybrid/             # Mixed/spliced audio
│
├── notebooks/
│   └── 01_train_model.ipynb  # Full training walkthrough
│
├── scripts/
│   └── train.py            # CLI training script
│
├── tests/
│   └── test_api.py         # API integration tests
│
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Quick Start

### Frontend demo mode (limited functionality without backend)

Just open `frontend/index.html` in any browser. No server needed — the UI runs in full demo mode showing realistic results.

### With backend

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Start the API server
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# 3. Open frontend/index.html in your browser
```

### With Docker

```bash
docker-compose up --build
# Backend available at http://localhost:8000
```

---

## API

### POST /analyze
Submit an audio file for analysis.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@sample.mp3"
# → { "job_id": "uuid", "status": "processing" }
```

### GET /results/{job_id}
Poll for results (202 while processing, 200 when done).

```bash
curl http://localhost:8000/results/{job_id}
# → { "ai_probability": 0.87, "classification": "AI Generated Voice", ... }
```

Full API reference: see `frontend/api.html` or run the server and visit `/docs`.

---

## How it works

1. **Ingest** — audio decoded, resampled to 16 kHz mono
2. **Extract** — 12 acoustic features computed over 25ms windows
3. **Classify** — GradientBoosting ensemble outputs P(AI)
4. **Explain** — per-signal scores + natural language breakdown returned

Key insight: AI voices are *unnaturally consistent*. Natural human speech has jitter, shimmer, irregular ZCR, and dynamic RMS variation. Neural TTS systems smooth all of this out — and that smoothness is measurable.

---

## Training your own model

```bash
# Add audio to data/human/ and data/ai/
# Then run:
python scripts/train.py --data-dir data/ --out models/speaksafe_v1.pkl
```

Or use the full walkthrough in `notebooks/01_train_model.ipynb`.

---

## Accuracy & Limitations

| Condition | Accuracy |
|-----------|----------|
| Clean audio, known AI system | ~91% |
| Compressed audio (< 64 kbps) | ~78% |
| Audio < 1 second | ~65% |
| Unknown / custom fine-tuned model | ~70% |

> **Note**: SpeakSafe is a forensic support tool. Do not use it as sole evidence in legal or consequential decisions.

---

## License

MIT — see LICENSE for details.

---

*Built for the age of synthetic voice.*
