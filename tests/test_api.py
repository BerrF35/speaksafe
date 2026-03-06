"""
tests/test_api.py
==================
Integration tests for the SpeakSafe FastAPI backend.

Run with:
  cd backend && pytest ../tests/ -v
"""

import io
import pytest
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from server import app

client = TestClient(app)


def make_silent_wav(duration_s=1.0, sr=16000) -> bytes:
    """Generate a minimal silent WAV file for testing."""
    import struct, math
    n_samples  = int(sr * duration_s)
    data_chunk = b'\x00\x00' * n_samples
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + len(data_chunk), b'WAVE',
        b'fmt ', 16, 1, 1, sr, sr*2, 2, 16,
        b'data', len(data_chunk)
    )
    return header + data_chunk


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_analyze_silent_wav():
    wav = make_silent_wav(duration_s=2.0)
    resp = client.post(
        "/analyze",
        files={"file": ("test.wav", io.BytesIO(wav), "audio/wav")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "processing"


def test_analyze_no_file():
    resp = client.post("/analyze")
    assert resp.status_code == 422


def test_results_not_found():
    resp = client.get("/results/nonexistent-job-id")
    assert resp.status_code == 404


def test_results_after_analyze():
    import time
    wav  = make_silent_wav(duration_s=2.0)
    resp = client.post(
        "/analyze",
        files={"file": ("test.wav", io.BytesIO(wav), "audio/wav")}
    )
    job_id = resp.json()["job_id"]
    time.sleep(4)  # wait for background task
    result = client.get(f"/results/{job_id}")
    assert result.status_code in (200, 202)
    if result.status_code == 200:
        d = result.json()
        assert "ai_probability" in d
        assert 0.0 <= d["ai_probability"] <= 1.0
        assert d["classification"] in ("Human Voice", "AI Generated Voice", "AI + Human Hybrid")
