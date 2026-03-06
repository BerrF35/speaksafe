"""
SpeakSafe – FastAPI Backend
============================
Endpoints:
  POST /analyze          → accepts audio file, returns job_id
  GET  /results/{job_id} → returns full analysis result
  GET  /health           → health check

Run with:
  cd backend
  uvicorn server:app --reload --host 0.0.0.0 --port 8000
"""

import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import tempfile, os, time

from audio_features import extract_features
from model import classify_audio

app = FastAPI(title="SpeakSafe API", version="1.0.0")

# ── CORS (allow frontend dev server or file:// origin) ────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store (use Redis in production) ─────────────
job_store: dict = {}


# ── Models ─────────────────────────────────────────────────────
class AnalyzeResponse(BaseModel):
    job_id: str
    status: str
    estimated_seconds: float

class ResultResponse(BaseModel):
    job_id: str
    status: str
    ai_probability: Optional[float] = None
    classification: Optional[str] = None
    likely_model: Optional[str] = None
    signal_scores: Optional[dict] = None
    explanation: Optional[list] = None
    processed_at: Optional[str] = None


# ── Background task: run analysis ─────────────────────────────
async def run_analysis(job_id: str, tmp_path: str):
    try:
        features = extract_features(tmp_path)
        result   = classify_audio(features)
        result["job_id"] = job_id
        result["status"] = "complete"
        result["processed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        job_store[job_id] = result
    except Exception as e:
        job_store[job_id] = {"job_id": job_id, "status": "error", "detail": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Routes ─────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    allowed = {"audio/mpeg", "audio/wav", "audio/ogg", "audio/flac", "audio/mp4",
               "audio/webm", "audio/x-wav", "audio/x-flac", "application/octet-stream"}

    if file.content_type not in allowed and not file.filename.endswith(
        (".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm")
    ):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    # Write to temp file
    suffix = os.path.splitext(file.filename)[-1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Max 25 MB.")
        tmp.write(content)
        tmp_path = tmp.name

    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "processing"}
    background_tasks.add_task(run_analysis, job_id, tmp_path)

    return {"job_id": job_id, "status": "processing", "estimated_seconds": 2.5}


@app.get("/results/{job_id}")
def get_results(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = job_store[job_id]

    if job["status"] == "processing":
        return JSONResponse(status_code=202, content={"job_id": job_id, "status": "processing"})

    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job.get("detail", "Analysis failed."))

    return job
