FROM python:3.11-slim

WORKDIR /app

# System deps for librosa / soundfile
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY models/  ./models/

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
