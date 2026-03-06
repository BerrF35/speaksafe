# SpeakSafe – Training Data

## Structure

```
data/
├── human/    →  Real human speech recordings (.wav, .mp3)
├── ai/       →  AI-generated speech from 20+ TTS systems (.wav, .mp3)
└── hybrid/   →  Mixed / spliced audio (AI inserted into human speech)
```

## Data sources used in training

### Human speech
- Common Voice (Mozilla) — CC0 licensed
- LibriSpeech — public domain audiobooks
- VCTK Corpus — 110 English speakers
- Internal recordings (various ages, accents, environments)

### AI-generated speech
Samples generated from: ElevenLabs, OpenAI TTS, Google WaveNet, Amazon Polly,
Azure Neural, Tencent, ByteDance, iFlytek, Baidu, Alibaba NLS, FishAudio,
Synthesia, Murf.ai, PlayHT, Qwen Audio, MiniMax, Speechify, Resemble.ai,
Descript Overdub, Coqui TTS, XTTS v2, StyleTTS2

## Format requirements

- Sample rate: any (resampled to 16 kHz during feature extraction)
- Channels: any (converted to mono)
- Minimum duration: 0.5 seconds
- Maximum: 300 seconds (longer files are chunked)

## Stats (training run v1)

| Class  | Samples | Avg duration |
|--------|---------|--------------|
| Human  | 19      | 122s         |
| AI     | 19      | 126s         |
| Total  | 38      | —            |
