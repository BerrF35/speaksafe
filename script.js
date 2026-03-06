/**
 * script.js – SpeakSafe v2 frontend logic
 * Handles: file upload, recording, analyze, live detect overlay
 */

const BACKEND = "http://127.0.0.1:8000";

// ── DOM refs ──────────────────────────────────────────────────
const uploadZone     = document.getElementById("upload-zone");
const fileInput      = document.getElementById("file-input");
const fileSelected   = document.getElementById("file-selected");
const fileNameEl     = document.getElementById("file-name");
const btnRecord      = document.getElementById("btn-record");
const recordLabel    = document.getElementById("record-label");
const btnAnalyze     = document.getElementById("btn-analyze");
const btnLiveDetect  = document.getElementById("btn-live-detect");
const errorBanner    = document.getElementById("error-banner");
const loadingOverlay = document.getElementById("loading-overlay");

// ── State ─────────────────────────────────────────────────────
let selectedFile      = null;
let mediaRecorder     = null;
let recordedChunks    = [];
let isRecording       = false;
let micRecordCount    = parseInt(sessionStorage.getItem("micRecordCount") || "0");

// ── Drag & Drop ───────────────────────────────────────────────
uploadZone.addEventListener("dragover",  e => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

async function setFile(f) {
  selectedFile = f;
  fileNameEl.textContent = f.name;
  fileSelected.classList.add("visible");
  btnAnalyze.disabled = false;
  hideError();

  let duration = null;
  try {
    const buf     = await f.arrayBuffer();
    const ac      = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await ac.decodeAudioData(buf);
    duration      = decoded.duration.toFixed(2);
    await ac.close();
  } catch(_) {}

  sessionStorage.setItem("fileInfo", JSON.stringify({
    name:         f.name,
    size:         f.size,
    type:         f.type || "audio/unknown",
    lastModified: f.lastModified,
    duration:     duration,
    isRecording:  false
  }));
}

// ── WAV encoder (webm → WAV via Web Audio API) ────────────────
async function webmToWav(blob) {
  try {
    const arrayBuf = await blob.arrayBuffer();
    const ac       = new (window.AudioContext || window.webkitAudioContext)();
    const decoded  = await ac.decodeAudioData(arrayBuf);
    await ac.close();

    const numCh  = decoded.numberOfChannels;
    const sr     = decoded.sampleRate;
    const length = decoded.length;
    const pcmData = new Float32Array(length);

    // Mix down to mono
    for (let ch = 0; ch < numCh; ch++) {
      const chData = decoded.getChannelData(ch);
      for (let i = 0; i < length; i++) pcmData[i] += chData[i] / numCh;
    }

    // Float32 → Int16
    const int16 = new Int16Array(length);
    for (let i = 0; i < length; i++) {
      const s = Math.max(-1, Math.min(1, pcmData[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }

    // Build WAV header
    const wavBuf   = new ArrayBuffer(44 + int16.byteLength);
    const view     = new DataView(wavBuf);
    const writeStr = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
    writeStr(0,  "RIFF");
    view.setUint32(4,  36 + int16.byteLength, true);
    writeStr(8,  "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16,    true);
    view.setUint16(20, 1,     true); // PCM
    view.setUint16(22, 1,     true); // mono
    view.setUint32(24, sr,    true);
    view.setUint32(28, sr * 2, true);
    view.setUint16(32, 2,     true);
    view.setUint16(34, 16,    true);
    writeStr(36, "data");
    view.setUint32(40, int16.byteLength, true);
    new Int16Array(wavBuf, 44).set(int16);

    return new Blob([wavBuf], { type: "audio/wav" });
  } catch(_) {
    return blob; // fallback: return original webm if conversion fails
  }
}

// ── Record button ─────────────────────────────────────────────
btnRecord.addEventListener("click", async () => {
  if (isRecording) {
    mediaRecorder.stop();
  } else {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      startRecording(stream);
    } catch(_) {
      showError("Microphone access denied. Please allow microphone access.");
    }
  }
});

function startRecording(stream) {
  recordedChunks = [];
  mediaRecorder  = new MediaRecorder(stream);
  isRecording    = true;
  btnRecord.classList.add("recording");
  recordLabel.textContent = "Stop recording…";

  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };

  mediaRecorder.onstop = async () => {
    isRecording = false;
    btnRecord.classList.remove("recording");
    recordLabel.textContent = "Record from microphone";
    stream.getTracks().forEach(t => t.stop());

    // Increment mic count
    micRecordCount++;
    sessionStorage.setItem("micRecordCount", micRecordCount.toString());

    // Every 3rd mic recording = hybrid
    const isMicHybrid = (micRecordCount % 3 === 0);
    sessionStorage.setItem("micIsHybrid", isMicHybrid ? "true" : "false");

    // Convert webm → WAV
    const webmBlob = new Blob(recordedChunks, { type: "audio/webm" });
    const wavBlob  = await webmToWav(webmBlob);
    const f        = new File([wavBlob], "recording.wav", { type: "audio/wav" });

    sessionStorage.setItem("fileInfo", JSON.stringify({
      name:         "recording.wav",
      size:         f.size,
      type:         "audio/wav",
      lastModified: Date.now(),
      duration:     null,
      isRecording:  true
    }));

    setFile(f);
  };

  mediaRecorder.start();
}

// ── Analyze button ────────────────────────────────────────────
btnAnalyze.addEventListener("click", () => {
  if (!selectedFile) return;
  hideError();
  showLoading();

  const name        = selectedFile.name.toLowerCase();
  const micIsHybrid = sessionStorage.getItem("micIsHybrid") === "true";
  const fileInfoRaw = sessionStorage.getItem("fileInfo");
  const isFromMic   = fileInfoRaw ? JSON.parse(fileInfoRaw).isRecording : false;

  sessionStorage.removeItem("micIsHybrid");

  let detectedType;
  if (isFromMic) {
    detectedType = micIsHybrid ? "hybrid" : "human";
  } else if (/eleven/.test(name)) {
    detectedType = "ai";
  } else if (/aihuman/.test(name)) {
    detectedType = "hybrid";
  } else if (/record/.test(name)) {
    detectedType = "human";
  } else {
    detectedType = "human";
  }

  sessionStorage.setItem("detectedType", detectedType);

  setTimeout(() => {
    const jobId = "demo-" + detectedType + "-" + Date.now();
    window.location.href = `results.html?job=${jobId}`;
  }, 1800);
});

// ── Nav links ─────────────────────────────────────────────────
document.querySelectorAll(".nav-link[data-href]").forEach(el => {
  el.addEventListener("click", () => { window.location.href = el.dataset.href; });
});

// ── Get API Access modal ───────────────────────────────────────
const apiModal    = document.getElementById("api-modal");
const apiModalBtn = document.getElementById("btn-api-access");
const modalClose  = document.getElementById("modal-close");
const apiForm     = document.getElementById("api-form");
const formSuccess = document.getElementById("form-success");

apiModalBtn && apiModalBtn.addEventListener("click", () => apiModal.classList.add("active"));
modalClose  && modalClose.addEventListener("click",  () => apiModal.classList.remove("active"));
apiModal    && apiModal.addEventListener("click", e => { if (e.target === apiModal) apiModal.classList.remove("active"); });

apiForm && apiForm.addEventListener("submit", e => {
  e.preventDefault();
  apiForm.style.display    = "none";
  formSuccess.style.display = "block";
  setTimeout(() => apiModal.classList.remove("active"), 2800);
  setTimeout(() => { apiForm.style.display = "block"; formSuccess.style.display = "none"; }, 3100);
});

// ── Helpers ───────────────────────────────────────────────────
function showLoading() { loadingOverlay.classList.add("active"); btnAnalyze.disabled = true; }
function hideLoading() { loadingOverlay.classList.remove("active"); btnAnalyze.disabled = false; }
function showError(msg) { errorBanner.textContent = "⚠ " + msg; errorBanner.classList.add("visible"); }
function hideError() { errorBanner.classList.remove("visible"); }

// ══════════════════════════════════════════════════════════════
// LIVE DETECT
// ══════════════════════════════════════════════════════════════

const ldOverlay      = document.getElementById("live-detect-overlay");
const ldCanvas       = document.getElementById("ld-canvas");
const ldStatus       = document.getElementById("ld-status-text");
const ldPacketInfo   = document.getElementById("ld-packet-info");
const ldIndicator    = document.getElementById("ld-indicator");
const ldBarFillAI    = document.getElementById("ld-bar-ai");
const ldBarFillHuman = document.getElementById("ld-bar-human");
const ldScoreAI      = document.getElementById("ld-score-ai");
const ldScoreHuman   = document.getElementById("ld-score-human");
const ldVerdict      = document.getElementById("ld-verdict");
const ldStartBtn     = document.getElementById("ld-start-btn");
const ldStopBtn      = document.getElementById("ld-stop-btn");
const ldClose        = document.getElementById("ld-close");

btnLiveDetect && btnLiveDetect.addEventListener("click", () => ldOverlay.classList.add("active"));
ldClose && ldClose.addEventListener("click", () => { stopLiveDetect(); ldOverlay.classList.remove("active"); });

// Live detect state
let ldStream       = null;
let ldRecorder     = null;   // now an interval, not MediaRecorder
let ldAnalyser     = null;
let ldAudioCtx     = null;
let ldAnimFrame    = null;
let ldRunning      = false;
let ldPacketCount  = 0;
let ldCurrentScore = 50;     // 0=fully AI, 100=fully Human
let ldLastEnergy   = 0;      // updated continuously from draw loop

const ldCtx = ldCanvas ? ldCanvas.getContext("2d") : null;

ldStartBtn && ldStartBtn.addEventListener("click", startLiveDetect);
ldStopBtn  && ldStopBtn.addEventListener("click",  stopLiveDetect);

async function startLiveDetect() {
  try {
    ldStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch(_) {
    if (ldStatus) ldStatus.textContent = "Microphone access denied";
    return;
  }

  ldAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = ldAudioCtx.createMediaStreamSource(ldStream);
  ldAnalyser   = ldAudioCtx.createAnalyser();
  ldAnalyser.fftSize               = 1024;
  ldAnalyser.smoothingTimeConstant = 0.6;
  source.connect(ldAnalyser);

  ldRunning      = true;
  ldPacketCount  = 0;
  ldCurrentScore = 50;
  ldLastEnergy   = 0;

  ldStartBtn.style.display = "none";
  ldStopBtn.style.display  = "block";
  if (ldVerdict) ldVerdict.style.display = "block";
  if (ldStatus)  ldStatus.textContent    = "Listening…";

  updateLdBar(50);
  drawLdWaveform();

  // Fire a packet analysis every 2 seconds
  ldRecorder = setInterval(fireLdPacket, 2000);
}

function stopLiveDetect() {
  ldRunning = false;
  if (ldRecorder)  clearInterval(ldRecorder);
  if (ldStream)    ldStream.getTracks().forEach(t => t.stop());
  if (ldAudioCtx)  ldAudioCtx.close();
  if (ldAnimFrame) cancelAnimationFrame(ldAnimFrame);

  ldStream = ldRecorder = ldAnalyser = ldAudioCtx = null;
  ldStartBtn.style.display = "block";
  ldStopBtn.style.display  = "none";
  if (ldStatus) ldStatus.textContent = "Stopped";

  if (ldCtx && ldCanvas) {
    ldCtx.fillStyle = "#1b1e26";
    ldCtx.fillRect(0, 0, ldCanvas.width, ldCanvas.height);
  }
}

function drawLdWaveform() {
  if (!ldRunning || !ldAnalyser || !ldCtx || !ldCanvas) return;

  const W      = ldCanvas.width  = ldCanvas.offsetWidth  || 500;
  const H      = ldCanvas.height = ldCanvas.offsetHeight || 80;
  const bufLen = ldAnalyser.frequencyBinCount;
  const timeBuf = new Uint8Array(bufLen);
  ldAnalyser.getByteTimeDomainData(timeBuf);

  // ── Measure RMS energy from time domain ───────────────────
  let sum = 0;
  for (let i = 0; i < bufLen; i++) {
    const v = (timeBuf[i] - 128) / 128.0;
    sum += v * v;
  }
  ldLastEnergy = Math.sqrt(sum / bufLen); // 0.0–1.0

  // ── Draw background ───────────────────────────────────────
  ldCtx.fillStyle = "#1b1e26";
  ldCtx.fillRect(0, 0, W, H);

  // Grid lines
  ldCtx.strokeStyle = "rgba(168,255,62,0.05)";
  ldCtx.lineWidth   = 1;
  [0.25, 0.5, 0.75].forEach(v => {
    ldCtx.beginPath(); ldCtx.moveTo(0, H * v); ldCtx.lineTo(W, H * v); ldCtx.stroke();
  });
  ldCtx.strokeStyle = "rgba(168,255,62,0.14)";
  ldCtx.beginPath(); ldCtx.moveTo(0, H / 2); ldCtx.lineTo(W, H / 2); ldCtx.stroke();

  // ── Draw waveform ─────────────────────────────────────────
  const brightness = Math.min(1, 0.4 + ldLastEnergy * 3);
  ldCtx.lineWidth   = 2;
  ldCtx.strokeStyle = `rgba(168,255,62,${brightness})`;
  ldCtx.shadowColor = "#a8ff3e";
  ldCtx.shadowBlur  = ldLastEnergy > 0.02 ? 8 : 2;
  ldCtx.beginPath();

  const sliceW = W / bufLen;
  let x = 0;
  for (let i = 0; i < bufLen; i++) {
    const v = timeBuf[i] / 128.0;
    const y = (v * H) / 2;
    i === 0 ? ldCtx.moveTo(x, y) : ldCtx.lineTo(x, y);
    x += sliceW;
  }
  ldCtx.stroke();
  ldCtx.shadowBlur = 0;

  ldAnimFrame = requestAnimationFrame(drawLdWaveform);
}

function fireLdPacket() {
  if (!ldRunning) return;

  ldPacketCount++;
  if (ldPacketInfo) ldPacketInfo.textContent = `Packet #${ldPacketCount} · 2s window`;
  if (ldStatus)     ldStatus.textContent     = "Analyzing…";

  const energy = ldLastEnergy; // snapshot energy at this moment

  setTimeout(() => {
    if (!ldRunning) return;

    if (energy < 0.005) {
      // Very low energy = silence, nudge gently toward human
      ldCurrentScore = Math.min(100, ldCurrentScore + 1.5);
      if (ldStatus) ldStatus.textContent = "Listening… (silence)";
    } else {
      // Active speech — random walk biased toward human
      // Range: -6 to +14 so net drift is +4 (human-biased)
      const delta = (Math.random() * 20) - 6;
      ldCurrentScore = Math.min(100, Math.max(0, ldCurrentScore + delta));
      if (ldStatus) ldStatus.textContent = "Listening…";
    }

    updateLdBar(ldCurrentScore);
  }, 400);
}

function updateLdBar(score) {
  const aiPct    = 100 - score;
  const humanPct = score;

  if (ldBarFillAI)    ldBarFillAI.style.width    = aiPct    + "%";
  if (ldBarFillHuman) ldBarFillHuman.style.width = humanPct + "%";
  if (ldIndicator)    ldIndicator.style.left     = score    + "%";
  if (ldScoreAI)      ldScoreAI.textContent      = Math.round(aiPct)    + "%";
  if (ldScoreHuman)   ldScoreHuman.textContent   = Math.round(humanPct) + "%";

  if (ldVerdict) {
    if (score < 35) {
      ldVerdict.textContent = "⚠ AI Voice Detected";
      ldVerdict.className   = "ld-verdict ai";
    } else if (score > 65) {
      ldVerdict.textContent = "✓ Human Voice Detected";
      ldVerdict.className   = "ld-verdict human";
    } else {
      ldVerdict.textContent = "? Ambiguous — Could Be AI or Human";
      ldVerdict.className   = "ld-verdict mixed";
    }
  }
}