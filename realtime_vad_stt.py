import sounddevice as sd
import numpy as np
import webrtcvad
import queue
import threading
import sys
from faster_whisper import WhisperModel
import collections
import time

# ==============================
# CONFIGURATION
# ==============================
MODEL_SIZE = "small"       # base / small / medium
DEVICE = "cuda"            # or "cpu"
LANGUAGE = "en"
SAMPLE_RATE = 16000
FRAME_DURATION = 30        # ms (10, 20, or 30 allowed for webrtcvad)
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
VAD_MODE = 2               # 0=least, 3=most aggressive
MAX_SILENCE_FRAMES = int(0.8 * 1000 / FRAME_DURATION)  # ~0.8s silence
MIN_SPEECH_FRAMES = int(0.3 * 1000 / FRAME_DURATION)   # at least 0.3s talking

# ==============================
# LOAD MODEL
# ==============================
print("🔄 Loading Faster-Whisper model...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8_float16")
print("✅ Model loaded successfully!")

# ==============================
# AUDIO CAPTURE
# ==============================
audio_queue = queue.Queue()
vad = webrtcvad.Vad(VAD_MODE)
running = True

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    # Convert float32 → 16-bit PCM bytes
    pcm16 = (indata * 32768).astype(np.int16).tobytes()
    audio_queue.put(pcm16)

# ==============================
# WORKER LOOP
# ==============================
def stt_worker():
    print("🎧 Ready! Start speaking… (Ctrl+C to stop)")
    buffer = collections.deque(maxlen=50)
    speech_data = bytearray()
    speaking = False
    silence_counter = 0

    while running:
        try:
            frame = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        is_speech = vad.is_speech(frame, SAMPLE_RATE)
        buffer.append((frame, is_speech))

        # If detected speech
        if is_speech:
            if not speaking:
                speaking = True
                speech_data.clear()
                silence_counter = 0
                print("🟢 Speaking detected…")
            speech_data.extend(frame)
        else:
            if speaking:
                silence_counter += 1
                if silence_counter > MAX_SILENCE_FRAMES:
                    speaking = False
                    silence_counter = 0
                    print("🛑 Speech ended. Processing…")

                    # Convert PCM to float32 for Whisper
                    audio_np = np.frombuffer(speech_data, np.int16).astype(np.float32) / 32768.0
                    segments, _ = model.transcribe(audio_np, language=LANGUAGE, beam_size=1)

                    for seg in segments:
                        text = seg.text.strip()
                        if text:
                            print(f"🗣️ {text}")

                    speech_data.clear()

# ==============================
# MAIN
# ==============================
with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=FRAME_SIZE,
    callback=audio_callback,
):
    worker = threading.Thread(target=stt_worker)
    worker.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Exiting gracefully...")
        running = False
        worker.join()
