import os, time, collections, threading
import numpy as np
import sounddevice as sd
import webrtcvad

SAMPLE_RATE    = 16000
FRAME_DURATION = int(os.getenv("FRAME_DURATION_MS", "20"))  # 10/20/30
FRAME_SIZE     = int(SAMPLE_RATE * FRAME_DURATION / 1000)
VAD_MODE       = int(os.getenv("VAD_MODE", "2"))

audio_queue = collections.deque()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    pcm16 = (indata * 32768).astype(np.int16).tobytes()
    # use deque to minimize locks; STT will pop via .popleft()
    audio_queue.append(pcm16)

def run_audio_loop(stop_event):
    while not stop_event.is_set():
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                blocksize=FRAME_SIZE, callback=audio_callback):
                while not stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            print("⚠️ Input stream error, retrying in 1s:", e)
            time.sleep(1.0)

# ---- Energy gate (moved as-is)
class EnergyGate:
    def __init__(self, frame_ms=FRAME_DURATION, calibrate_ms=int(os.getenv("CALIBRATE_MS","1000")),
                 floor_mult=float(os.getenv("ENERGY_FLOOR_MULT","2.2")), 
                 min_rms=float(os.getenv("ENERGY_MIN_RMS","0.0055"))):
        self.frame_ms = frame_ms
        self.target = max(1, calibrate_ms // frame_ms)
        self.buf = collections.deque(maxlen=self.target)
        self.baseline = None
        self.floor_mult = floor_mult
        self.min_rms = min_rms
        self.frames_seen = 0

    @staticmethod
    def rms(frame_bytes):
        x = np.frombuffer(frame_bytes, np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def loud_enough(self, frame_bytes):
        rms = self.rms(frame_bytes)
        if self.baseline is None:
            self.buf.append(rms); self.frames_seen += 1
            if self.frames_seen >= self.target:
                import numpy as _np
                self.baseline = float(_np.median(_np.array(self.buf)))
            return False
        thr = max(self.baseline * self.floor_mult, self.min_rms)
        return rms >= thr

vad = webrtcvad.Vad(VAD_MODE)
