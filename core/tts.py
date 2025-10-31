# core/tts.py
import os, time, json, shutil, subprocess, threading
import sounddevice as sd
from core.utils import to_speakable, set_user_identity_async
from core import store

# === NEW: AudioSocket sink helpers ===
from core.audio import send_tts_pcm16_16k, audiosocket_ready, resample_int16_mono

# ===============================================================
# Piper Configuration
# ===============================================================
VOICES_BASE     = os.getenv("PIPER_VOICES_BASE", r"F:\ai cold calling agent\voices")

FEMALE_MODEL    = os.getenv(
    "PIPER_FEMALE_MODEL",
    os.path.join(VOICES_BASE, "female_en", "en_US-hfc_female-medium.onnx"),
)
FEMALE_CONFIG   = os.getenv(
    "PIPER_FEMALE_CONFIG",
    os.path.join(VOICES_BASE, "female_en", "en_US-hfc_female-medium.onnx.json"),
)

MALE_MODEL      = os.getenv(
    "PIPER_MALE_MODEL",
    os.path.join(VOICES_BASE, "male_en", "en_US-bryce-medium.onnx"),
)
MALE_CONFIG     = os.getenv(
    "PIPER_MALE_CONFIG",
    os.path.join(VOICES_BASE, "male_en", "en_US-bryce-medium.onnx.json"),
)

PIPER_VOICE_SELECT = os.getenv("PIPER_VOICE", "female").lower()
PIPER_EXE       = os.getenv("PIPER_EXE", "piper")
PIPER_USE_CUDA  = os.getenv("PIPER_CUDA", "0") not in ("0", "false", "False")

PIPER_CHUNK     = int(os.getenv("PIPER_CHUNK", "4096"))
LENGTH_SCALE    = float(os.getenv("PIPER_LENGTH_SCALE", "0.95"))
NOISE_SCALE     = float(os.getenv("PIPER_NOISE_SCALE",  "0.45"))
NOISE_W         = float(os.getenv("PIPER_NOISE_W",      "0.55"))

TTS_BARGE_IN_PROTECT_MS = int(os.getenv("TTS_BARGE_IN_PROTECT_MS", "700"))
ECHO_TRACK_DECAY        = float(os.getenv("ECHO_TRACK_DECAY", "0.93"))

# Where TTS should go: "audiosocket" | "speakers" | "both"
AUDIO_SINK = os.getenv("AUDIO_SINK", "audiosocket").lower()
TTS_WAIT_FOR_SOCKET = os.getenv("TTS_WAIT_FOR_SOCKET", "1") != "0"

# Playback state
tts_playing = threading.Event()
_tts_started_at = 0.0


# ===============================================================
# Helper Functions
# ===============================================================
def _select_voice_paths(default_sel: str):
    if default_sel == "male":
        return MALE_MODEL, MALE_CONFIG
    return FEMALE_MODEL, FEMALE_CONFIG


def _load_voice_sr(config_path: str) -> int:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg.get("sample_rate", 22050))
    except Exception:
        return 22050


def _derive_params(text: str):
    """Adjust speaking style slightly based on text tone."""
    ls = LENGTH_SCALE
    ns = NOISE_SCALE
    nw = NOISE_W
    t = (text or "").strip()

    if "!" in t and len(t) < 140:
        ls = max(0.85, LENGTH_SCALE - 0.05)
        ns = min(0.70, NOISE_SCALE + 0.15)
    if t.lower().startswith(("sorry", "i’m sorry", "im sorry", "unfortunately")):
        ls = min(1.10, LENGTH_SCALE + 0.15)
        ns = max(0.30, NOISE_SCALE - 0.10)
    return ls, ns, nw


# ===============================================================
# Piper Engine Class
# ===============================================================
class PiperEngine:
    def __init__(self):
        self.proc = None
        self.stream = None               # speakers stream (only if needed)
        self.reader = None
        self.stop_reader = threading.Event()
        self.sr = 22050
        self.args = None
        self._io_lock = threading.Lock()
        self._last_data_ts = 0.0
        self._model_path = None
        self._config_path = None

    # ---------------------------
    # STDERR Reader for Debugging
    # ---------------------------
    def _stderr_tap(self, proc):
        try:
            for line in iter(proc.stderr.readline, b''):
                if not line:
                    break
                print("[Piper STDERR]", line.decode(errors="ignore").rstrip())
        except Exception:
            pass

    # ---------------------------
    # Start / Stop / Restart
    # ---------------------------
    def start(self, model_path=None, config_path=None):
        if model_path and config_path:
            self._model_path, self._config_path = model_path, config_path
        elif not (self._model_path and self._config_path):
            self._model_path, self._config_path = _select_voice_paths(PIPER_VOICE_SELECT)

        print(f"[Piper] exe={PIPER_EXE} which={shutil.which(PIPER_EXE)}")
        print(f"[Piper] model={self._model_path} exists={os.path.exists(self._model_path)}")
        print(f"[Piper] config={self._config_path} exists={os.path.exists(self._config_path)}")

        if not (shutil.which(PIPER_EXE) and os.path.exists(self._model_path) and os.path.exists(self._config_path)):
            print("🔊 Piper not available; audio disabled.")
            return False

        self.sr = _load_voice_sr(self._config_path)
        ls, ns, nw = _derive_params("")

        self.args = [
            PIPER_EXE, "--model", self._model_path, "--config", self._config_path,
            "--output_raw", "--length-scale", f"{ls}",
            "--noise-scale", f"{ns}", "--noise-w", f"{nw}"
        ]
        if PIPER_USE_CUDA:
            self.args.append("--cuda")

        self.proc = subprocess.Popen(
            self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        threading.Thread(target=self._stderr_tap, args=(self.proc,), daemon=True).start()

        # Only open local speakers if requested
        if AUDIO_SINK in ("speakers", "both"):
            try:
                self.stream = sd.RawOutputStream(samplerate=self.sr, channels=1, dtype="int16")
                self.stream.start()
            except Exception as e:
                print(f"⚠️ Could not open speaker output: {e}")
                self.stream = None
        else:
            self.stream = None

        # Launch reader thread
        self.stop_reader.clear()
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()
        return True

    def _read_loop(self):
        """Continuously read Piper audio output and route to sink(s)."""
        try:
            first_chunk_sent = False
            while not self.stop_reader.is_set():
                if self.proc is None or self.proc.stdout is None:
                    time.sleep(0.005)
                    continue

                data = self.proc.stdout.read(PIPER_CHUNK)  # PCM16 @ self.sr
                now = time.time()

                if not data:
                    if tts_playing.is_set() and (now - self._last_data_ts) > 0.6:
                        tts_playing.clear()
                    time.sleep(0.002)
                    continue

                self._last_data_ts = now
                tts_playing.set()

                # Ensure 16k for AudioSocket (the call side)
                data16 = data if self.sr == 16000 else resample_int16_mono(data, self.sr, 16000)

                # Gate the FIRST audible chunk until AudioSocket is connected
                if AUDIO_SINK in ("audiosocket", "both"):
                    if not first_chunk_sent and TTS_WAIT_FOR_SOCKET:
                        audiosocket_ready.wait(timeout=5.0)  # returns immediately if already connected
                    send_tts_pcm16_16k(data16)
                    first_chunk_sent = True

                # Optional speakers (for local monitoring)
                if AUDIO_SINK in ("speakers", "both") and self.stream:
                    try:
                        # Speakers expect Piper's native sample rate:
                        self.stream.write(data)
                    except Exception:
                        pass

        except Exception:
            pass

    # ---------------------------
    # Speaking
    # ---------------------------
    def say(self, text: str):
        """Send text to Piper for speech."""
        global _tts_started_at
        s = to_speakable(text)
        if not s or self.proc is None or self.proc.stdin is None:
            return
        with self._io_lock:
            try:
                self.proc.stdin.write((s + "\n").encode("utf-8"))
                self.proc.stdin.flush()
                _tts_started_at = time.time()
                self._last_data_ts = _tts_started_at
                tts_playing.set()
                store.publish_event({"type": "state", "name": "tts_playing", "value": True})
            except Exception:
                self.restart()
                try:
                    self.proc.stdin.write((s + "\n").encode("utf-8"))
                    self.proc.stdin.flush()
                    _tts_started_at = time.time()
                    self._last_data_ts = _tts_started_at
                    tts_playing.set()
                    store.publish_event({"type": "state", "name": "tts_playing", "value": True})
                except Exception:
                    pass

    # ---------------------------
    # Stop & Restart
    # ---------------------------
    def stop(self):
        tts_playing.clear()
        try:
            self.stop_reader.set()
        except Exception:
            pass
        try:
            if self.stream:
                try: self.stream.abort()
                except Exception: pass
                try: self.stream.close()
                except Exception: pass
        finally:
            self.stream = None
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass
        finally:
            self.proc = None

    def restart(self, model_path=None, config_path=None):
        self.stop()
        time.sleep(0.03)
        self.start(model_path=model_path, config_path=config_path)


# ===============================================================
# Global Engine & Utility Wrappers
# ===============================================================
_engine = PiperEngine()
_started = False


def init_engine():
    global _started
    if not _started:
        _started = _engine.start()


def use_voice(voice_entry: dict):
    """Switch Piper model dynamically."""
    model = voice_entry.get("model")
    config = voice_entry.get("config")
    _engine.restart(model_path=model, config_path=config)


def speak(text: str, user_text_for_name_detect: str = None):
    """Say text aloud, restarting Piper if needed."""
    if not text:
        return

    if _engine.proc is None:
        try:
            _engine.start()
        except Exception:
            print("⚠️ Piper failed to start on demand.")
            return

    try:
        _engine.say(text)
    except Exception as e:
        print(f"⚠️ Piper speak error: {e}")
        try:
            _engine.restart()
            _engine.say(text)
        except Exception:
            print("⚠️ Piper restart failed permanently.")

    if user_text_for_name_detect:
        threading.Thread(
            target=set_user_identity_async,
            args=(user_text_for_name_detect,),
            daemon=True
        ).start()


def tts_interrupt():
    """Forcefully restarts Piper voice engine if barge-in or stuck."""
    try:
        _engine.restart()
    except Exception:
        try:
            _engine.start()
        except Exception:
            print("⚠️ Piper restart failed.")
    try:
        store.publish_event({"type": "state", "name": "tts_playing", "value": False})
    except Exception:
        pass


def tts_shutdown():
    try:
        _engine.stop()
    except Exception:
        pass


def get_tts_started_ts():
    return float(globals().get("_tts_started_at", 0.0))


# ===============================================================
# Optional Startup Voice Test
# ===============================================================
if __name__ == "__main__":
    print("🔊 Initializing Piper engine test...")
    _engine.start()
    speak("Voice check. If you hear this, Piper is working.")
