# core/tts.py
import os, time, json, shutil, subprocess, threading, tempfile
from uuid import uuid4
import numpy as np
import sounddevice as sd
from core.utils import to_speakable, set_user_identity_async
from core import store

# === NEW: AudioSocket sink helpers ===
from core.audio import (
    send_tts_pcm16_16k,
    audiosocket_ready,
    resample_int16_mono,
    FRAME_SIZE,
    FRAME_DURATION,
    flush_tts_tx,  # NEW: used to drop buffered TX audio on barge-in
)

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
LENGTH_SCALE    = float(os.getenv("PIPER_LENGTH_SCALE", "1.08"))
NOISE_SCALE     = float(os.getenv("PIPER_NOISE_SCALE",  "0.45"))
NOISE_W         = float(os.getenv("PIPER_NOISE_W",      "0.55"))

TTS_BARGE_IN_PROTECT_MS = int(os.getenv("TTS_BARGE_IN_PROTECT_MS", "700"))
ECHO_TRACK_DECAY        = float(os.getenv("ECHO_TRACK_DECAY", "0.93"))

# Where TTS should go: "audiosocket" | "speakers" | "both"
AUDIO_SINK = os.getenv("AUDIO_SINK", "audiosocket").lower()
TTS_WAIT_FOR_SOCKET = os.getenv("TTS_WAIT_FOR_SOCKET", "1") != "0"

# Streaming controls
TTS_STREAMING_ENABLED = os.getenv("TTS_STREAMING", "1") not in ("0", "false", "False")
TTS_STREAM_MIN_CHARS = int(os.getenv("TTS_STREAM_MIN_CHARS", "22"))
TTS_STREAM_FLUSH_MS = int(os.getenv("TTS_STREAM_FLUSH_MS", "420"))

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
        self._duck_lock = threading.Lock()
        self._duck_target = 1.0
        self._duck_current = 1.0
        self._soft_interrupt_flag = threading.Event()

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

        _tlog(f"[Piper] exe={PIPER_EXE} which={shutil.which(PIPER_EXE)}")
        _tlog(f"[Piper] model={self._model_path} exists={os.path.exists(self._model_path)}")
        _tlog(f"[Piper] config={self._config_path} exists={os.path.exists(self._config_path)}")

        if not (shutil.which(PIPER_EXE) and os.path.exists(self._model_path) and os.path.exists(self._config_path)):
            _tlog("🔊 Piper not available; audio disabled.")
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

                ducked = self._apply_ducking(data16)
                data16 = ducked

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
                        self.stream.write(self._apply_ducking(data))
                    except Exception:
                        pass

        except Exception:
            pass

    def _apply_ducking(self, pcm16: bytes) -> bytes:
        """
        Apply gentle gain changes so we can duck or fade out without audible clicks.
        """
        with self._duck_lock:
            target = self._duck_target
            current = self._duck_current
            if abs(current - target) <= 0.02:
                self._duck_current = target
            else:
                step = 0.08 if current > target else 0.12
                self._duck_current = max(target, current - step) if current > target else min(target, current + step)
            gain = self._duck_current
        if gain >= 0.995:
            return pcm16
        arr = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
        arr *= gain
        np.clip(arr, -32768, 32767, out=arr)
        return arr.astype(np.int16).tobytes()

    def request_duck(self, level: float = 0.4):
        with self._duck_lock:
            self._duck_target = float(min(max(level, 0.05), 1.0))

    def release_duck(self):
        with self._duck_lock:
            self._duck_target = 1.0

    def soft_interrupt(self, fade_ms: int = 450):
        if not tts_playing.is_set() or self._soft_interrupt_flag.is_set():
            return
        self._soft_interrupt_flag.set()

        def _fade_and_restart():
            try:
                self.request_duck(0.05)
                time.sleep(max(0.05, fade_ms / 1000.0))
                tts_interrupt()
            finally:
                self.release_duck()
                self._soft_interrupt_flag.clear()

        threading.Thread(target=_fade_and_restart, daemon=True).start()

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
                try:
                    self.stream.abort()
                except Exception:
                    pass
                try:
                    self.stream.close()
                except Exception:
                    pass
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
class _StreamSession:
    def __init__(self, session_id: str | None):
        self.session_id = session_id or f"tts_{uuid4().hex[:8]}"
        self.buffer: list[str] = []
        self.pending_chars = 0
        self.lock = threading.Lock()
        self.last_flush = time.time()
        self.active = True

    def _flush_locked(self) -> str:
        chunk = "".join(self.buffer).strip()
        self.buffer.clear()
        self.pending_chars = 0
        self.last_flush = time.time()
        return chunk

    def append(self, text: str, force: bool = False) -> bool:
        if not self.active or not text:
            return False
        chunk = None
        now = time.time()
        with self.lock:
            self.buffer.append(text)
            self.pending_chars += len(text)
            should_flush = force
            if not should_flush:
                if self.pending_chars >= TTS_STREAM_MIN_CHARS:
                    should_flush = True
                elif text[-1:] in (".", "?", "!"):
                    should_flush = True
                elif (now - self.last_flush) * 1000.0 >= TTS_STREAM_FLUSH_MS:
                    should_flush = True
            if should_flush:
                chunk = self._flush_locked()
        if chunk:
            _engine.say(chunk)
            store.publish_event({
                "type": "state",
                "name": "tts_stream_chunk",
                "stream_id": self.session_id,
                "text": chunk,
            })
            return True
        return False

    def flush(self) -> bool:
        with self.lock:
            if not self.buffer:
                return False
            chunk = self._flush_locked()
        if chunk:
            _engine.say(chunk)
            store.publish_event({
                "type": "state",
                "name": "tts_stream_chunk",
                "stream_id": self.session_id,
                "text": chunk,
                "final": True,
            })
            return True
        return False

    def close(self):
        self.active = False


_engine = PiperEngine()
_started = False
_stream_lock = threading.Lock()
_active_stream: _StreamSession | None = None


def init_engine():
    global _started
    if not _started:
        _started = _engine.start()


def use_voice(voice_entry: dict):
    """Switch Piper model dynamically."""
    model = voice_entry.get("model")
    config = voice_entry.get("config")
    _engine.restart(model_path=model, config_path=config)


def _current_voice_paths():
    model = getattr(_engine, "_model_path", None)
    config = getattr(_engine, "_config_path", None)
    if not (model and config):
        model, config = _select_voice_paths(PIPER_VOICE_SELECT)
    return model, config


def begin_stream(session_id: str | None = None) -> str | None:
    if not TTS_STREAMING_ENABLED:
        return None
    with _stream_lock:
        global _active_stream
        if _active_stream:
            try:
                _active_stream.flush()
            except Exception:
                pass
            _active_stream.close()
            _active_stream = None
        _active_stream = _StreamSession(session_id)
        return _active_stream.session_id


def stream_text(stream_id: str | None, text: str, force_flush: bool = False) -> bool:
    if not text:
        return False
    if not TTS_STREAMING_ENABLED or not stream_id:
        speak(text)
        return True
    with _stream_lock:
        session = _active_stream if (_active_stream and _active_stream.session_id == stream_id) else None
    if not session:
        new_id = begin_stream(stream_id)
        if not new_id:
            speak(text)
            return True
        with _stream_lock:
            session = _active_stream
    return session.append(text, force=force_flush) if session else False


def end_stream(stream_id: str | None, final_text: str | None = None):
    if not TTS_STREAMING_ENABLED or not stream_id:
        if final_text:
            speak(final_text)
        return
    with _stream_lock:
        global _active_stream
        session = _active_stream if (_active_stream and _active_stream.session_id == stream_id) else None
        _active_stream = None
    if session:
        try:
            session.flush()
        except Exception:
            pass
        session.close()
    if final_text:
        speak(final_text)


def abort_stream(stream_id: str | None = None):
    if not TTS_STREAMING_ENABLED:
        return
    with _stream_lock:
        global _active_stream
        session = _active_stream
        if session and stream_id and session.session_id != stream_id:
            session = None
        if session:
            session.close()
        _active_stream = None


def synthesize_to_pcm16(text: str) -> bytes:
    """Generate 16 kHz PCM bytes for the given text using Piper CLI."""
    s = to_speakable(text)
    if not s:
        return b""
    model_path, config_path = _current_voice_paths()
    if not (shutil.which(PIPER_EXE) and os.path.exists(model_path) and os.path.exists(config_path)):
        return b""

    sr = _load_voice_sr(config_path)
    ls, ns, nw = _derive_params(s)
    output_path = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt") as tf:
            tf.write(s)
            text_path = tf.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".raw") as of:
            output_path = of.name
        args = [
            PIPER_EXE,
            "--model", model_path,
            "--config", config_path,
            "--input_text", text_path,
            "--output_raw",
            "--output_file", output_path,
            "--length-scale", f"{ls}",
            "--noise-scale", f"{ns}",
            "--noise-w", f"{nw}",
        ]
        if PIPER_USE_CUDA:
            args.append("--cuda")
        result = subprocess.run(
            args,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"⚠️ Piper synth failed ({result.returncode}): {result.stderr.decode(errors='ignore')}")
            return b""
        try:
            with open(output_path, "rb") as f:
                raw = f.read()
        except Exception as read_exc:
            print(f"⚠️ Piper synth read error: {read_exc}")
            raw = b""
        if not raw:
            err_text = (result.stderr or b"").decode(errors="ignore").strip()
            if err_text:
                print(f"⚠️ Piper synth produced no data. STDERR: {err_text}")
            return b""
        if sr != 16000:
            return resample_int16_mono(raw, sr, 16000)
        return raw
    except Exception as exc:
        print(f"⚠️ Piper synth exception: {exc}")
        return b""
    finally:
        try:
            if 'text_path' in locals():
                os.unlink(text_path)
            if output_path:
                os.unlink(output_path)
        except Exception:
            pass


def play_pcm16_buffer(pcm_bytes: bytes):
    """Stream pre-generated PCM16 (16 kHz) to the current audio sink."""
    if not pcm_bytes:
        return
    frame_bytes = FRAME_SIZE * 2
    if AUDIO_SINK in ("audiosocket", "both"):
        audiosocket_ready.wait(timeout=0.75)
    total = len(pcm_bytes)
    idx = 0
    global _tts_started_at
    _tts_started_at = time.time()
    tts_playing.set()
    try:
        store.publish_event({"type": "state", "name": "tts_playing", "value": True})
    except Exception:
        pass
    start = time.perf_counter()
    frame_dur = FRAME_DURATION / 1000.0
    frame_idx = 0
    while idx < total:
        chunk = pcm_bytes[idx:idx + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
        send_tts_pcm16_16k(chunk)
        idx += frame_bytes
        frame_idx += 1
        target = start + frame_idx * frame_dur
        remaining = target - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
    tts_playing.clear()
    try:
        store.publish_event({"type": "state", "name": "tts_playing", "value": False})
    except Exception:
        pass


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
    # Stop any active streaming session
    abort_stream(None)

    # 🔹 NEW: flush any audio that is already buffered to go out to the caller
    try:
        flush_tts_tx()
    except Exception:
        pass

    # Restart Piper so no new audio is generated
    try:
        _engine.restart()
    except Exception:
        try:
            _engine.start()
        except Exception:
            print("⚠️ Piper restart failed.")

    # Clear state flags
    try:
        tts_playing.clear()
    except Exception:
        pass
    try:
        store.publish_event({"type": "state", "name": "tts_playing", "value": False})
    except Exception:
        pass
    try:
        _engine.release_duck()
    except Exception:
        pass


def tts_shutdown():
    try:
        _engine.stop()
    except Exception:
        pass


def get_tts_started_ts():
    return float(globals().get("_tts_started_at", 0.0))


def request_duck(level: float = 0.4):
    try:
        _engine.request_duck(level)
    except Exception:
        pass


def release_duck(delay_ms: int = 0):
    def _release():
        try:
            _engine.release_duck()
        except Exception:
            pass
    if delay_ms > 0:
        threading.Timer(delay_ms / 1000.0, _release).start()
    else:
        _release()


def tts_soft_interrupt(fade_ms: int = 450):
    try:
        _engine.soft_interrupt(fade_ms=fade_ms)
    except Exception:
        tts_interrupt()
    abort_stream(None)


# ===============================================================
# Optional Startup Voice Test
# ===============================================================
if __name__ == "__main__":
    print("🔊 Initializing Piper engine test...")
    _engine.start()
    speak("Voice check. If you hear this, Piper is working.")

LOG_TTS_EVENTS = os.getenv("LOG_TTS_EVENTS", "1") not in ("0","false","False")


def _tlog(msg):
    if LOG_TTS_EVENTS:
        print(msg)
