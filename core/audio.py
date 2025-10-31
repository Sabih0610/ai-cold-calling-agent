import os, time, collections, threading, socket, struct, queue, atexit, math
import numpy as np
import sounddevice as sd
import webrtcvad

# ---------- High-res timer for Windows ----------
if os.name == "nt":
    try:
        import ctypes
        _winmm = ctypes.WinDLL("winmm")
        if _winmm.timeBeginPeriod(1) == 0:
            print("[AudioSocket] Windows timer set to 1 ms")
            atexit.register(lambda: _winmm.timeEndPeriod(1))
    except Exception as e:
        print("[AudioSocket] Could not set 1 ms timer:", e)

# ---------- Optional HQ resampler ----------
try:
    import soxr
    _HAS_SOXR = True
except Exception:
    _HAS_SOXR = False
    print("[AudioSocket] soxr not found, falling back to linear resampler")

# =======================
# Global audio constants
# =======================
SAMPLE_RATE    = 16000
FRAME_DURATION = int(os.getenv("FRAME_DURATION_MS", "20"))         # 10/20/30
FRAME_SIZE     = int(SAMPLE_RATE * FRAME_DURATION / 1000)          # 320 @16k
VAD_MODE       = int(os.getenv("VAD_MODE", "2"))

# Allow WIRE_DEFAULT override; fall back to legacy AUDIOSOCKET_RATE
WIRE_DEFAULT   = int(os.getenv("WIRE_DEFAULT", os.getenv("AUDIOSOCKET_RATE", "16000")))
# Public hint for runner to seed baseline (True if 8k, False if 16k, None if unknown)
WIRE_IS_8K: bool | None = None

# ---- Mode switch: old (compat) vs new (guarded) ----
_RX_MODE = os.getenv("RX_MODE", "compat").strip().lower()  # "compat" or "guarded"
# Back-compat: treat "passthrough" as legacy alias for "compat"
if _RX_MODE == "passthrough":
    _RX_MODE = "compat"

# ---- Echo shield tuning (guarded mode only) ----
ECHO_CANCEL         = os.getenv("ECHO_CANCEL", "1") == "1"
ECHO_CORR_THRESH    = float(os.getenv("ECHO_CORR_THRESH", "0.60"))
ECHO_MAX_LAG_MS     = int(os.getenv("ECHO_MAX_LAG_MS", "160"))
ECHO_REF_WINDOW_MS  = int(os.getenv("ECHO_REF_WINDOW_MS", "400"))
ECHO_TRACK_TAU_MS   = float(os.getenv("ECHO_TRACK_TAU_MS", "240"))  # wall-clock decay
ECHO_BLOCK          = os.getenv("ECHO_BLOCK", "1") == "1"
ECHO_ATTENUATE      = float(os.getenv("ECHO_ATTENUATE", "0.25"))
ECHO_DEBUG          = os.getenv("ECHO_DEBUG", "0") == "1"

# ---- RX ducking + VAD (guarded) ----
RX_VAD          = os.getenv("RX_VAD", "1") == "1"
RX_VAD_MODE     = int(os.getenv("RX_VAD_MODE", "2"))
RX_VAD_CONSEC   = int(os.getenv("RX_VAD_CONSEC", "1"))
TX_DUCK_MS      = int(os.getenv("TX_DUCK_MS", "0"))
RX_ZERO_ON_DROP = os.getenv("RX_ZERO_ON_DROP", "0") == "1"

# ---- Debug / safety switches ----
RX_DEBUG  = os.getenv("RX_DEBUG",  "1") == "1"

# ---- TX energy gate (only duck when TX frame is audible) ----
TX_ENERGY_GATE = float(os.getenv("TX_ENERGY_GATE", "0.012"))

# =======================
# Queues & helpers
# =======================
audio_queue = collections.deque()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    pcm16 = (indata * 32768).astype(np.int16).tobytes()
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

# ---- Energy gate (median baseline) ----
class EnergyGate:
    def __init__(self,
                 frame_ms=FRAME_DURATION,
                 calibrate_ms=int(os.getenv("CALIBRATE_MS","1000")),
                 floor_mult=float(os.getenv("ENERGY_FLOOR_MULT","2.2")),
                 min_rms=float(os.getenv("ENERGY_MIN_RMS","0.0055"))):
        self.frame_ms   = frame_ms
        self.target     = max(1, calibrate_ms // frame_ms)
        self.buf        = collections.deque(maxlen=self.target)
        self.baseline   = None
        self.floor_mult = float(floor_mult)
        self.min_rms    = float(min_rms)
        self.frames_seen= 0

    @staticmethod
    def rms(frame_bytes: bytes) -> float:
        if not frame_bytes: return 0.0
        x = np.frombuffer(frame_bytes, np.int16).astype(np.float32) / 32768.0
        if x.size == 0: return 0.0
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def loud_enough(self, frame_bytes: bytes) -> bool:
        rms = self.rms(frame_bytes)
        if self.baseline is None:
            self.buf.append(rms); self.frames_seen += 1
            if self.frames_seen >= self.target:
                import numpy as _np
                self.baseline = float(_np.median(_np.array(self.buf)))
            return False
        thr = max(self.baseline * self.floor_mult, self.min_rms)
        return rms >= thr

    # New: explicit seed (so we don't wait CALIBRATE_MS)
    def seed(self, baseline_rms: float):
        if baseline_rms is not None and baseline_rms > 0:
            self.baseline = float(baseline_rms)
            self.buf.clear()
            self.frames_seen = self.target

    # Back-compat reset(baseline)
    def reset(self, baseline: float | None = None) -> None:
        if baseline is None:
            self.buf.clear()
            self.baseline = None
            self.frames_seen = 0
        else:
            self.seed(baseline)

    def threshold(self) -> float:
        if self.baseline is None:
            return self.min_rms
        return max(self.baseline * self.floor_mult, self.min_rms)

vad    = webrtcvad.Vad(VAD_MODE)

# ================================
# Resampling / audio math helpers
# ================================
def _resample_float(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32, copy=False)
    if _HAS_SOXR:
        return soxr.resample(x.astype(np.float32), sr_in, sr_out, quality="HQ")
    ratio = sr_out / float(sr_in)
    n_out = max(1, int(round(x.shape[0] * ratio)))
    src_idx = np.arange(x.shape[0], dtype=np.float32)
    dst_idx = np.linspace(0, x.shape[0]-1, n_out, dtype=np.float32)
    return np.interp(dst_idx, src_idx, x.astype(np.float32))

def resample_int16_mono(pcm_bytes: bytes, sr_in: int, sr_out: int) -> bytes:
    if sr_in == sr_out or not pcm_bytes:
        return pcm_bytes
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    y = _resample_float(x, sr_in, sr_out)
    y = np.clip(np.round(y), -32768, 32767).astype(np.int16)
    return y.tobytes()

def _downsample_16k_to_8k_int16(x16: np.ndarray) -> np.ndarray:
    if _HAS_SOXR:
        y = soxr.resample(x16.astype(np.float32), 16000, 8000, quality="HQ")
        return np.clip(np.round(y), -32768, 32767).astype(np.int16)
    n = x16.size
    if n == 0:
        return x16
    if n < 5:
        return x16[::2]
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    y = np.convolve(x16.astype(np.float32), kernel, mode='same')
    y2 = y[::2]
    return np.round(np.clip(y2, -32768, 32767)).astype(np.int16)

def _apply_gain_int16(x: np.ndarray, mul: float) -> np.ndarray:
    if mul == 1.0 or x.size == 0:
        return x
    return np.round(np.clip(x.astype(np.float32) * mul, -32768, 32767)).astype(np.int16)

def _preemphasis_int16(x: np.ndarray, a: float = 0.85) -> np.ndarray:
    if x.size == 0:
        return x
    y = x.astype(np.float32).copy()
    y[1:] = y[1:] - a * y[:-1]
    return np.round(np.clip(y, -32768, 32767)).astype(np.int16)

def _soft_limiter_with_dither_int16(x: np.ndarray, thr: float = 0.88) -> np.ndarray:
    if x.size == 0:
        return x
    xf = x.astype(np.float32) / 32768.0
    a = np.abs(xf) > thr
    xf[a] = np.sign(xf[a]) * (thr + np.tanh((np.abs(xf[a]) - thr)) * (1.0 - thr))
    d = (np.random.rand(x.size).astype(np.float32) - np.random.rand(x.size).astype(np.float32)) * (1.0/65536.0)
    xf = xf + d
    return np.round(np.clip(xf * 32768.0, -32768, 32767)).astype(np.int16)

def _rms_int16(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    xf = x.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(xf * xf) + 1e-12))

class _SlowRmsNormalizer:
    def __init__(self, target_rms=float(os.getenv("ASOCK_TX_RMS_TARGET", "0.09")),
                 alpha=0.005, min_gain=0.6, max_gain=1.5):
        self.target = float(target_rms)
        self.alpha = float(alpha)
        self.g = 1.0
        self.min_g = float(min_gain); self.max_g = float(max_gain)
    def step(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        rms = float(np.sqrt(np.mean((x.astype(np.float32)/32768.0)**2) + 1e-12))
        if rms > 1e-6:
            desired = self.target / rms
            self.g = (1.0 - self.alpha) * self.g + self.alpha * desired
            self.g = float(np.clip(self.g, self.min_g, self.max_g))
        return _apply_gain_int16(x, self.g)

# =======================
# Duplex echo shield (guarded mode)
# =======================
class _DuplexShield:
    def __init__(self):
        self.ref_len   = max(320, int(SAMPLE_RATE * ECHO_REF_WINDOW_MS / 1000))
        self.ref       = np.zeros(self.ref_len, dtype=np.int16)
        self.write_pos = 0
        self.tx_level  = 0.0
        self._last_tx_update = time.perf_counter()
        self.max_lag   = max(0, int(SAMPLE_RATE * ECHO_MAX_LAG_MS / 1000))
        self.step      = max(1, FRAME_SIZE // 4)

    def _append_ref_16k(self, x16: np.ndarray):
        n = x16.size
        if n >= self.ref_len:
            self.ref[:] = x16[-self.ref_len:]
            self.write_pos = 0
        else:
            end = min(self.ref_len - self.write_pos, n)
            self.ref[self.write_pos:end+self.write_pos] = x16[:end]
            rem = n - end
            if rem > 0:
                self.ref[0:rem] = x16[end:end+rem]
            self.write_pos = (self.write_pos + n) % self.ref_len

        now = time.perf_counter()
        dt_ms = (now - self._last_tx_update) * 1000.0
        if dt_ms > 0:
            decay = math.exp(-dt_ms / max(1.0, ECHO_TRACK_TAU_MS))
            self.tx_level *= decay
        rms = float(np.sqrt(np.mean((x16.astype(np.float32)/32768.0)**2) + 1e-12))
        self.tx_level = max(self.tx_level, rms)
        self._last_tx_update = now

    def _decayed_tx_level(self) -> float:
        now = time.perf_counter()
        dt_ms = (now - self._last_tx_update) * 1000.0
        if dt_ms <= 0:
            return self.tx_level
        decay = math.exp(-dt_ms / max(1.0, ECHO_TRACK_TAU_MS))
        return self.tx_level * decay

    @staticmethod
    def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
        af = a.astype(np.float32); bf = b.astype(np.float32)
        af -= af.mean(); bf -= bf.mean()
        na = float(np.linalg.norm(af)) + 1e-9
        nb = float(np.linalg.norm(bf)) + 1e-9
        return float(np.dot(af, bf) / (na * nb))

    def push_tx(self, frame_int16: np.ndarray, sr_hz: int):
        if not ECHO_CANCEL or frame_int16.size == 0:
            return
        if sr_hz != SAMPLE_RATE:
            from_sr = frame_int16.astype(np.float32)
            x = _resample_float(from_sr, sr_hz, SAMPLE_RATE)
            x16 = np.clip(np.round(x), -32768, 32767).astype(np.int16)
        else:
            x16 = frame_int16
        self._append_ref_16k(x16)

    def filter_rx(self, frame16: np.ndarray) -> np.ndarray | None:
        if not ECHO_CANCEL or frame16.size != FRAME_SIZE:
            return frame16
        if self._decayed_tx_level() < 0.003:
            return frame16

        R = np.concatenate([self.ref[self.write_pos:], self.ref[:self.write_pos]])
        tail_len = FRAME_SIZE + max(0, int(SAMPLE_RATE * ECHO_MAX_LAG_MS / 1000))
        tail = R[-tail_len:] if len(R) >= tail_len else R
        if tail.size < FRAME_SIZE:
            return frame16

        best = 0.0
        for lag in range(0, min(max(0, int(SAMPLE_RATE * ECHO_MAX_LAG_MS / 1000)), tail.size - FRAME_SIZE) + 1, max(1, FRAME_SIZE // 4)):
            seg = tail[tail.size - FRAME_SIZE - lag : tail.size - lag]
            c = self._norm_corr(frame16, seg)
            if c > best: best = c
            if best >= ECHO_CORR_THRESH: break

        if best >= ECHO_CORR_THRESH:
            if ECHO_DEBUG:
                print(f"[EchoShield] block corr={best:.2f} txlvl={self._decayed_tx_level():.3f}")
            if ECHO_BLOCK:
                return None
            y = np.clip(np.round(frame16.astype(np.float32) * ECHO_ATTENUATE), -32768, 32767).astype(np.int16)
            return y
        return frame16

_SHIELD = _DuplexShield()

# =======================
# TX path (agent -> caller)
# =======================
_ASOCK_TX = None
_last_tx_frame_ts = 0.0
def _mark_tx_frame():
    global _last_tx_frame_ts
    _last_tx_frame_ts = time.perf_counter()

class _AudioSocketTx:
    """
    Clocked sender with dynamic wire-rate:
      - Producer queues 16k chunks from TTS
      - Wire is 8k (NB) or 16k (WB)
      - Every 20 ms send exactly one frame (160 @8k or 320 @16k)
    """
    def __init__(self, conn: socket.socket, init_rate_hz: int = 8000):
        self.conn = conn
        self.q = queue.Queue(maxsize=256)
        self.buf = np.zeros(0, dtype=np.int16)
        self.alive = True

        self.pace_ms     = int(os.getenv("ASOCK_TX_PACE_MS", "20"))
        self.pace_s      = max(0.005, self.pace_ms / 1000.0)
        self.tx_gain     = float(os.getenv("ASOCK_TX_GAIN", "0.78"))
        self.jit_frames  = int(os.getenv("ASOCK_TX_JITTER_FRAMES", "35"))
        self.jit_cap     = int(os.getenv("ASOCK_TX_JITTER_CAP", "8"))
        self.fill_sil    = os.getenv("ASOCK_TX_SILENCE_WHEN_EMPTY", "1") == "1"
        self.use_limiter = os.getenv("ASOCK_TX_LIMITER", "0") == "1"
        self._rms_norm   = _SlowRmsNormalizer(float(os.getenv("ASOCK_TX_RMS_TARGET","0.08")))
        self._plc_mode_zero = os.getenv("ASOCK_TX_PLC_MODE", "zero").lower() == "zero"

        self.rate_hz = 8000
        self.target_samples = 160
        self.payload_len = 320
        self._last_frame = np.zeros(self.target_samples, dtype=np.int16)
        self.set_rate(init_rate_hz)

        self.t = threading.Thread(target=self._tx_loop, daemon=True)
        self.t.start()

    def set_rate(self, rate_hz: int):
        global WIRE_IS_8K
        rate_hz = 16000 if int(rate_hz) >= 16000 else 8000
        if rate_hz == self.rate_hz:
            WIRE_IS_8K = (rate_hz == 8000)
            return
        self.rate_hz = rate_hz
        self.target_samples = 320 if self.rate_hz == 16000 else 160
        self.payload_len = self.target_samples * 2
        self._last_frame = np.zeros(self.target_samples, dtype=np.int16)
        WIRE_IS_8K = (rate_hz == 8000)
        print(f"[AudioSocket][TX] Wire rate set to {self.rate_hz/1000:.0f} kHz")

    def send_16k_bytes(self, pcm16_bytes: bytes):
        if not self.alive or not pcm16_bytes:
            return
        try:
            self.q.put(pcm16_bytes, timeout=0.1)
        except queue.Full:
            pass

    def _drain_producer(self):
        got_any = False
        while True:
            try:
                chunk = self.q.get_nowait()
            except queue.Empty:
                break
            got_any = True
            x16 = np.frombuffer(chunk, dtype=np.int16)

            if self.rate_hz == 8000:
                x = _downsample_16k_to_8k_int16(x16)
                if os.getenv("ASOCK_TX_PREEMPH", "1") == "1":
                    x = _preemphasis_int16(x, float(os.getenv("ASOCK_TX_PREEMPH_A", "0.78")))
            else:
                x = x16

            if self.tx_gain != 1.0:
                x = _apply_gain_int16(x, self.tx_gain)
            x = self._rms_norm.step(x)
            if self.use_limiter:
                x = _soft_limiter_with_dither_int16(x, thr=0.88)

            self.buf = x if self.buf.size == 0 else np.concatenate([self.buf, x])
        return got_any

    def _tx_loop(self):
        next_tick = time.perf_counter()
        # Prime jitter buffer (fast start: cap the prime)
        while self.alive:
            self._drain_producer()
            need = max(0, min(self.jit_frames, self.jit_cap))
            if (self.buf.size // self.target_samples) >= need:
                break
            time.sleep(0.005)

        try:
            while self.alive:
                next_tick += self.pace_s
                self._drain_producer()

                if self.buf.size >= self.target_samples:
                    frame = self.buf[:self.target_samples]
                    self.buf  = self.buf[self.target_samples:]
                    self._last_frame = frame
                else:
                    if self.fill_sil:
                        frame = np.zeros_like(self._last_frame)
                        self._last_frame = frame
                    else:
                        frame = None

                if frame is not None:
                    try:
                        _SHIELD.push_tx(frame, self.rate_hz)
                    except Exception:
                        pass

                    payload = frame.tobytes()
                    hdr = struct.pack(">BH", 0x10, self.payload_len)
                    try:
                        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    except Exception:
                        pass
                    try:
                        self.conn.sendall(hdr + payload)
                        if _rms_int16(frame) > TX_ENERGY_GATE:
                            _mark_tx_frame()
                    except Exception:
                        self.alive = False
                        break

                # high-precision pacing
                while True:
                    now = time.perf_counter()
                    remaining = next_tick - now
                    if remaining <= 0:
                        break
                    if remaining > 0.002:
                        time.sleep(remaining - 0.0015)

        finally:
            self.alive = False

def send_tts_pcm16_16k(pcm16_bytes: bytes):
    tx = _ASOCK_TX
    if tx and tx.alive:
        tx.send_16k_bytes(pcm16_bytes)

# Pre-roll: enqueue N ms of 16k silence into TX before speaking opener
def prime_tx_silence(ms: int = 200):
    tx = _ASOCK_TX
    if not tx or not tx.alive or ms <= 0:
        return
    frames = max(1, ms // FRAME_DURATION)
    silence = (np.zeros(FRAME_SIZE, dtype=np.int16)).tobytes()
    for _ in range(frames):
        tx.send_16k_bytes(silence)

# =======================
# RX path (caller -> agent)
# =======================
audiosocket_ready = threading.Event()

def run_audiosocket_loop(stop_event,
                         host=os.getenv("AUDIOSOCKET_HOST","0.0.0.0"),
                         port=int(os.getenv("AUDIOSOCKET_PORT","9092"))):
    MSG_HANGUP = 0x00
    MSG_UUID   = 0x01
    MSG_DTMF   = 0x03
    MSG_AUDIO  = 0x10
    MSG_ERROR  = 0xFF

    def _upsample_8k_to_16k_int16(x_int16: np.ndarray) -> np.ndarray:
        if x_int16.size == 0:
            return x_int16
        if _HAS_SOXR:
            y = soxr.resample(x_int16.astype(np.float32), 8000, 16000, quality="HQ")
            return np.clip(np.round(y), -32768, 32767).astype(np.int16)
        x = x_int16.astype(np.float32)
        orig_idx = np.arange(x.shape[0], dtype=np.float32)
        up_idx = np.arange(0, x.shape[0]-1, 0.5, dtype=np.float32)
        y = np.interp(up_idx, orig_idx, x)
        return np.round(np.clip(y, -32768, 32767)).astype(np.int16)

    def _recvall(conn, n):
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    print(f"[AudioSocket] Listening on {host}:{port} ...")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    srv.settimeout(1.0)

    target_bytes_per_frame = FRAME_SIZE * 2  # 640 bytes @16k/20ms
    leftover_16k = np.zeros(0, dtype=np.int16)

    # SINGLE global declaration (keep only this one)
    global _ASOCK_TX, WIRE_IS_8K
    try:
        while not stop_event.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue

            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass

            print(f"[AudioSocket] Connected: {addr}")
            _ASOCK_TX = _AudioSocketTx(conn, init_rate_hz=WIRE_DEFAULT)
            first_audio_deadline = time.time() + (int(os.getenv("ASOCK_FIRST_AUDIO_DEADLINE_MS","600")) / 1000.0)
            rate_locked = False
            WIRE_IS_8K = None

            # per-connection meters
            vad_run = 0
            enq = drop_echo = drop_duck = drop_vad = 0
            last_report = time.time()

            try:
                leftover_16k = np.zeros(0, dtype=np.int16)
                while not stop_event.is_set():
                    hdr = _recvall(conn, 3)
                    if not hdr:
                        break
                    mtype = hdr[0]
                    mlen  = struct.unpack(">H", hdr[1:3])[0]
                    payload = _recvall(conn, mlen) if mlen else b""

                    if mtype == MSG_AUDIO and payload and mlen in (320, 640):
                        if not rate_locked:
                            if mlen == 640:
                                _ASOCK_TX.set_rate(16000)
                                print("[AudioSocket] RX detected 16 kHz wire")
                                # export flag (no nested 'global' needed)
                                WIRE_IS_8K = False
                            else:
                                _ASOCK_TX.set_rate(8000)
                                print("[AudioSocket] RX detected 8 kHz wire")
                                # export flag (no nested 'global' needed)
                                WIRE_IS_8K = True
                            rate_locked = True
                            audiosocket_ready.set()

                        # upsample 8k to 16k for STT path
                        if mlen == 320:
                            pcm8k = np.frombuffer(payload, dtype=np.int16)
                            pcm16k = _upsample_8k_to_16k_int16(pcm8k)
                        else:
                            pcm16k = np.frombuffer(payload, dtype=np.int16)

                        # Stitch then slice into 20ms @16k frames
                        if leftover_16k.size:
                            pcm16k = np.concatenate([leftover_16k, pcm16k])
                            leftover_16k = np.zeros(0, dtype=np.int16)

                        off = 0
                        total = pcm16k.size * 2  # bytes
                        raw = pcm16k.tobytes()

                        while off + target_bytes_per_frame <= total:
                            frame_bytes = raw[off:off+target_bytes_per_frame]
                            off += target_bytes_per_frame

                            if _RX_MODE == "compat":
                                audio_queue.append(frame_bytes)
                                enq += 1
                                continue

                            # guarded path
                            f16 = np.frombuffer(frame_bytes, dtype=np.int16)

                            filt = _SHIELD.filter_rx(f16)
                            if filt is None:
                                if RX_DEBUG: print("[RX] echo")
                                drop_echo += 1
                                continue

                            # TX duck
                            if (TX_DUCK_MS > 0
                                and (time.perf_counter() - _last_tx_frame_ts) * 1000.0 < TX_DUCK_MS):
                                if RX_DEBUG: print("[RX] duck")
                                drop_duck += 1
                                if RX_ZERO_ON_DROP:
                                    audio_queue.append(np.zeros_like(filt).tobytes())
                                continue

                            # RX VAD gate
                            do_enqueue = True
                            if RX_VAD:
                                rx_vad = webrtcvad.Vad(RX_VAD_MODE)
                                is_sp = rx_vad.is_speech(filt.tobytes(), SAMPLE_RATE)
                                vad_run = (vad_run + 1) if is_sp else 0
                                do_enqueue = vad_run >= int(os.getenv("RX_VAD_CONSEC", "1"))

                            if do_enqueue:
                                audio_queue.append(filt.tobytes())
                                enq += 1
                            else:
                                if RX_DEBUG: print("[RX] vad")
                                drop_vad += 1
                                if RX_ZERO_ON_DROP:
                                    audio_queue.append(np.zeros_like(filt).tobytes())

                            nowr = time.time()
                            if nowr - last_report >= 1.0:
                                print(f"[RX] enq={enq}/s drop_echo={drop_echo} drop_duck={drop_duck} drop_vad={drop_vad}")
                                enq = drop_echo = drop_duck = drop_vad = 0
                                last_report = nowr

                        if off < total:
                            rem = raw[off:total]
                            leftover_16k = np.frombuffer(rem, dtype=np.int16)

                    elif mtype == MSG_DTMF:
                        pass

                    elif mtype in (MSG_HANGUP, MSG_ERROR):
                        print(f"[AudioSocket] Call ended (type {hex(mtype)})")
                        break

                    elif mtype == MSG_UUID:
                        pass

                    if not rate_locked and time.time() > first_audio_deadline:
                        # No RX audio yet; assume default wire to start opener promptly
                        print(f"[AudioSocket] No RX audio yet; defaulting wire to {WIRE_DEFAULT//1000} kHz")
                        rate_locked = True
                        _ASOCK_TX.set_rate(WIRE_DEFAULT)
                        # export flag on default too
                        WIRE_IS_8K = (WIRE_DEFAULT < 16000)
                        audiosocket_ready.set()

            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                try:
                    audiosocket_ready.clear()
                except Exception:
                    pass
                # IMPORTANT: clear pending frames so STT doesn't print late after hangup
                try:
                    audio_queue.clear()
                except Exception:
                    pass
                if _ASOCK_TX:
                    _ASOCK_TX.alive = False
                _ASOCK_TX = None
                leftover_16k = np.zeros(0, dtype=np.int16)
                print("[AudioSocket] Disconnected.")
    finally:
        try:
            srv.close()
        except Exception:
            pass
