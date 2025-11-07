import os
import wave
import threading
import time
import json
import numpy as np

_lock = threading.Lock()
_current = None


def _recordings_dir() -> str:
    base = os.getenv("RECORDINGS_DIR")
    if not base:
        base = os.path.join(os.getcwd(), "data", "recordings")
    os.makedirs(base, exist_ok=True)
    return base


class _Recorder:
    def __init__(self, basename: str, sample_rate: int = 16000):
        self.basename = basename or f"call_{int(time.time())}"
        self.sample_rate = sample_rate
        directory = _recordings_dir()
        self.wav_path = os.path.join(directory, f"{self.basename}.wav")
        self._rx_path = os.path.join(directory, f"{self.basename}_rx.raw")
        self._tx_path = os.path.join(directory, f"{self.basename}_tx.raw")
        self._rx_file = open(self._rx_path, "ab")
        self._tx_file = open(self._tx_path, "ab")
        self._closed = False

    def write_rx(self, frame: bytes) -> None:
        if not frame or self._closed:
            return
        try:
            self._rx_file.write(frame)
        except Exception:
            pass

    def write_tx(self, frame: bytes) -> None:
        if not frame or self._closed:
            return
        try:
            self._tx_file.write(frame)
        except Exception:
            pass

    def stop(self) -> str:
        if self._closed:
            return self.basename, self.wav_path
        self._closed = True
        try:
            self._rx_file.close()
        except Exception:
            pass
        try:
            self._tx_file.close()
        except Exception:
            pass
        try:
            _mix_to_wav(self._rx_path, self._tx_path, self.wav_path, self.sample_rate)
        finally:
            for path in (self._rx_path, self._tx_path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        return self.basename, self.wav_path


def _mix_to_wav(rx_path: str, tx_path: str, wav_path: str, sample_rate: int) -> None:
    try:
        rx = np.fromfile(rx_path, dtype=np.int16)
    except Exception:
        rx = np.zeros(0, dtype=np.int16)
    try:
        tx = np.fromfile(tx_path, dtype=np.int16)
    except Exception:
        tx = np.zeros(0, dtype=np.int16)

    length = max(rx.size, tx.size)
    if length == 0:
        # create empty wav file for consistency
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"")
        return

    if rx.size < length:
        rx = np.pad(rx, (0, length - rx.size))
    if tx.size < length:
        tx = np.pad(tx, (0, length - tx.size))

    stereo = np.stack((rx, tx), axis=1).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())


def start(basename: str | None, sample_rate: int = 16000) -> None:
    global _current
    with _lock:
        if _current is not None:
            try:
                _current.stop()
            except Exception:
                pass
        _current = _Recorder(basename or f"call_{int(time.time())}", sample_rate)


def write_rx(frame: bytes) -> None:
    with _lock:
        if _current is not None:
            _current.write_rx(frame)


def write_tx(frame: bytes) -> None:
    with _lock:
        if _current is not None:
            _current.write_tx(frame)


def stop() -> tuple[str, str] | None:
    global _current
    with _lock:
        if _current is None:
            return None
        try:
            info = _current.stop()
        finally:
            _current = None
        return info


def current_wav_path() -> str | None:
    with _lock:
        if _current is None:
            return None
        return _current.wav_path


def save_transcript(basename: str, transcript: list) -> str | None:
    if not basename:
        return None
    path = os.path.join(_recordings_dir(), f"{basename}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(transcript or [], f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return None


def get_recording_duration_secs(basename: str) -> tuple[bool, float, str]:
    if not basename:
        return False, 0.0, ""
    path = os.path.join(_recordings_dir(), f"{basename}.wav")
    if not os.path.exists(path):
        return False, 0.0, path
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            duration = frames / float(rate)
        return True, float(duration), path
    except Exception:
        return False, 0.0, path
