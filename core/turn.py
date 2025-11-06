#core\turn.py
import os, time, threading
from core import store, llm
from core.tts import speak
from core.audio import audiosocket_ready

# Coalescer state
COALESCE_GRACE_MS = int(os.getenv("COALESCE_GRACE_MS", "500"))
COALESCE_MAX_MS   = int(os.getenv("COALESCE_MAX_MS", "1500"))
HEARTBEAT_IDLE_SEC= int(os.getenv("HEARTBEAT_IDLE_SEC", "60"))
HEARTBEAT_TEXT    = os.getenv("HEARTBEAT_TEXT", "Still there? What should we do next?")

_coalesce_lock = threading.Lock()
_coalesce_timer = None
_coalesce_started_at = 0.0
_pending_user_turn = ""
_last_activity_ts = time.time()

def _now(): return time.time()

def touch_activity():
    global _last_activity_ts
    _last_activity_ts = _now()

def _cancel_coalesce_timer():
    global _coalesce_timer
    try:
        if _coalesce_timer and _coalesce_timer.is_alive():
            _coalesce_timer.cancel()
    except Exception:
        pass
    finally:
        _coalesce_timer = None

def _coalesce_fire():
    global _pending_user_turn, _coalesce_started_at
    with _coalesce_lock:
        text = _pending_user_turn.strip()
        _pending_user_turn = ""
        _coalesce_started_at = 0.0
        _cancel_coalesce_timer()
    if text:
        llm.enqueue_user_text(text)

def _schedule_coalesce_fire(delay_ms: int):
    global _coalesce_timer
    _cancel_coalesce_timer()
    _coalesce_timer = threading.Timer(delay_ms/1000.0, _coalesce_fire)
    _coalesce_timer.daemon = True
    _coalesce_timer.start()

def commit_user_turn(new_text: str):
    global _pending_user_turn, _coalesce_started_at
    if not new_text: return
    with _coalesce_lock:
        now = _now()
        if not _pending_user_turn:
            _pending_user_turn = new_text.strip()
            _coalesce_started_at = now
        else:
            if not _pending_user_turn.endswith((".", "?", "!")) and not new_text.lstrip().startswith((",", "and", "but")):
                _pending_user_turn += " "
            _pending_user_turn += new_text.strip()
        elapsed = int((now - _coalesce_started_at) * 1000.0)
        remaining = max(0, COALESCE_MAX_MS - elapsed)
        _schedule_coalesce_fire(min(COALESCE_GRACE_MS, remaining))

def heartbeat_worker():
    while not store.STOP.is_set():
        try:
            idle = _now() - _last_activity_ts
            if (idle >= HEARTBEAT_IDLE_SEC
                and not llm.tts_playing.is_set()
                and llm.llm_queue.empty()
                and audiosocket_ready.is_set()):
                touch_activity()
                speak(HEARTBEAT_TEXT)
            time.sleep(1.0)
        except Exception:
            time.sleep(1.0)
