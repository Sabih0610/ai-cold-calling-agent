# core\backchannel.py
import os
import random
import time
from core.tts import speak, tts_playing

_PHRASES = [p.strip() for p in os.getenv(
    "BACKCHANNEL_PHRASES",
    "yeah,mm-hmm,uh huh,sure,gotcha,absolutely"
).split(",") if p.strip()]
_MIN_INTERVAL = float(os.getenv("BACKCHANNEL_MIN_INTERVAL", "4.0"))
_TRIGGER_MS = int(os.getenv("BACKCHANNEL_TRIGGER_MS", "1800"))
_ENABLE = os.getenv("BACKCHANNEL_ENABLE", "1") not in ("0", "false", "False")
_last_ack_ts = 0.0
_turn_active = False

def reset_turn():
    global _turn_active
    _turn_active = False

def maybe_ack(user_ms: int):
    global _last_ack_ts, _turn_active
    if not _ENABLE or not _PHRASES or _turn_active:
        return
    if user_ms < _TRIGGER_MS:
        return
    now = time.time()
    if now - _last_ack_ts < _MIN_INTERVAL:
        return
    if tts_playing.is_set():
        return
    phrase = random.choice(_PHRASES)
    speak(phrase)
    _last_ack_ts = now
    _turn_active = True
