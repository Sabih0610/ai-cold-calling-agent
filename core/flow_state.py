import threading

class FlowState:
    def __init__(self):
        self.stage_idx = 0
        self.identity_ack = False
        self.last_question = None
        self.discovery_hits = 0
        self.close_ready = False

_STATE = FlowState()
_LOCK = threading.Lock()

def reset():
    with _LOCK:
        _STATE.stage_idx = 0
        _STATE.identity_ack = False
        _STATE.last_question = None
        _STATE.discovery_hits = 0
        _STATE.close_ready = False

def mark_identity():
    with _LOCK:
        _STATE.identity_ack = True

def identity_handled():
    with _LOCK:
        return _STATE.identity_ack

def advance_stage():
    with _LOCK:
        _STATE.stage_idx += 1

def set_stage(idx: int):
    with _LOCK:
        _STATE.stage_idx = max(0, idx)

def current_stage_idx():
    with _LOCK:
        return _STATE.stage_idx

def record_question(q: str):
    with _LOCK:
        _STATE.last_question = q

def last_question():
    with _LOCK:
        return _STATE.last_question

def increment_discovery():
    with _LOCK:
        _STATE.discovery_hits += 1

def discovery_count():
    with _LOCK:
        return _STATE.discovery_hits

def mark_close_ready():
    with _LOCK:
        _STATE.close_ready = True

def is_close_ready():
    with _LOCK:
        return _STATE.close_ready
