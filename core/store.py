#core\store.py
import os, json, time, queue, threading, uuid
import redis
from dotenv import load_dotenv
from core.utils import slug, safe_write_json, load_json, utcnow_iso

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
LOG_ROOT = os.getenv("LOG_DIR", "logs/users")
os.makedirs(LOG_ROOT, exist_ok=True)

rds = redis.from_url(REDIS_URL, decode_responses=True)

# STOP flag (global across modules)
STOP = threading.Event()
STOP_REASON = ""

def set_stop(reason: str):
    global STOP_REASON
    STOP_REASON = reason or "unspecified"
    STOP.set()

# ------------- Conversation (Redis list semantics) -------------
class Conversation:
    def __init__(self, key=None, max_turns=int(os.getenv("CONV_TURNS","8")), ttl=3600):
        if key is None: key = f"unknown_{uuid.uuid4().hex[:6]}"
        self.key = f"conv:{key}"
        self.max_turns = max_turns
        self.ttl = ttl
        self._lock = threading.Lock()
    def load(self):
        try:
            data = rds.lrange(self.key, 0, -1)
            return [json.loads(x) for x in data]
        except Exception:
            return []
    def append(self, role, content):
        msg = {"role": role, "content": content}
        try:
            with self._lock:
                pipe = rds.pipeline()
                pipe.rpush(self.key, json.dumps(msg, ensure_ascii=False))
                pipe.ltrim(self.key, -2*self.max_turns, -1)
                pipe.expire(self.key, self.ttl)
                pipe.execute()
        except Exception:
            pass
    def clear(self):
        try: rds.delete(self.key)
        except Exception: pass

conv = Conversation()

# ------------- File IO / session logging (moved as-is) -------------
io_queue = queue.Queue()

class IOState:
    def __init__(self):
        self.user_name = None
        self.user_key  = None
        self.user_dir  = None
        self.user_path = None
        self.root = None
        self.session_id = uuid.uuid4().hex[:12]
        self.session = {
            "session_id": self.session_id,
            "started_at": utcnow_iso(),
            "ended_at": None,
            "turns": []
        }
        self.session_path = None

IO = IOState()

def _user_file_skeleton(name, key):
    return {
        "user": name, "user_key": key,
        "created_at": utcnow_iso(),
        "updated_at": None,
        "sessions": []
    }

def _write_session_file():
    if not (IO.user_dir and IO.session_id): return
    if IO.session_path is None:
        IO.session_path = os.path.join(IO.user_dir, f"{IO.session_id}.json")
    payload = {"user": IO.user_name, "user_key": IO.user_key, "session": IO.session}
    safe_write_json(IO.session_path, payload)

def io_worker():
    while not STOP.is_set():
        try:
            ev = io_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            typ = ev.get("type")

            if typ == "identify":
                name = (ev["name"] or "").strip()
                if not IO.user_name and name:
                    IO.user_name = name
                    IO.user_key  = slug(name)
                    IO.user_dir  = os.path.join(LOG_ROOT, IO.user_key)
                    os.makedirs(IO.user_dir, exist_ok=True)
                    IO.user_path = os.path.join(LOG_ROOT, f"{IO.user_key}.json")

                    if os.path.exists(IO.user_path):
                        with open(IO.user_path, "r", encoding="utf-8") as f:
                            IO.root = json.load(f)
                    else:
                        IO.root = _user_file_skeleton(IO.user_name, IO.user_key)
                    if "sessions" not in IO.root:
                        IO.root["sessions"] = []

                    IO.root["sessions"].append(dict(IO.session))
                    IO.root["updated_at"] = utcnow_iso()
                    safe_write_json(IO.user_path, IO.root)
                    _write_session_file()

            elif typ == "turn":
                IO.session["turns"].append({"ts": ev["ts"], "role": ev["role"], "content": ev["content"]})
                if IO.user_path:
                    if not IO.root["sessions"] or IO.root["sessions"][-1]["session_id"] != IO.session["session_id"]:
                        IO.root["sessions"].append({"session_id": IO.session["session_id"],
                                                    "started_at": IO.session["started_at"],
                                                    "ended_at": None, "turns": []})
                    IO.root["sessions"][-1] = dict(IO.session)
                    IO.root["updated_at"] = utcnow_iso()
                    safe_write_json(IO.user_path, IO.root)
                    _write_session_file()

            elif typ == "persist_learn":
                # handled in enqueue_learn_async, persisted again here
                pass

            elif typ == "close_session":
                if IO.session["ended_at"] is None:
                    IO.session["ended_at"] = utcnow_iso()
                if IO.user_path:
                    if IO.root["sessions"] and IO.root["sessions"][-1]["session_id"] == IO.session["session_id"]:
                        IO.root["sessions"][-1] = dict(IO.session)
                    else:
                        IO.root["sessions"].append(dict(IO.session))
                    IO.root["updated_at"] = utcnow_iso()
                    safe_write_json(IO.user_path, IO.root)
                    _write_session_file()
                else:
                    unknown = os.path.join(LOG_ROOT, f"unknown_{IO.session['session_id']}.json")
                    safe_write_json(unknown, {
                        "user": None, "user_key": None,
                        "created_at": utcnow_iso(),
                        "updated_at": utcnow_iso(),
                        "sessions": [dict(IO.session)]
                    })
        except Exception as e:
            print("⚠️ IO worker error:", e)
        finally:
            io_queue.task_done()

def enqueue_turn(role, content):
    io_queue.put({"type":"turn","ts":int(time.time()),"role":role,"content":content})

def enqueue_identify(name):
    io_queue.put({"type":"identify","name":name})

# ---- static/learned memory files (same names)
STATIC_FILE  = os.getenv("STATIC_FILE", "data/static_responses.json")
LEARN_FILE   = os.getenv("LEARN_FILE",  "data/learned_phrases.json")
os.makedirs(os.path.dirname(STATIC_FILE), exist_ok=True)

STATIC_RESPONSES = load_json(STATIC_FILE)
LEARNED_PHRASES  = load_json(LEARN_FILE)

PROMOTE_N = int(os.getenv("PROMOTE_N", "2"))

def enqueue_learn_async(prompt_norm, reply):
    rec = LEARNED_PHRASES.get(prompt_norm)
    if rec is None:
        LEARNED_PHRASES[prompt_norm] = {"count": 1, "reply": reply}
    else:
        rec["count"] = rec.get("count", 0) + 1
        if not rec.get("reply"): rec["reply"] = reply
        LEARNED_PHRASES[prompt_norm] = rec
    if LEARNED_PHRASES[prompt_norm]["count"] >= PROMOTE_N and prompt_norm not in STATIC_RESPONSES:
        STATIC_RESPONSES[prompt_norm] = reply
    # persist async via IO
    io_queue.put({"type":"persist_learn","prompt_norm":prompt_norm,"reply":reply})
    try:
        safe_write_json(LEARN_FILE, LEARNED_PHRASES)
        if prompt_norm in STATIC_RESPONSES:
            safe_write_json(STATIC_FILE, STATIC_RESPONSES)
    except Exception:
        pass

# ------------- Event bus for API/SSE -------------
events = queue.Queue()

def publish_event(ev: dict):
    try: events.put_nowait(ev)
    except Exception: pass

# ---------------- Convenience helpers for call-scoped rotation ----------------
def new_conversation(key: str | None = None):
    """Rotate to a fresh Redis conversation list (per call)."""
    global conv
    conv = Conversation(key=key)

def close_session():
    """Signal IO worker to close the current session (sets ended_at, persists)."""
    try:
        io_queue.put({"type": "close_session"})
    except Exception:
        pass

def open_new_session():
    """Start a new session in-memory; next IO events will persist under a new file."""
    try:
        IO.session_id = uuid.uuid4().hex[:12]
        IO.session = {
            "session_id": IO.session_id,
            "started_at": utcnow_iso(),
            "ended_at": None,
            "turns": []
        }
        IO.session_path = None
        _write_session_file()
    except Exception:
        pass

__all__ = [
    "rds", "STOP", "set_stop", "events", "publish_event",
    "Conversation", "conv", "new_conversation",
    "enqueue_turn", "enqueue_identify",
    "STATIC_RESPONSES", "LEARNED_PHRASES", "enqueue_learn_async",
    "close_session", "open_new_session",
]
