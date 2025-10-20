# --- Ensure CUDA/cuDNN from pip wheels are discoverable on Windows ---
import os, sys, pathlib
if os.name == "nt":
    def _add(p: str):
        if os.path.isdir(p):
            os.add_dll_directory(p)
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
    base = pathlib.Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    _add(str(base / "cudnn_cu12" / "bin"))
    _add(str(base / "cublas_cu12" / "bin"))
    _add(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin")
# --- end Windows CUDA path bootstrap ---

import re, html, json, time, queue, threading, hashlib, shutil, subprocess, collections, signal, uuid
from datetime import datetime
import numpy as np
import sounddevice as sd
import webrtcvad
import redis
from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import fuzz
from faster_whisper import WhisperModel

# ==============================
# ENV & CONFIG
# ==============================
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

STATIC_FILE = os.getenv("STATIC_FILE", "static_responses.json")
LEARN_FILE  = os.getenv("LEARN_FILE",  "learned_phrases.json")
PROMOTE_N   = int(os.getenv("PROMOTE_N", "3"))

MODEL_SIZE = os.getenv("WHISPER_SIZE", "small")
DEVICE     = os.getenv("WHISPER_DEVICE", "cuda")  # "cuda" or "cpu"
LANGUAGE   = os.getenv("LANGUAGE", "en")

# ==== TTS / barge-in knobs ====
TTS_BARGE_IN_PROTECT_MS = int(os.getenv("TTS_BARGE_IN_PROTECT_MS", "400"))   # a bit longer to avoid false cuts
PIPER_CHUNK              = int(os.getenv("PIPER_CHUNK", "4096"))             # Piper stdout read size

# ==== VAD / Endpointing ====
SAMPLE_RATE     = 16000
FRAME_DURATION  = int(os.getenv("FRAME_DURATION_MS", "20"))  # 10/20/30
FRAME_SIZE      = int(SAMPLE_RATE * FRAME_DURATION / 1000)
VAD_MODE        = int(os.getenv("VAD_MODE", "2"))            # 0..3

# Turn-ending tuned for human pauses (VAD-1.1)
END_SILENCE_MS     = int(os.getenv("END_SILENCE_MS", "3500"))   # was 300
CONTINUE_WINDOW_MS = int(os.getenv("CONTINUE_WINDOW_MS", "800"))
PRE_ROLL_MS        = int(os.getenv("PRE_ROLL_MS", "120"))
POST_ROLL_MS       = int(os.getenv("POST_ROLL_MS", "120"))
MIN_TURN_MS        = int(os.getenv("MIN_TURN_MS", "350"))       # was 250

# Early-finalize disabled by default (no punctuation commits)
EARLY_FINALIZE_PUNCT = os.getenv("EARLY_FINALIZE_PUNCT", "0") not in ("0","false","False")
EARLY_SILENCE_MS     = int(os.getenv("EARLY_SILENCE_MS", "250"))

END_SILENCE_FRAMES   = max(1, END_SILENCE_MS     // FRAME_DURATION)
CONTINUE_WINDOW_FR   = max(1, CONTINUE_WINDOW_MS // FRAME_DURATION)
PRE_FRAMES           = max(1, PRE_ROLL_MS        // FRAME_DURATION)
POST_FRAMES          = max(1, POST_ROLL_MS       // FRAME_DURATION)
MIN_TURN_FRAMES      = max(1, MIN_TURN_MS        // FRAME_DURATION)
EARLY_SILENCE_FR     = max(1, EARLY_SILENCE_MS   // FRAME_DURATION)

# ==== STT False-positive controls ====
CALIBRATE_MS          = int(os.getenv("CALIBRATE_MS", "1200"))
ENERGY_FLOOR_MULT     = float(os.getenv("ENERGY_FLOOR_MULT", "2.5"))
ENERGY_MIN_RMS        = float(os.getenv("ENERGY_MIN_RMS", "0.006"))
START_TRIGGER_MS      = int(os.getenv("START_TRIGGER_MS", "240"))
MIN_VALID_WORDS       = int(os.getenv("MIN_VALID_WORDS", "2"))
MIN_VALID_CHARS       = int(os.getenv("MIN_VALID_CHARS", "6"))
START_TRIGGER_FRAMES  = max(1, START_TRIGGER_MS // FRAME_DURATION)

# ==== LLM enqueue/debounce + dedupe ====
LLM_DEBOUNCE_WINDOW_MS = int(os.getenv("LLM_DEBOUNCE_WINDOW_MS", "1200"))  # drop near-duplicates arriving too quickly
LLM_DEBOUNCE_SIM       = int(os.getenv("LLM_DEBOUNCE_SIM", "92"))          # fuzzy similarity for 'too similar'
LLM_RECENT_TTL_SEC     = float(os.getenv("LLM_RECENT_TTL_SEC", "3.0"))     # don't re-answer same text soon
STT_FINALIZE_COOLDOWN_MS = int(os.getenv("STT_FINALIZE_COOLDOWN_MS", "300"))

# ==== Turn coalescing & heartbeat ====
COALESCE_GRACE_MS   = int(os.getenv("COALESCE_GRACE_MS", "1200"))   # extend if user resumes
COALESCE_MAX_MS     = int(os.getenv("COALESCE_MAX_MS", "6000"))     # safety cap for very long coalesces
HEARTBEAT_IDLE_SEC  = int(os.getenv("HEARTBEAT_IDLE_SEC", "45"))    # nudge after prolonged silence
HEARTBEAT_TEXT      = os.getenv("HEARTBEAT_TEXT", "Still there? What should we do next?")

# ==== Barge-in robustness (echo-resistant) ====
BARGE_IN_TRIGGER_MS     = int(os.getenv("BARGE_IN_TRIGGER_MS", "260"))     # sustain time required before interrupt
BARGE_IN_MIN_RMS_MULT   = float(os.getenv("BARGE_IN_MIN_RMS_MULT", "3.0"))  # vs ambient baseline (nearfield > echo)
BARGE_IN_ECHO_MULT      = float(os.getenv("BARGE_IN_ECHO_MULT", "2.0"))     # vs tracked echo RMS while TTS is playing
ECHO_TRACK_DECAY        = float(os.getenv("ECHO_TRACK_DECAY", "0.9"))       # EMA decay for echo RMS
BARGE_IN_TRIGGER_FRAMES = max(1, BARGE_IN_TRIGGER_MS // FRAME_DURATION)

# ==== End-call policy (strict) ====
END_CALL_STRICT   = os.getenv("END_CALL_STRICT", "1") not in ("0","false","False")
EXACT_END_PHRASE  = os.getenv("EXACT_END_PHRASE", "end call now").strip().lower()
END_CALL_PHRASES = [p.strip().lower() for p in os.getenv(
    "END_CALL_PHRASES",
    "goodbye,bye,that will be all,that's all,hang up,thanks that's it,thank you that's it,end call,please disconnect,quit,exit,done"
).split(",") if p.strip()]

STOP = threading.Event()

# ==============================
# Coalescing state + heartbeat activity tracking
# ==============================
_coalesce_lock = threading.Lock()
_coalesce_timer = None
_coalesce_started_at = 0.0
_pending_user_turn = ""  # text being coalesced
_last_activity_ts = time.time()

def _now(): return time.time()

def _touch_activity():
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
        _enqueue_latest_user_text(text)

def _schedule_coalesce_fire(delay_ms: int):
    global _coalesce_timer
    _cancel_coalesce_timer()
    _coalesce_timer = threading.Timer(delay_ms/1000.0, _coalesce_fire)
    _coalesce_timer.daemon = True
    _coalesce_timer.start()

def commit_user_turn(new_text: str):
    """Append new_text into a pending coalesced turn and (re)schedule commit."""
    global _pending_user_turn, _coalesce_started_at
    if not new_text: return
    with _coalesce_lock:
        now = _now()
        if not _pending_user_turn:
            _pending_user_turn = new_text.strip()
            _coalesce_started_at = now
        else:
            # merge with minimal punctuation cleanup
            if not _pending_user_turn.endswith((".", "?", "!")) and not new_text.lstrip().startswith((",", "and", "but")):
                _pending_user_turn += " "
            _pending_user_turn += new_text.strip()
        elapsed = int((now - _coalesce_started_at) * 1000.0)
        remaining = max(0, COALESCE_MAX_MS - elapsed)
        _schedule_coalesce_fire(min(COALESCE_GRACE_MS, remaining))

# ==============================
# UTILS
# ==============================
def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in s) or "user"

def _safe_write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def _load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def normalize_text(text: str) -> str:
    return text.lower().strip().translate(str.maketrans("", "", ".,?!"))

def _word_count(s: str) -> int:
    return len([w for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", s)])

# ==============================
# IN-MEM STORES
# ==============================
STATIC_RESPONSES = _load_json(STATIC_FILE)
LEARNED_PHRASES  = _load_json(LEARN_FILE)
print(f"📚 Static: {len(STATIC_RESPONSES)} | Learned: {len(LEARNED_PHRASES)}")

# ==============================
# ASYNC IO WORKER (per-user + per-session files)
# ==============================
LOG_ROOT = os.getenv("LOG_DIR", "logs/users")
os.makedirs(LOG_ROOT, exist_ok=True)

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
            "started_at": datetime.utcnow().isoformat()+"Z",
            "ended_at": None,
            "turns": []
        }
        self.session_path = None

IO = IOState()

def _user_file_skeleton(name, key):
    return {
        "user": name,
        "user_key": key,
        "created_at": datetime.utcnow().isoformat()+"Z",
        "updated_at": None,
        "sessions": []
    }

def _write_session_file():
    if not (IO.user_dir and IO.session_id):
        return
    if IO.session_path is None:
        IO.session_path = os.path.join(IO.user_dir, f"{IO.session_id}.json")
    payload = {"user": IO.user_name, "user_key": IO.user_key, "session": IO.session}
    _safe_write_json(IO.session_path, payload)

def io_worker():
    while not STOP.is_set():
        try:
            ev = io_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            typ = ev.get("type")

            if typ == "identify":
                name = ev["name"].strip()
                if not IO.user_name:
                    IO.user_name = name
                    IO.user_key  = _slug(name)
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
                    IO.root["updated_at"] = datetime.utcnow().isoformat()+"Z"
                    _safe_write_json(IO.user_path, IO.root)
                    _write_session_file()

            elif typ == "turn":
                IO.session["turns"].append({"ts": ev["ts"], "role": ev["role"], "content": ev["content"]})
                if IO.user_path:
                    if not IO.root["sessions"] or IO.root["sessions"][-1]["session_id"] != IO.session["session_id"]:
                        IO.root["sessions"].append({"session_id": IO.session["session_id"],
                                                    "started_at": IO.session["started_at"],
                                                    "ended_at": None, "turns": []})
                    IO.root["sessions"][-1] = dict(IO.session)
                    IO.root["updated_at"] = datetime.utcnow().isoformat()+"Z"
                    _safe_write_json(IO.user_path, IO.root)
                    _write_session_file()

            elif typ == "persist_learn":
                pn = ev["prompt_norm"]; reply = ev["reply"]
                rec = LEARNED_PHRASES.get(pn)
                if rec is None:
                    LEARNED_PHRASES[pn] = {"count": 1, "reply": reply}
                else:
                    rec["count"] = rec.get("count", 0) + 1
                    if not rec.get("reply"): rec["reply"] = reply
                    LEARNED_PHRASES[pn] = rec
                if LEARNED_PHRASES[pn]["count"] >= PROMOTE_N and pn not in STATIC_RESPONSES:
                    STATIC_RESPONSES[pn] = reply
                    _safe_write_json(STATIC_FILE, STATIC_RESPONSES)
                _safe_write_json(LEARN_FILE, LEARNED_PHRASES)

            elif typ == "close_session":
                if IO.session["ended_at"] is None:
                    IO.session["ended_at"] = datetime.utcnow().isoformat()+"Z"
                if IO.user_path:
                    if IO.root["sessions"] and IO.root["sessions"][-1]["session_id"] == IO.session["session_id"]:
                        IO.root["sessions"][-1] = dict(IO.session)
                    else:
                        IO.root["sessions"].append(dict(IO.session))
                    IO.root["updated_at"] = datetime.utcnow().isoformat()+"Z"
                    _safe_write_json(IO.user_path, IO.root)
                    _write_session_file()
                else:
                    unknown = os.path.join(LOG_ROOT, f"unknown_{IO.session['session_id']}.json")
                    _safe_write_json(unknown, {
                        "user": None, "user_key": None,
                        "created_at": datetime.utcnow().isoformat()+"Z",
                        "updated_at": datetime.utcnow().isoformat()+"Z",
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
    io_queue.put({"type":"persist_learn","prompt_norm":prompt_norm,"reply":reply})

def close_session():
    io_queue.put({"type":"close_session"})

# ==============================
# TTS sanitizer (+ typography fixes)
# ==============================
EMOJI_RE = re.compile("[" +
    "\U0001F600-\U0001F64F" + "\U0001F300-\U0001F5FF" + "\U0001F680-\U0001F6FF" +
    "\U0001F700-\U0001F77F" + "\U0001F780-\U0001F7FF" + "\U0001F800-\U0001F8FF" +
    "\U0001F900-\U0001F9FF" + "\U0001FA00-\U0001FA6F" + "\U0001FA70-\U0001FAFF" +
    "\u2600-\u26FF" + "\u2700-\u27BF" + "]")
URL_RE = re.compile(r"https?://\S+")

TYPO_MAP = str.maketrans({
    "’": "'", "‘": "'", "‛": "'", "ʼ": "'", "ʹ": "'", "ˈ": "'", "ꞌ": "'", "＇": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "＂": '"',
    "—": "-", "–": "-", "‒": "-", "―": "-",
    "\u00A0": " ", "\u2009": " ", "\u200A": " ", "\u202F": " ",
    "\u200B": "", "\u200C": "", "\u200D": "", "\uFEFF": " ",
})
def _normalize_typography(s: str) -> str:
    s = s.translate(TYPO_MAP)
    s = re.sub(r'\s*"\s*', '"', s)
    s = re.sub(r"\s*'\s*", "'", s)
    s = re.sub(r"\b([A-Za-z]+)\s+'\s*([A-Za-z]+)\b", r"\1'\2", s)
    return s

TTS_EXPAND_CONTRACTIONS = os.getenv("TTS_EXPAND_CONTRACTIONS", "0") not in ("0","false","False")
_CONTRACTIONS = {
    "I'm": "I am", "I’m": "I am",
    "you're": "you are", "you’re": "you are",
    "we're": "we are", "we’re": "we are",
    "they're": "they are", "they’re": "they are",
    "it's": "it is", "it’s": "it is",
    "that's": "that is", "that’s": "that is",
    "there's": "there is", "there’s": "there is",
    "can't": "cannot", "can’t": "cannot",
    "won't": "will not", "won’t": "will not",
    "don't": "do not", "don’t": "do not",
    "didn't": "did not", "didn’t": "did not",
    "isn't": "is not", "isn’t": "is not",
    "aren't": "are not", "aren’t": "are not",
    "wasn't": "was not", "wasn’t": "was not",
    "weren't": "were not", "weren’t": "were not",
    "I've": "I have", "I’ve": "I have",
    "you've": "you have", "you’ve": "you have",
    "we've": "we have", "we’ve": "we have",
    "they've": "they have", "they’ve": "they have",
    "I'll": "I will", "I’ll": "I will",
    "you'll": "you will", "you’ll": "you will",
    "we'll": "we will", "we’ll": "we will",
    "they'll": "they will", "they’ll": "they will",
}
def _maybe_expand_contractions(s: str) -> str:
    if not TTS_EXPAND_CONTRACTIONS:
        return s
    for k, v in _CONTRACTIONS.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s, flags=re.IGNORECASE)
    return s

def _strip_emojis(s: str) -> str:
    s = EMOJI_RE.sub("", s)
    s = s.replace("\uFE0F", "")
    s = re.sub(r"[\U0001F3FB-\U0001F3FF]", "", s)
    return s

def _strip_markdown(s: str) -> str:
    s = re.sub(r"```.*?```", "", s, flags=re.S)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"[*_~]{1,3}([^*_~]+)[*_~]{1,3}", r"\1", s)
    return s

def _strip_emotes(s: str) -> str:
    return re.sub(r"(^|\s)([:;]-?[\)D\(PpOo/\\])", r"\1", s)

def to_speakable(text: str) -> str:
    s = html.unescape(text or "")
    s = _normalize_typography(s)
    s = _strip_markdown(s)
    s = URL_RE.sub("", s)
    s = _maybe_expand_contractions(s)
    s = _strip_emotes(s)
    s = _strip_emojis(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ==============================
# CLIENTS
# ==============================
client = OpenAI(base_url="https://api.deepseek.com/v1", api_key=DEEPSEEK_API_KEY) if DEEPSEEK_API_KEY else None
rds = redis.from_url(REDIS_URL, decode_responses=True)

SYSTEM_PROMPT = (
    "You are CloumenAI — a warm, concise phone agent. "
    "Speak in natural, human conversation. Use slight interjections ('hmm', 'oh') when they help. "
    "Keep replies short and conversational. Qualify interest in our web/app/AI/cloud services. "
    "Use PLAIN TEXT only: no emojis, emoticons, markdown, or code formatting. "
    "Use straight ASCII quotes only: ' and \". "
    "Remember prior turns in this conversation.\n"
    "\n"
    "CRITICAL RULES:\n"
    "- Never generate a concluding/terminal phrase (no 'goodbye', 'that's all', 'anything else?') unless the user explicitly commands the conversation to end.\n"
    "- Always end each reply with a brief, open-ended prompt that invites the user to continue (1 short clause, no list of options).\n"
    "- Adapt tone to emotion (frustration, confusion, excitement) with calm, concise empathy; avoid over-apologizing.\n"
)

# ==============================
# Conversation memory (Redis) with rekey
# ==============================
class Conversation:
    def __init__(self, key="session:unknown", max_turns=12, ttl=3600):
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
                rds.rpush(self.key, json.dumps(msg, ensure_ascii=False))
                rds.ltrim(self.key, -2*self.max_turns, -1)
                rds.expire(self.key, self.ttl)
        except Exception:
            pass
    def rekey(self, new_key):
        new_key = f"conv:{new_key}"
        try:
            with self._lock:
                items = rds.lrange(self.key, 0, -1)
                if items:
                    rds.delete(new_key)
                    rds.rpush(new_key, *items)
                rds.delete(self.key)
                self.key = new_key
        except Exception:
            self.key = new_key

conv = Conversation(key=f"unknown_{uuid.uuid4().hex[:6]}", max_turns=int(os.getenv("CONV_TURNS", "12")))

# ==============================
# Name detection (stricter; do not flip once set)
# ==============================
STOP_TOKENS = {"from","with","at","of","the","this","that","company","inc","ltd","llc","limited","systems","solutions","speaking","call","calling","team","department","services","group"}
PATTERNS = [
    re.compile(r"\b(?:my\s+name\s+is|this\s+is)\s+([a-z][a-z\s'-]{1,40})", re.I),
    re.compile(r"\b([a-z][a-z'-]{1,20})\s+speaking\b", re.I),
]
def _clean_name_tokens(tokens):
    out = []
    for t in tokens:
        t = re.sub(r"[^a-zA-Z\-']", "", t)
        if not t: continue
        low = t.lower()
        if low in STOP_TOKENS: break
        if low in ("not","no","dont","don't"): return ""
        out.append(t)
        if len(out) >= 3: break
    return " ".join(out)
def detect_name_from_text(text: str):
    if IO.user_name:
        return None
    t = text.strip()
    for p in PATTERNS:
        m = p.search(t)
        if not m: continue
        span = m.group(1).strip()
        tokens = span.split()
        name = _clean_name_tokens(tokens)
        if name and len(name) <= 40:
            return " ".join(w.capitalize() for w in name.split())
    return None
def set_user_identity_async(text: str):
    name = detect_name_from_text(text or "")
    if not name:
        return
    try: conv.rekey(_slug(name))
    except Exception: pass
    enqueue_identify(name)
    print(f"👤 Identified user as: {name}")

# ==============================
# Piper TTS (persistent engine + barge-in)
# ==============================
VOICES_BASE = os.getenv("PIPER_VOICES_BASE", r"F:\ai cold calling agent\voices")
FEMALE_MODEL  = os.getenv("PIPER_FEMALE_MODEL", os.path.join(VOICES_BASE, "female_en", "en_US-hfc_female-medium.onnx"))
FEMALE_CONFIG = os.getenv("PIPER_FEMALE_CONFIG", os.path.join(VOICES_BASE, "female_en", "en_US-hfc_female-medium.onnx.json"))
MALE_MODEL    = os.getenv("PIPER_MALE_MODEL",   os.path.join(VOICES_BASE, "male_en", "en_US-bryce-medium.onnx"))
MALE_CONFIG   = os.getenv("PIPER_MALE_CONFIG",  os.path.join(VOICES_BASE, "male_en", "en_US-bryce-medium.onnx.json"))

PIPER_VOICE_SELECT = os.getenv("PIPER_VOICE", "female").lower()
PIPER_EXE    = os.getenv("PIPER_EXE", "piper")
PIPER_USE_CUDA = os.getenv("PIPER_CUDA", "0") not in ("0","false","False")

LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "0.95"))
NOISE_SCALE  = float(os.getenv("PIPER_NOISE_SCALE",  "0.45"))
NOISE_W      = float(os.getenv("PIPER_NOISE_W",      "0.55"))

tts_playing = threading.Event()
_tts_started_at = 0.0  # for barge-in protection

def _select_voice():
    return (MALE_MODEL, MALE_CONFIG) if PIPER_VOICE_SELECT == "male" else (FEMALE_MODEL, FEMALE_CONFIG)

def _load_voice_sr(config_path: str) -> int:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg.get("sample_rate", 22050))
    except Exception:
        return 22050

def _derive_tts_params(text: str):
    ls = LENGTH_SCALE; ns = NOISE_SCALE; nw = NOISE_W
    t = text.strip()
    if "!" in t and len(t) < 140:
        ls = max(0.85, LENGTH_SCALE - 0.05)
        ns = min(0.70, NOISE_SCALE + 0.15)
    if t.lower().startswith(("sorry", "i’m sorry", "im sorry", "unfortunately")):
        ls = min(1.10, LENGTH_SCALE + 0.15)
        ns = max(0.30, NOISE_SCALE - 0.10)
    return ls, ns, nw

class PiperEngine:
    """Keep a single piper.exe alive; stream multiple utterances."""
    def __init__(self):
        self.proc = None
        self.stream = None
        self.reader = None
        self.stop_reader = threading.Event()
        self.sr = 22050
        self.args = None
        self._io_lock = threading.Lock()
        self._last_data_ts = 0.0

    def start(self):
        model, config = _select_voice()
        if not (shutil.which(PIPER_EXE) and os.path.exists(model) and os.path.exists(config)):
            print("🔊 Piper not available; audio disabled.")
            return False
        self.sr = _load_voice_sr(config)
        ls, ns, nw = _derive_tts_params("")
        self.args = [PIPER_EXE, "--model", model, "--config", config, "--output_raw",
                     "--length-scale", f"{ls}", "--noise-scale", f"{ns}", "--noise-w", f"{nw}"]
        if PIPER_USE_CUDA:
            self.args.append("--cuda")

        self.proc = subprocess.Popen(self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=0)
        self.stream = sd.RawOutputStream(samplerate=self.sr, channels=1, dtype="int16")
        self.stream.start()

        self.stop_reader.clear()
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()
        return True

    def _read_loop(self):
        try:
            while not self.stop_reader.is_set():
                if self.proc is None or self.proc.stdout is None:
                    time.sleep(0.005); continue
                data = self.proc.stdout.read(PIPER_CHUNK)
                now = time.time()
                if data:
                    self._last_data_ts = now
                    self.stream.write(data)
                    _touch_activity()
                    if not tts_playing.is_set():
                        tts_playing.set()
                else:
                    # longer idle gap prevents flicker
                    if tts_playing.is_set() and (now - self._last_data_ts) > 0.6:
                        tts_playing.clear()
                    time.sleep(0.002)
        except Exception:
            pass

    def say(self, text: str):
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
                _touch_activity()
            except Exception:
                self.restart()
                try:
                    self.proc.stdin.write((s + "\n").encode("utf-8"))
                    self.proc.stdin.flush()
                    _tts_started_at = time.time()
                    self._last_data_ts = _tts_started_at
                    tts_playing.set()
                    _touch_activity()
                except Exception:
                    pass

    def stop(self):
        tts_playing.clear()
        try: self.stop_reader.set()
        except Exception: pass
        try:
            if self.stream:
                try: self.stream.abort()
                except Exception: pass
                try: self.stream.close()
                except Exception: pass
        finally: self.stream = None
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
        except Exception: pass
        finally: self.proc = None

    def restart(self):
        self.stop()
        time.sleep(0.03)
        self.start()

_piper = PiperEngine()
_piper_started = _piper.start()

def tts_interrupt():
    if _piper_started:
        _piper.restart()
    _touch_activity()

def tts_shutdown():
    try: _piper.stop()
    except Exception: pass
    _touch_activity()

def speak(text: str, user_text_for_name_detect: str = None):
    if not text:
        return
    if _piper_started:
        _piper.say(text)
    if user_text_for_name_detect:
        threading.Thread(target=set_user_identity_async, args=(user_text_for_name_detect,), daemon=True).start()

# ==============================
# Energy gate (cuts phantom speech)
# ==============================
class EnergyGate:
    def __init__(self, frame_ms=FRAME_DURATION, calibrate_ms=CALIBRATE_MS,
                 floor_mult=ENERGY_FLOOR_MULT, min_rms=ENERGY_MIN_RMS):
        self.frame_ms = frame_ms
        self.target = max(1, calibrate_ms // frame_ms)
        self.buf = collections.deque(maxlen=self.target)
        self.baseline = None
        self.floor_mult = floor_mult
        self.min_rms = min_rms
        self.frames_seen = 0

    @staticmethod
    def _rms(frame_bytes):
        x = np.frombuffer(frame_bytes, np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def loud_enough(self, frame_bytes):
        rms = self._rms(frame_bytes)
        if self.baseline is None:
            self.buf.append(rms); self.frames_seen += 1
            if self.frames_seen >= self.target:
                self.baseline = float(np.median(np.array(self.buf)))
            return False  # no triggers during calibration
        thr = max(self.baseline * self.floor_mult, self.min_rms)
        return rms >= thr

# ==============================
# LLM path (latest-only, deduped)
# ==============================
llm_queue = queue.Queue()
LLM_LOCK = threading.Lock()

_last_enq_text = ""
_last_enq_ts = 0.0
_recent_texts = collections.deque()  # (ts, text)

def _similar(a: str, b: str) -> int:
    try: return fuzz.ratio(a, b)
    except Exception: return 0

def _accept_against_recent(text: str) -> bool:
    """Reject if text ~equals something we answered very recently."""
    now = time.time()
    while _recent_texts and (now - _recent_texts[0][0]) > LLM_RECENT_TTL_SEC:
        _recent_texts.popleft()
    for _, t in _recent_texts:
        if _similar(text.lower(), t.lower()) >= 96:
            return False
    _recent_texts.append((now, text))
    return True

def _enqueue_latest_user_text(text: str):
    """Coalesce: clear pending, debounce near-duplicates, enforce recent TTL, queue the latest once."""
    global _last_enq_text, _last_enq_ts
    s = (text or "").strip()
    if not s:
        return
    now = time.time()
    if _last_enq_text and (now - _last_enq_ts) * 1000.0 < LLM_DEBOUNCE_WINDOW_MS:
        if _similar(s.lower(), _last_enq_text.lower()) >= LLM_DEBOUNCE_SIM:
            return  # too similar, too soon
    if not _accept_against_recent(s):
        return  # already answered something effectively identical very recently
    _last_enq_text = s
    _last_enq_ts = now
    try:
        while True:
            llm_queue.get_nowait()
            llm_queue.task_done()
    except queue.Empty:
        pass
    llm_queue.put(s)

def deepseek_reply(prompt: str) -> str:
    norm = normalize_text(prompt)
    conv.append("user", prompt)
    enqueue_turn("user", prompt)

    plow = prompt.lower()
    if END_CALL_STRICT:
        if EXACT_END_PHRASE and EXACT_END_PHRASE in plow:
            reply = "Understood. Ending the call now."
            conv.append("assistant", reply); enqueue_turn("assistant", reply)
            speak(reply, user_text_for_name_detect=prompt)
            STOP.set()
            return reply
        if any(p in plow for p in END_CALL_PHRASES):
            reply = f"If you want me to end the call, please say exactly: {EXACT_END_PHRASE}. Otherwise, how can I help?"
            conv.append("assistant", reply); enqueue_turn("assistant", reply)
            return reply
    else:
        if any(p in plow for p in END_CALL_PHRASES):
            reply = "Alright, I’ll end the call here. Have a great day!"
            conv.append("assistant", reply); enqueue_turn("assistant", reply)
            speak(reply, user_text_for_name_detect=prompt)
            STOP.set()
            return reply

    # in-memory hits
    if norm in STATIC_RESPONSES:
        reply = to_speakable(STATIC_RESPONSES[norm])
        print("💾 STATIC HIT")
        conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply)
        return reply
    if norm in LEARNED_PHRASES:
        reply = to_speakable(LEARNED_PHRASES[norm]["reply"])
        print("🧠 LEARNED HIT")
        conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply)
        return reply
    if STATIC_RESPONSES:
        m = max(((k, fuzz.token_sort_ratio(norm, k)) for k in STATIC_RESPONSES.keys()), key=lambda x: x[1], default=None)
        if m and m[1] >= 85:
            reply = to_speakable(STATIC_RESPONSES[m[0]])
            print("💾 STATIC (FUZZY) HIT")
            conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply)
            return reply
    if LEARNED_PHRASES:
        m2 = max(((k, fuzz.token_sort_ratio(norm, k)) for k in LEARNED_PHRASES.keys()), key=lambda x: x[1], default=None)
        if m2 and m2[1] >= 88:
            reply = to_speakable(LEARNED_PHRASES[m2[0]]["reply"])
            print("🧠 LEARNED (FUZZY) HIT")
            conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply)
            return reply

    # Redis cache
    cache_key = hashlib.sha256(norm.encode()).hexdigest()
    try: cached = rds.get(cache_key)
    except Exception: cached = None
    if cached:
        print("🔥 Cache HIT — serving cached LLM reply (cache hot).")
        reply = to_speakable(cached)
        conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply)
        return reply

    # LLM
    if not client:
        reply = "Sorry, my brain isn’t online right now. Please try again later."
        print("❌ No LLM client — offline fallback.")
        conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply)
        return reply

    try:
        print("🌐 LLM API HIT — querying model …")
        messages = [{"role":"system","content":SYSTEM_PROMPT}, *conv.load()]
        resp = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.6, top_p=0.9, presence_penalty=0.2, frequency_penalty=0.2,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "160")),
            messages=messages,
            timeout=20,
        )
        raw = (resp.choices[0].message.content or "").strip() or "Sorry, I didn’t catch that — could you repeat?"
        reply = to_speakable(raw)
        try:
            rds.setex(cache_key, 86400, reply)
        except Exception:
            pass
        conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply)
        return reply
    except Exception as e:
        print(f"⚠️ DeepSeek error: {e}")
        reply = "I’m having a connection issue right now. Can we try again in a moment."
        conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply)
        return reply

def llm_worker():
    while not STOP.is_set():
        try:
            text = llm_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if text is None:
            llm_queue.task_done(); continue
        try:
            # Ensure single reply at a time (prevents double-speaks)
            with LLM_LOCK:
                reply = deepseek_reply(text)
                print(f"🤖 CloumenAI: {reply}")
                # Speak after LLM returns (one speak per turn)
                speak(reply, user_text_for_name_detect=text)
        except Exception as e:
            print("⚠️ llm_worker error:", e)
        finally:
            llm_queue.task_done()

# ==============================
# Filler / validity filters
# ==============================
_FILLER_TOKENS = {
    "um","uh","hmm","mm","erm","uhm","ah","oh","yeah","yep","yup",
    "okay","ok","okey","uh-huh","huh","hmm-mm","mmm","hmm-m"
}

def _looks_like_filler(s: str) -> bool:
    if not s: return True
    t = re.sub(r"[^a-zA-Z\s'-]", " ", s.lower()).strip()
    if not t: return True
    toks = [w for w in t.split() if w]
    if not toks: return True
    if len(" ".join(toks)) <= 8 and all(w in _FILLER_TOKENS for w in toks):
        return True
    if len(toks) == 1 and toks[0] in _FILLER_TOKENS:
        return True
    return False

def _text_seems_valid(s: str) -> bool:
    if not s: return False
    if _looks_like_filler(s): return False
    if len(s.strip()) < MIN_VALID_CHARS: return False
    if _word_count(s) < MIN_VALID_WORDS: return False
    return True

# ==============================
# STT + VAD + anti-phantom logic
# ==============================
DECODE = dict(
    language=LANGUAGE,
    beam_size=1,
    without_timestamps=True,
    vad_filter=False,
    condition_on_previous_text=False,
    temperature=0.0,
    log_prob_threshold=-2.0,
    compression_ratio_threshold=2.8,
)

audio_queue = queue.Queue()
vad = webrtcvad.Vad(VAD_MODE)

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    pcm16 = (indata * 32768).astype(np.int16).tobytes()
    audio_queue.put(pcm16)

def make_model():
    print("🔄 Loading Faster-Whisper model...")
    m = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="float16", device_index=0)
    try:
        _ = m.transcribe(np.zeros(3200, dtype=np.float32), language=LANGUAGE, beam_size=1)
    except Exception as e:
        print("⚠️ warmup ignored:", e)
    print("✅ Whisper loaded on", DEVICE, "(float16)!")
    return m

def stt_worker(model: WhisperModel):
    print("🎧 Ready! Start speaking… (Ctrl+C to stop)")
    pre_ring  = collections.deque(maxlen=PRE_FRAMES)
    post_ring = collections.deque(maxlen=POST_FRAMES)
    triggered = False; silence = 0; speech = 0; hold_left = 0
    utter = bytearray(); last_partial = ""; frames_since_partial = 0
    last_finalized_at = 0.0
    cooldown_until = 0.0

    gate = EnergyGate()

    # Barge-in echo resistance state
    barge_run = 0            # counts consecutive frames that qualify for barge-in
    echo_rms  = 0.0          # rolling estimate of echo loudness while TTS is playing

    def run_partial_for_punct():
        nonlocal last_partial
        if len(utter) < 2 * FRAME_SIZE * MIN_TURN_FRAMES:
            return ""
        audio_np = np.frombuffer(utter, np.int16).astype(np.float32) / 32768.0
        try:
            segs, _ = model.transcribe(audio_np, **DECODE)
            txt = " ".join(s.text.strip() for s in segs).strip()
            if txt: last_partial = txt
            return txt
        except Exception:
            return ""

    while not STOP.is_set():
        try:
            frame = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            now = time.time()

            # Respect finalize cooldown to avoid echo re-triggers
            if now < cooldown_until:
                pre_ring.clear()
                triggered = False
                continue

            # Compute frame RMS once
            frame_rms = EnergyGate._rms(frame)

            vad_speech = vad.is_speech(frame, SAMPLE_RATE)
            energy_ok  = gate.loud_enough(frame)
            is_speech  = vad_speech and energy_ok

            # Echo-aware barge-in:
            if tts_playing.is_set():
                # Track echo while TTS playing (EMA)
                echo_rms = (ECHO_TRACK_DECAY * echo_rms) + ((1.0 - ECHO_TRACK_DECAY) * frame_rms)

                loud_vs_ambient = frame_rms >= (gate.baseline or 0.0) * BARGE_IN_MIN_RMS_MULT
                loud_vs_echo    = frame_rms >= max(1e-6, echo_rms) * BARGE_IN_ECHO_MULT

                if is_speech and loud_vs_ambient and loud_vs_echo:
                    barge_run += 1
                else:
                    barge_run = 0

                if barge_run >= BARGE_IN_TRIGGER_FRAMES and ((time.time() - _tts_started_at) * 1000.0 >= TTS_BARGE_IN_PROTECT_MS):
                    tts_interrupt()
                    barge_run = 0

            # activity on speech frames
            if is_speech:
                _touch_activity()

            if not triggered:
                pre_ring.append(frame)
                if is_speech:
                    speech += 1
                    if speech >= START_TRIGGER_FRAMES:
                        triggered = True
                        utter.extend(b"".join(pre_ring)); pre_ring.clear()
                        utter.extend(frame)
                        silence = 0; post_ring.clear()
                        frames_since_partial = 0; last_partial = ""
                else:
                    if speech > 0: speech = 0
                continue

            if is_speech:
                if hold_left > 0: hold_left = 0
                silence = 0; post_ring.clear()
                utter.extend(frame)
            else:
                silence += 1
                post_ring.append(frame)
                utter.extend(frame)

                # Early finalize (punctuation) — disabled by default unless env enables it
                if EARLY_FINALIZE_PUNCT and silence >= EARLY_SILENCE_FR:
                    frames_since_partial += 1
                    if frames_since_partial >= max(1, (100 // FRAME_DURATION)):
                        frames_since_partial = 0
                        txt = run_partial_for_punct()
                        if txt and txt[-1:] in ".?!":
                            print("🛑 Early end (punct). Transcribing…")
                            audio_np = np.frombuffer(utter, np.int16).astype(np.float32) / 32768.0
                            try:
                                segs, _ = model.transcribe(audio_np, **DECODE)
                                full_text = " ".join(s.text.strip() for s in segs).strip()
                            except Exception as e:
                                print(f"⚠️ ASR error: {e}"); full_text = ""
                            if _text_seems_valid(full_text):
                                if (now - last_finalized_at) * 1000.0 > 150:
                                    print(f"🗣️ User: {full_text}")
                                    commit_user_turn(full_text)   # coalescer, not direct LLM enqueue
                                    _touch_activity()
                                    last_finalized_at = now
                                    cooldown_until = now + (STT_FINALIZE_COOLDOWN_MS / 1000.0)
                            triggered = False; silence = 0; speech = 0
                            pre_ring.clear(); post_ring.clear(); utter.clear()
                            last_partial = ""; frames_since_partial = 0
                            continue

                # Normal finalize after pause (VAD-1.1/1.2)
                if silence >= END_SILENCE_FRAMES:
                    if hold_left == 0:
                        utter.extend(b"".join(post_ring)); post_ring.clear()
                        hold_left = CONTINUE_WINDOW_FR
                    else:
                        hold_left -= 1
                    if hold_left == 0:
                        total_frames = len(utter) // (2 * FRAME_SIZE)
                        if total_frames >= MIN_TURN_FRAMES:
                            print("🛑 Speech ended. Transcribing…")
                            audio_np = np.frombuffer(utter, np.int16).astype(np.float32) / 32768.0
                            try:
                                segs, _ = model.transcribe(audio_np, **DECODE)
                                full_text = " ".join(s.text.strip() for s in segs).strip()
                            except Exception as e:
                                print(f"⚠️ ASR error: {e}"); full_text = ""
                            if _text_seems_valid(full_text):
                                if (now - last_finalized_at) * 1000.0 > 150:
                                    print(f"🗣️ User: {full_text}")
                                    commit_user_turn(full_text)   # <-- coalescer
                                    _touch_activity()
                                    last_finalized_at = now
                                    cooldown_until = now + (STT_FINALIZE_COOLDOWN_MS / 1000.0)
                        triggered = False; silence = 0; speech = 0
                        pre_ring.clear(); post_ring.clear(); utter.clear()
                        last_partial = ""; frames_since_partial = 0
        except Exception as e:
            print("⚠️ stt_worker loop error:", e)

# ==============================
# Heartbeat nudge (PERS-3.2)
# ==============================
def heartbeat_worker():
    # gentle check-in if prolonged idle, only when not speaking and no pending STT trigger
    while not STOP.is_set():
        try:
            idle = _now() - _last_activity_ts
            if idle >= HEARTBEAT_IDLE_SEC and not tts_playing.is_set():
                _touch_activity()
                speak(HEARTBEAT_TEXT)
            time.sleep(1.0)
        except Exception:
            time.sleep(1.0)

# ==============================
# Exit & signals
# ==============================
def _cleanup_and_close():
    try: tts_shutdown()
    except Exception: pass
    try:
        close_session()
        # drain IO queue
        while not io_queue.empty():
            try: io_queue.get_nowait(); io_queue.task_done()
            except Exception: break
    except Exception: pass

def _handle_signal(signum, frame):
    STOP.set()

for s in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
    try: signal.signal(s, _handle_signal)
    except Exception: pass

# ==============================
# MAIN
# ==============================
def run_audio_loop():
    while not STOP.is_set():
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                blocksize=FRAME_SIZE, callback=audio_callback):
                while not STOP.is_set():
                    time.sleep(0.1)
        except Exception as e:
            print("⚠️ Input stream error, retrying in 1s:", e)
            time.sleep(1.0)

def main():
    t_io   = threading.Thread(target=io_worker, daemon=True)
    t_stt  = threading.Thread(target=stt_worker, args=(make_model(),), daemon=True)
    t_llm  = threading.Thread(target=llm_worker, daemon=True)
    t_hb   = threading.Thread(target=heartbeat_worker, daemon=True)

    t_io.start(); t_stt.start(); t_llm.start(); t_hb.start()

    try:
        run_audio_loop()
    finally:
        STOP.set()
        _cleanup_and_close()
        for t in (t_stt, t_llm, t_io, t_hb):
            try: t.join(timeout=2)
            except Exception: pass
        if IO.user_name:
            print(f"📝 Master log: {IO.user_path}")
            if IO.session_path:
                print(f"📝 Session log: {IO.session_path}")
        else:
            print("📝 Unknown session(s) saved under logs/users/unknown_*.json")

if __name__ == "__main__":
    main()