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
from typing import Optional
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
PROMOTE_N   = int(os.getenv("PROMOTE_N", "2"))

MODEL_SIZE = os.getenv("WHISPER_SIZE", "small")
DEVICE     = os.getenv("WHISPER_DEVICE", "cuda")  # "cuda" or "cpu"
LANGUAGE   = os.getenv("LANGUAGE", "en")

# ==== TTS / barge-in knobs ====
TTS_BARGE_IN_PROTECT_MS = int(os.getenv("TTS_BARGE_IN_PROTECT_MS", "700"))
PIPER_CHUNK              = int(os.getenv("PIPER_CHUNK", "4096"))

# ==== VAD / Endpointing ====
SAMPLE_RATE     = 16000
FRAME_DURATION  = int(os.getenv("FRAME_DURATION_MS", "20"))  # 10/20/30
FRAME_SIZE      = int(SAMPLE_RATE * FRAME_DURATION / 1000)
VAD_MODE        = int(os.getenv("VAD_MODE", "2"))

END_SILENCE_MS     = int(os.getenv("END_SILENCE_MS", "500"))
CONTINUE_WINDOW_MS = int(os.getenv("CONTINUE_WINDOW_MS", "500"))
PRE_ROLL_MS        = int(os.getenv("PRE_ROLL_MS", "100"))
POST_ROLL_MS       = int(os.getenv("POST_ROLL_MS", "100"))
MIN_TURN_MS        = int(os.getenv("MIN_TURN_MS", "140"))

EARLY_FINALIZE_PUNCT = os.getenv("EARLY_FINALIZE_PUNCT", "0") not in ("0","false","False")
EARLY_SILENCE_MS     = int(os.getenv("EARLY_SILENCE_MS", "220"))

END_SILENCE_FRAMES   = max(1, END_SILENCE_MS     // FRAME_DURATION)
CONTINUE_WINDOW_FR   = max(1, CONTINUE_WINDOW_MS // FRAME_DURATION)
PRE_FRAMES           = max(1, PRE_ROLL_MS        // FRAME_DURATION)
POST_FRAMES          = max(1, POST_ROLL_MS       // FRAME_DURATION)
MIN_TURN_FRAMES      = max(1, MIN_TURN_MS        // FRAME_DURATION)
EARLY_SILENCE_FR     = max(1, EARLY_SILENCE_MS   // FRAME_DURATION)

# ==== STT False-positive controls ====
CALIBRATE_MS          = int(os.getenv("CALIBRATE_MS", "1000"))
ENERGY_FLOOR_MULT     = float(os.getenv("ENERGY_FLOOR_MULT", "2.2"))
ENERGY_MIN_RMS        = float(os.getenv("ENERGY_MIN_RMS", "0.0055"))
START_TRIGGER_MS      = int(os.getenv("START_TRIGGER_MS", "180"))
MIN_VALID_WORDS       = int(os.getenv("MIN_VALID_WORDS", "2"))
MIN_VALID_CHARS       = int(os.getenv("MIN_VALID_CHARS", "6"))
START_TRIGGER_FRAMES  = max(1, START_TRIGGER_MS // FRAME_DURATION)

# ==== LLM enqueue/debounce + dedupe ====
LLM_DEBOUNCE_WINDOW_MS = int(os.getenv("LLM_DEBOUNCE_WINDOW_MS", "1000"))
LLM_DEBOUNCE_SIM       = int(os.getenv("LLM_DEBOUNCE_SIM", "92"))
LLM_RECENT_TTL_SEC     = float(os.getenv("LLM_RECENT_TTL_SEC", "3.0"))
STT_FINALIZE_COOLDOWN_MS = int(os.getenv("STT_FINALIZE_COOLDOWN_MS", "180"))

# ==== Turn coalescing & heartbeat ====
COALESCE_GRACE_MS   = int(os.getenv("COALESCE_GRACE_MS", "500"))
COALESCE_MAX_MS     = int(os.getenv("COALESCE_MAX_MS", "1500"))
HEARTBEAT_IDLE_SEC  = int(os.getenv("HEARTBEAT_IDLE_SEC", "60"))
HEARTBEAT_TEXT      = os.getenv("HEARTBEAT_TEXT", "Still there? What should we do next?")

# ==== Barge-in robustness (echo-aware) ====
BARGE_IN_TRIGGER_MS     = int(os.getenv("BARGE_IN_TRIGGER_MS", "260"))
BARGE_IN_MIN_RMS_MULT   = float(os.getenv("BARGE_IN_MIN_RMS_MULT", "2.2"))
BARGE_IN_ECHO_MULT      = float(os.getenv("BARGE_IN_ECHO_MULT", "2.5"))
ECHO_TRACK_DECAY        = float(os.getenv("ECHO_TRACK_DECAY", "0.93"))
BARGE_IN_TRIGGER_FRAMES = max(1, BARGE_IN_TRIGGER_MS // FRAME_DURATION)

# ==== End-call policy (strict) ====
END_CALL_STRICT   = os.getenv("END_CALL_STRICT", "1") not in ("0","false","False")
EXACT_END_PHRASE  = os.getenv("EXACT_END_PHRASE", "end call now").strip().lower()
END_CALL_PHRASES = [p.strip().lower() for p in os.getenv(
    "END_CALL_PHRASES",
    "goodbye,bye,that will be all,that's all,hang up,thanks that's it,thank you that's it,end call,please disconnect,quit,exit,done"
).split(",") if p.strip()]

STOP = threading.Event()
STOP_REASON = ""

def set_stop(reason: str):
    global STOP_REASON
    STOP_REASON = reason or "unspecified"
    STOP.set()

# ==============================
# Coalescing state + heartbeat
# ==============================
_coalesce_lock = threading.Lock()
_coalesce_timer = None
_coalesce_started_at = 0.0
_pending_user_turn = ""
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
# ASYNC IO WORKER
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

TTS_EXPAND_CONTRACTIONS = os.getenv("TTS_EXPAND_CONTRACTIONS", "1") not in ("0","false","False")
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

# Collapse letter-by-letter artifacts
def _fix_stream_artifacts(s: str) -> str:
    patterns = [
        (r'(?i)\b"h"\s*"m"\s*"m"\b', 'hmm'),
        (r'(?i)\bh\s*m\s*m\b', 'hmm'),
        (r'(?i)\b"u"\s*"h"\s*"h"\b', 'uhh'),
        (r'(?i)\bu\s*h\s*h\b', 'uhh'),
        (r'(?i)\b"m"\s*"m"\s*"m"\b', 'mmm'),
        (r'(?i)\bm\s*m\s*m\b', 'mmm'),
        (r'(?i)\b"o"\s*"k"\b', 'ok'),
        (r'(?i)\bo\s*k\b', 'ok'),
    ]
    for pat, rep in patterns:
        s = re.sub(pat, rep, s)
    def _collapse_seq(m):
        letters = re.findall(r"[A-Za-z]", m.group(0))
        return "".join(letters)
    s = re.sub(r"\b(?:[A-Za-z]\s+){1,3}[A-Za-z]\b", _collapse_seq, s)
    return s

def to_speakable(text: str) -> str:
    s = html.unescape(text or "")
    s = _normalize_typography(s)
    s = _strip_markdown(s)
    s = URL_RE.sub("", s)
    s = _maybe_expand_contractions(s)
    s = _strip_emotes(s)
    s = _strip_emojis(s)
    s = _fix_stream_artifacts(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- Punctuation tidy ---
_PUNCT_FIX = re.compile(r"\s+([,.!?])")
def tidy_punctuation(s: str) -> str:
    s = _PUNCT_FIX.sub(r"\1", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if s and s[-1] not in ".?!":
        s += "."
    return s

# --- Short-sentence shaper (no comma/hyphen splits) ---
TTS_SHAPER_MODE = os.getenv("TTS_SHAPER_MODE","light").lower()  # off|light|strict
SENTENCE_MAX_WORDS   = int(os.getenv("SENTENCE_MAX_WORDS", "18"))
MAX_SENTENCES_OUTPUT = int(os.getenv("MAX_SENTENCES_OUTPUT", "2"))

_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")

def _split_into_sentences(s: str):
    # strict: treat some punctuation as stronger breaks; light/off: don't split on commas/hyphens
    if TTS_SHAPER_MODE == "strict":
        s = s.replace("…", ".").replace(";", ".")
    return [p.strip() for p in _SENT_SPLIT.split(s) if p.strip()]

def _cap_words(sent: str):
    if TTS_SHAPER_MODE == "off":
        return sent
    words = sent.split()
    if len(words) <= SENTENCE_MAX_WORDS:
        return sent
    mid = len(words)//2
    for i in range(max(1, mid-3), min(len(words)-1, mid+4)):
        if words[i].lower() in {"and","but"}:
            left = " ".join(words[:i])
            right = " ".join(words[i+1:])
            return f"{left}. {right}"
    left = " ".join(words[:SENTENCE_MAX_WORDS])
    right = " ".join(words[SENTENCE_MAX_WORDS:])
    return f"{left}. {right}"

def _prune_to_two(sentences):
    if TTS_SHAPER_MODE == "off":
        return sentences
    if len(sentences) <= MAX_SENTENCES_OUTPUT:
        return sentences
    q = None
    for i in range(len(sentences)-1, 0, -1):
        if sentences[i].endswith("?"):
            q = i; break
    return [sentences[0], sentences[q]] if q and q != 0 else sentences[:MAX_SENTENCES_OUTPUT]

def shorten_for_tts(text: str) -> str:
    sents = _split_into_sentences(text)
    if TTS_SHAPER_MODE == "off":
        return " ".join(sents).strip()
    sents2 = []
    for s in sents:
        s2 = _cap_words(s)
        sents2.extend(_split_into_sentences(s2))
    sents2 = _prune_to_two(sents2)
    return " ".join(sents2).strip()

def shape_for_tts(raw: str) -> str:
    s = tidy_punctuation(raw)
    s = shorten_for_tts(s)
    s = tidy_punctuation(s)
    return to_speakable(s)

# ==============================
# CLIENTS
# ==============================
client = OpenAI(base_url="https://api.deepseek.com/v1", api_key=DEEPSEEK_API_KEY) if DEEPSEEK_API_KEY else None
rds = redis.from_url(REDIS_URL, decode_responses=True)

SYSTEM_PROMPT = (
    "You are CloumenAI — the outbound calling agent for Cloumen (cloumen.com). "
    "Speak warmly, simply, and very briefly. Strictly 1–2 sentences per turn. "
    "Keep each sentence short (8–16 words). Avoid hyphens. Prefer two clear sentences over one long one. "
    "Use plain punctuation. Never spell words letter-by-letter. "
    "ALWAYS end with a short open question to keep the conversation going.\n"
    "\n"
    "Company services (reference only if relevant):\n"
    "- AI Solutions & Integration (automation, chatbots, predictive analytics)\n"
    "- Data & Analytics (dashboards, KPIs, decision support)\n"
    "- Web & App Development (modern, fast, mobile/web)\n"
    "- Cloud Solutions (migration, modernization, reliability, cost)\n"
    "- Digital Marketing (SEO/social/content)\n"
    "- Technology Consulting (roadmaps, audits, strategy)\n"
    "\n"
    "Primary goal: qualify interest and book a short meeting, not to explain everything.\n"
    "Qualification: decision-maker, current stack, pains (manual work, scheduling, leads), timeline, willingness to meet.\n"
    "If they ask for a price or quote: NEVER give a number. Reply exactly: "
    "'We tailor pricing to scope. If we come back with the most competitive price and a value-for-money outcome, would you be open to exploring it?'\n"
    "\n"
    "Objections playbook:\n"
    "- Not interested → Acknowledge, one quick value angle, then offer a 10–15 min slot.\n"
    "- Busy/Email me → Ask for best email and a time to circle back; confirm timezone.\n"
    "- Already have dev/agency → Probe gaps (speed, quality, AI automation, costs), then offer a brief comparison call.\n"
    "\n"
    "Tone rules: Plain text only. ASCII quotes only. No emojis/markdown. Empathetic but concise. "
    "Never end the call unless the user says the exact phrase: 'end call now'.\n"
)

# ==============================
# Conversation memory (Redis)
# ==============================
class Conversation:
    def __init__(self, key="session:unknown", max_turns=int(os.getenv("CONV_TURNS","8")), ttl=3600):
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

conv = Conversation(key=f"unknown_{uuid.uuid4().hex[:6]}")

# ==============================
# Name detection (no rekey)
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
# Energy gate
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
            return False
        thr = max(self.baseline * self.floor_mult, self.min_rms)
        return rms >= thr

# ==============================
# LLM path (latest-only, deduped) + streaming
# ==============================
llm_queue = queue.Queue()
LLM_LOCK = threading.Lock()

_last_enq_text = ""
_last_enq_ts = 0.0
_recent_texts = collections.deque()

def _similar(a: str, b: str) -> int:
    try: return fuzz.ratio(a, b)
    except Exception: return 0

def _accept_against_recent(text: str) -> bool:
    now = time.time()
    while _recent_texts and (now - _recent_texts[0][0]) > LLM_RECENT_TTL_SEC:
        _recent_texts.popleft()
    for _, t in _recent_texts:
        if _similar(text.lower(), t.lower()) >= 96:
            return False
    _recent_texts.append((now, text))
    return True

def _enqueue_latest_user_text(text: str):
    global _last_enq_text, _last_enq_ts
    s = (text or "").strip()
    if not s: return
    now = time.time()
    if _last_enq_text and (now - _last_enq_ts) * 1000.0 < LLM_DEBOUNCE_WINDOW_MS:
        if _similar(s.lower(), _last_enq_text.lower()) >= LLM_DEBOUNCE_SIM:
            return
    if not _accept_against_recent(s):
        return
    _last_enq_text = s
    _last_enq_ts = now
    try:
        while True:
            llm_queue.get_nowait()
            llm_queue.task_done()
    except queue.Empty:
        pass
    llm_queue.put(s)

# === Guards (LLM-free) ===
PRICE_GUARD_COOLDOWN_SEC = float(os.getenv("PRICE_GUARD_COOLDOWN_SEC", "12"))
_LAST_PRICE_GUARD_TS = 0.0
PRICE_PATTERNS = [
    r"\bprice(s)?\b", r"\bpricing\b", r"\bcost(s)?\b", r"\bcharge(d|s)?\b",
    r"\brate(s)?\b", r"\bbudget(s)?\b", r"\bquote(s|d|ation)?\b",
    r"\bestimate(s|d)?\b", r"\bhow\s+much\b", r"\bfee(s)?\b",
    r"\bper\s+hour\b", r"\bper\s+project\b",
]
_PRICE_REGEXES = [re.compile(p, re.I) for p in PRICE_PATTERNS]

def price_guard(user_text: str) -> Optional[str]:
    global _LAST_PRICE_GUARD_TS
    t = (user_text or "").lower()
    now = time.time()
    if (now - _LAST_PRICE_GUARD_TS) < PRICE_GUARD_COOLDOWN_SEC:
        return None
    if not any(rx.search(t) for rx in _PRICE_REGEXES):
        return None
    reply = ("We tailor pricing to scope. "
             "If we come back with the most competitive price and a value-for-money outcome, "
             "would you be open to exploring it?")
    conv.append("assistant", reply); enqueue_turn("assistant", reply)
    enqueue_learn_async("pricing_guard", reply)
    speak(reply, user_text_for_name_detect=user_text)
    _LAST_PRICE_GUARD_TS = now
    return reply

LOCATION_KEYWORDS = {
    "where are you calling", "where are you based", "location", "from where", "which country", "which city"
}
def location_guard(user_text: str) -> Optional[str]:
    t = (user_text or "").lower()
    if any(k in t for k in LOCATION_KEYWORDS):
        reply = ("We operate globally with engineering across timezones. "
                 "Would mornings or afternoons suit you better for a quick intro?")
        conv.append("assistant", reply); enqueue_turn("assistant", reply)
        speak(reply, user_text_for_name_detect=user_text)
        return reply
    return None

BUSINESS_PITCHES = {
    "ecommerce": "For e-commerce, we add AI chat for conversion and automate support. We also build dashboards for ROAS and inventory. Should we explore a quick win?",
    "restaurant": "For restaurants, we automate reservations and feedback, and improve local search. Would that help your bookings?",
    "clinic": "For clinics, we automate intake and reminders, plus clear reporting for patient flow. Is reducing no-shows a priority?",
    "salon": "For salons, we streamline booking and follow-ups, and boost local visibility. Want to try a quick tactic to lift repeat visits?",
    "real estate": "For real estate, we qualify leads with AI and automate follow-ups. Are you focused on lead quality or volume?",
    "logistics": "For logistics, we automate ops updates and build live dashboards. Which bottleneck slows your team most right now?",
    "retail": "For retail, we improve site speed and conversion and automate customer queries. Is conversion or retention more urgent?",
    "saas": "For SaaS, we speed up onboarding and add AI ticket deflection. Should we look at support load or trial conversion first?",
    "education": "For education, we automate admissions queries and build progress dashboards. Are you aiming for more enrollments?",
    "fitness": "For fitness, we streamline signups and reminders and improve discovery. Is churn reduction on your radar?",
    "legal": "For legal firms, we triage intake with AI and improve site speed/SEO. Are you targeting better-qualified leads?",
    "hotel": "For hotels, we improve direct booking UX and local search. Would lifting direct bookings help margins?",
    "construction": "For construction, we streamline lead intake and progress updates. Is visibility into costs and timelines a pain?",
}
ECOM_HINTS = ("ecommerce","e-commerce","e commerce","online store","marketplace","shopify","amazon","ebay","etsy","woocommerce")
FOOTWEAR_HINTS = ("shoe","shoes","footwear","sneaker","sneakers","boot","boots")

def pitch_guard(user_text: str) -> Optional[str]:
    t = (user_text or "").lower()
    t_norm = re.sub(r"[-_/]+", " ", t)
    is_ecom = any(h in t_norm for h in ECOM_HINTS)
    is_footwear = any(h in t_norm for h in FOOTWEAR_HINTS)
    if is_ecom or is_footwear:
        reply = BUSINESS_PITCHES["ecommerce"]
        conv.append("assistant", reply); enqueue_turn("assistant", reply)
        enqueue_learn_async("pitch_ecommerce", reply)
        speak(reply, user_text_for_name_detect=user_text)
        return reply
    for kw, msg in BUSINESS_PITCHES.items():
        if kw in t_norm:
            reply = msg
            conv.append("assistant", reply); enqueue_turn("assistant", reply)
            enqueue_learn_async(f"pitch_{kw}", reply)
            speak(reply, user_text_for_name_detect=user_text)
            return reply
    return None

# ==============================
# LLM calls
# ==============================
def _end_call_checks(plow: str, original_text: str) -> Optional[str]:
    if END_CALL_STRICT:
        if EXACT_END_PHRASE and EXACT_END_PHRASE in plow:
            reply = "Understood. Ending the call now."
            conv.append("assistant", reply); enqueue_turn("assistant", reply)
            speak(reply, user_text_for_name_detect=original_text)
            set_stop("explicit_end_phrase")
            return reply
        if any(p in plow for p in END_CALL_PHRASES):
            reply = f"If you want me to end the call, please say exactly: {EXACT_END_PHRASE}. Otherwise, how can I help?"
            conv.append("assistant", reply); enqueue_turn("assistant", reply)
            speak(reply, user_text_for_name_detect=original_text)
            return reply
    return None

def deepseek_reply(prompt: str) -> str:
    norm = normalize_text(prompt)
    conv.append("user", prompt); enqueue_turn("user", prompt)

    plow = prompt.lower()
    g = _end_call_checks(plow, prompt)
    if g is not None: return g

    # Guards — order: location → pitch → price
    g = location_guard(prompt)
    if g: return g
    g = pitch_guard(prompt)
    if g: return g
    g = price_guard(prompt)
    if g: return g

    # Static / learned / cached
    if norm in STATIC_RESPONSES:
        reply = shape_for_tts(STATIC_RESPONSES[norm]); conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply); return reply
    if norm in LEARNED_PHRASES:
        reply = shape_for_tts(LEARNED_PHRASES[norm]["reply"]); conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply); return reply
    if STATIC_RESPONSES:
        m = max(((k, fuzz.token_sort_ratio(norm, k)) for k in STATIC_RESPONSES.keys()), key=lambda x: x[1], default=None)
        if m and m[1] >= 85:
            reply = shape_for_tts(STATIC_RESPONSES[m[0]]); conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply); return reply
    if LEARNED_PHRASES:
        m2 = max(((k, fuzz.token_sort_ratio(norm, k)) for k in LEARNED_PHRASES.keys()), key=lambda x: x[1], default=None)
        if m2 and m2[1] >= 88:
            reply = shape_for_tts(LEARNED_PHRASES[m2[0]]["reply"]); conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply); return reply

    cache_key = hashlib.sha256(norm.encode()).hexdigest()
    try: cached = rds.get(cache_key)
    except Exception: cached = None
    if cached:
        reply = shape_for_tts(cached); conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply); return reply

    if not client:
        reply = shape_for_tts("Sorry, my brain isn’t online right now. Could we try again shortly?")
        conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply)
        return reply

    try:
        messages = [{"role":"system","content":SYSTEM_PROMPT}, *conv.load()]
        resp = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.6, top_p=0.9, presence_penalty=0.2, frequency_penalty=0.2,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "120")),
            messages=messages,
            timeout=12,
        )
        raw = (resp.choices[0].message.content or "").strip() or "Sorry, I didn’t catch that. Could you repeat?"
        reply = shape_for_tts(raw)
        try: rds.setex(cache_key, 86400, reply)
        except Exception: pass
        conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply)
        return reply
    except Exception as e:
        print(f"⚠️ DeepSeek error: {e}")
        reply = shape_for_tts("I’m having a connection issue. Can we try again in a moment?")
        conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply)
        return reply

# --- Streaming flush policy (no choppy mid-sentence output) ---
STREAM_FLUSH_ON_PUNCT_ONLY = os.getenv("STREAM_FLUSH_ON_PUNCT_ONLY","1") not in ("0","false","False")
STREAM_FIRST_FLUSH_CHARS = int(os.getenv("STREAM_FIRST_FLUSH_CHARS","0"))  # 0 = disabled
STREAM_BUFFER_FLUSH_CHARS = int(os.getenv("STREAM_BUFFER_FLUSH_CHARS","0"))  # 0 = disabled

def deepseek_stream_and_speak(prompt: str) -> str:
    norm = normalize_text(prompt)
    conv.append("user", prompt); enqueue_turn("user", prompt)

    plow = prompt.lower()
    g = _end_call_checks(plow, prompt)
    if g is not None: return g

    # Guards — order: location → pitch → price
    g = location_guard(prompt)
    if g: return g
    g = pitch_guard(prompt)
    if g: return g
    g = price_guard(prompt)
    if g: return g

    cache_key = hashlib.sha256(norm.encode()).hexdigest()
    try: cached = rds.get(cache_key)
    except Exception: cached = None
    if norm in STATIC_RESPONSES:
        reply = shape_for_tts(STATIC_RESPONSES[norm]); conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply); return reply
    if cached:
        reply = shape_for_tts(cached); conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply); return reply

    if not client:
        reply = shape_for_tts("Sorry, my brain isn’t online right now. Could we try again shortly?")
        conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply)
        return reply

    messages = [{"role":"system","content":SYSTEM_PROMPT}, *conv.load()]
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.6, top_p=0.9, presence_penalty=0.2, frequency_penalty=0.2,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS","120")),
            messages=messages,
            stream=True,
            timeout=12,
        )

        buffer, full = [], []
        last_emitted_tail = ""
        first_flush_done = False

        def flush():
            nonlocal last_emitted_tail, first_flush_done
            if not buffer: return
            chunk = "".join(buffer)
            if last_emitted_tail in {",",";",":",")",'"',"”"} and chunk and chunk[0].isalnum():
                chunk = " " + chunk
            # For streamed chunks we keep natural style; final message will be shaped.
            speak(to_speakable(chunk))
            last_emitted_tail = chunk[-1] if chunk else last_emitted_tail
            buffer.clear()
            first_flush_done = True

        for chunk in resp:
            if not getattr(chunk, "choices", None):
                continue
            delta = getattr(chunk.choices[0].delta, "content", None)
            if not delta:
                continue

            full.append(delta)
            buffer.append(delta)

            last = delta[-1:]
            def _buflen(): return sum(len(x) for x in buffer)

            if STREAM_FLUSH_ON_PUNCT_ONLY:
                should_flush = last in (".","?","!")
            else:
                should_flush = False
                if not first_flush_done and STREAM_FIRST_FLUSH_CHARS > 0 and _buflen() >= STREAM_FIRST_FLUSH_CHARS:
                    should_flush = True
                elif last in (".","?","!"):
                    should_flush = True
                elif STREAM_BUFFER_FLUSH_CHARS > 0 and _buflen() >= STREAM_BUFFER_FLUSH_CHARS:
                    should_flush = True

            if should_flush:
                flush()

        flush()
        reply = shape_for_tts("".join(full).strip() or "Could you repeat that?")
        try: rds.setex(cache_key, 86400, reply)
        except Exception: pass
        conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply)
        return reply

    except Exception as e:
        print(f"⚠️ DeepSeek stream error: {e}")
        reply = shape_for_tts("I’m having a connection issue. Can we try again in a moment?")
        conv.append("assistant", reply); enqueue_turn("assistant", reply); enqueue_learn_async(norm, reply)
        return reply

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

    # Barge-in echo resistance
    barge_run = 0
    echo_rms  = 0.0

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

            if now < cooldown_until:
                pre_ring.clear()
                triggered = False
                continue

            frame_rms = EnergyGate._rms(frame)

            vad_speech = vad.is_speech(frame, SAMPLE_RATE)
            energy_ok  = gate.loud_enough(frame)
            is_speech  = vad_speech and energy_ok

            # Echo-aware barge-in:
            if tts_playing.is_set():
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

                # Early finalize (punctuation)
                if EARLY_FINALIZE_PUNCT and silence >= EARLY_SILENCE_FR:
                    frames_since_partial += 1
                    if frames_since_partial >= max(1, (200 // FRAME_DURATION)):
                        frames_since_partial = 0
                        txt = run_partial_for_punct()
                        if txt and txt[-1:] in ".?!" and len(txt) >= 12 and _word_count(txt) >= 3:
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
                                    commit_user_turn(full_text)
                                    _touch_activity()
                                    last_finalized_at = now
                                    cooldown_until = now + (STT_FINALIZE_COOLDOWN_MS / 1000.0)
                            triggered = False; silence = 0; speech = 0
                            pre_ring.clear(); post_ring.clear(); utter.clear()
                            last_partial = ""; frames_since_partial = 0
                            continue

                # Normal finalize after pause
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
                                    commit_user_turn(full_text)
                                    _touch_activity()
                                    last_finalized_at = now
                                    cooldown_until = now + (STT_FINALIZE_COOLDOWN_MS / 1000.0)
                        triggered = False; silence = 0; speech = 0
                        pre_ring.clear(); post_ring.clear(); utter.clear()
                        last_partial = ""; frames_since_partial = 0
        except Exception as e:
            print("⚠️ stt_worker loop error:", e)

# ==============================
# Heartbeat nudge
# ==============================
def heartbeat_worker():
    while not STOP.is_set():
        try:
            idle = _now() - _last_activity_ts
            if (idle >= HEARTBEAT_IDLE_SEC
                and not tts_playing.is_set()
                and llm_queue.empty()
                and not _pending_user_turn.strip()):
                _touch_activity()
                speak(HEARTBEAT_TEXT)
            time.sleep(1.0)
        except Exception:
            time.sleep(1.0)

# ==============================
# LLM worker
# ==============================
_first_turn_done = False
INSTANT_BACKCHANNEL = os.getenv("INSTANT_BACKCHANNEL","1") not in ("0","false","False")

def llm_worker():
    global _first_turn_done
    while not STOP.is_set():
        try:
            text = llm_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            with LLM_LOCK:
                if text is None:
                    continue

                plow = text.lower()
                g = _end_call_checks(plow, text)
                if g is not None:
                    continue

                if INSTANT_BACKCHANNEL and not _first_turn_done:
                    speak("Okay, one moment.")

                if os.getenv("LLM_STREAM","1") not in ("0","false","False"):
                    reply = deepseek_stream_and_speak(text)
                else:
                    reply = deepseek_reply(text)
                    speak(reply, user_text_for_name_detect=text)

                print(f"🤖 CloumenAI: {reply}")
                _first_turn_done = True

        except Exception as e:
            print("⚠️ llm_worker error:", e)
        finally:
            try: llm_queue.task_done()
            except Exception as e:
                print("⚠️ llm_worker task_done error:", e)

# ==============================
# Warmup
# ==============================
def _warmup_llm():
    if not client:
        return
    try:
        client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role":"system","content":"ping"}, {"role":"user","content":"hi"}],
            max_tokens=1, timeout=5
        )
    except Exception:
        pass

# ==============================
# Intro opener
# ==============================
def intro_opening():
    global _first_turn_done
    opener = (
        "Hi, this is Emma from Cloumen. We help businesses automate work and modernize apps. "
        "Could I confirm your name and what kind of business you run?"
    )
    conv.append("assistant", opener); enqueue_turn("assistant", opener)
    speak(opener)
    _first_turn_done = True

# ==============================
# Exit & signals
# ==============================
def _cleanup_and_close():
    try: tts_shutdown()
    except Exception: pass
    try:
        close_session()
        conv.clear()
        while not io_queue.empty():
            try: io_queue.get_nowait(); io_queue.task_done()
            except Exception: break
    except Exception: pass

def _handle_signal(signum, frame):
    set_stop(f"signal:{signum}")

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

    threading.Thread(target=_warmup_llm, daemon=True).start()

    time.sleep(0.2)
    intro_opening()

    try:
        run_audio_loop()
    finally:
        set_stop(STOP_REASON or "main_exit")
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
        print(f"🛑 STOP REASON: {STOP_REASON}")

if __name__ == "__main__":
    main()
