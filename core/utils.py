# F:\ai cold calling agent\cold_calling_agent\core\utils.py
import os, re, html, json, time
from datetime import datetime

# ---- Lightweight file/json helpers (moved)
def slug(s: str) -> str:
    s = (s or "").strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in s) or "user"

def safe_write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def normalize_text(text: str) -> str:
    return (text or "").lower().strip().translate(str.maketrans("", "", ".,?!"))

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
def word_count(s: str) -> int:
    return len([w for w in _WORD_RE.findall(s or "")])

# ==============================
# TTS sanitizer / shaping (moved as-is)
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
    "what's": "what is", "what’s": "what is",
    "who's": "who is",   "who’s": "who is",
    "where's": "where is", "where’s": "where is",
    "when's": "when is", "when’s": "when is",
    "how's": "how is",   "how’s": "how is",
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
    # Replace stray "'s" that might appear after typography normalization
    s = re.sub(r"\b([A-Za-z]+)\s*'\s*s\b", r"\1 is", s)
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
    original = text
    s = html.unescape(text or "")
    s = _normalize_typography(s)
    s = _strip_markdown(s)
    s = URL_RE.sub("", s)
    s = _maybe_expand_contractions(s)
    s = _strip_emotes(s)
    s = _strip_emojis(s)
    s = _fix_stream_artifacts(s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        print(f"[TTS DEBUG] to_speakable() returned empty for: {repr(original)}")
    return s


_PUNCT_FIX = re.compile(r"\s+([,.!?])")
def tidy_punctuation(s: str) -> str:
    s = _PUNCT_FIX.sub(r"\1", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if s and s[-1] not in ".?!":
        s += "."
    return s

TTS_SHAPER_MODE = os.getenv("TTS_SHAPER_MODE", "light").lower()  # off|light|strict
SENTENCE_MAX_WORDS   = int(os.getenv("SENTENCE_MAX_WORDS", "18"))
MAX_SENTENCES_OUTPUT = int(os.getenv("MAX_SENTENCES_OUTPUT", "2"))
_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")

def _split_into_sentences(s: str):
    if TTS_SHAPER_MODE == "strict":
        s = s.replace("…", ".").replace(";", ".")
    return [p.strip() for p in _SENT_SPLIT.split(s or "") if p.strip()]

def _cap_words(sent: str):
    if TTS_SHAPER_MODE == "off":
        return sent
    words = (sent or "").split()
    if len(words) <= SENTENCE_MAX_WORDS:
        return sent
    mid = len(words) // 2
    for i in range(max(1, mid - 3), min(len(words) - 1, mid + 4)):
        if words[i].lower() in {"and", "but"}:
            left = " ".join(words[:i]); right = " ".join(words[i + 1:])
            return f"{left}. {right}"
    left = " ".join(words[:SENTENCE_MAX_WORDS]); right = " ".join(words[SENTENCE_MAX_WORDS:])
    return f"{left}. {right}"

def _prune_to_two(sentences):
    if TTS_SHAPER_MODE == "off":
        return sentences
    if len(sentences) <= MAX_SENTENCES_OUTPUT:
        return sentences
    q = None
    for i in range(len(sentences) - 1, 0, -1):
        if sentences[i].endswith("?"):
            q = i
            break
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
    s = tidy_punctuation(raw or "")
    s = shorten_for_tts(s)
    s = tidy_punctuation(s)
    return to_speakable(s)

# ---- simple timestamp
def utcnow_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ===============================================================
# Async name detection helper
# ===============================================================
import threading
from services.name_detect import detect_name_from_text

def set_user_identity_async(user_text: str):
    """Detects and stores user's name asynchronously from text."""
    if not user_text:
        return
    def _task():
        try:
            name = detect_name_from_text(user_text)
            if name:
                print(f"[Memory] Detected user name: {name}")
        except Exception as e:
            print(f"[Memory] Name detection failed: {e}")
    threading.Thread(target=_task, daemon=True).start()


# (optional) public API for easier imports elsewhere
__all__ = [
    "slug", "safe_write_json", "load_json", "normalize_text", "word_count",
    "to_speakable", "shape_for_tts", "shorten_for_tts", "tidy_punctuation",
    "utcnow_iso", "set_user_identity_async"
]
