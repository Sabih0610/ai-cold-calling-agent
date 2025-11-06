#core\llm.py
import os, time, hashlib, threading, queue
from typing import Optional, Callable
import hashlib
from openai import OpenAI
from rapidfuzz import fuzz
from core.prompts import SYSTEM_PROMPT
from core.utils import normalize_text, shape_for_tts
from core import store
from core.tts import speak, tts_playing
from core import turn
from core import guards

# ===============================================================
# DeepSeek / LLM Configuration
# ===============================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
client = OpenAI(base_url="https://api.deepseek.com/v1", api_key=DEEPSEEK_API_KEY) if DEEPSEEK_API_KEY else None
rds = store.rds

LLM_DEBOUNCE_WINDOW_MS = int(os.getenv("LLM_DEBOUNCE_WINDOW_MS", "1000"))
LLM_DEBOUNCE_SIM       = int(os.getenv("LLM_DEBOUNCE_SIM", "92"))
LLM_RECENT_TTL_SEC     = float(os.getenv("LLM_RECENT_TTL_SEC", "3.0"))

STREAM_FLUSH_ON_PUNCT_ONLY = os.getenv("STREAM_FLUSH_ON_PUNCT_ONLY","1") not in ("0","false","False")
STREAM_FIRST_FLUSH_CHARS = int(os.getenv("STREAM_FIRST_FLUSH_CHARS","0"))
STREAM_BUFFER_FLUSH_CHARS = int(os.getenv("STREAM_BUFFER_FLUSH_CHARS","0"))

_context_provider: Callable[[], str] = lambda: ""
_hint_provider: Callable[[], str] = lambda: ""

SPECIAL_COMMANDS = {"start_call"}

# ===============================================================
# Providers Setup
# ===============================================================
def set_providers(context_provider: Callable[[], str] | None, hint_provider: Callable[[], str] | None):
    global _context_provider, _hint_provider
    _context_provider = context_provider or (lambda: "")
    _hint_provider    = hint_provider or (lambda: "")

# ===============================================================
# Queue / Debounce Logic
# ===============================================================
llm_queue = queue.Queue()
LLM_LOCK = threading.Lock()
_last_enq_text = ""
_last_enq_ts = 0.0
_recent_texts = []

def _similar(a: str, b: str) -> int:
    try: return fuzz.ratio(a, b)
    except Exception: return 0

def _accept_against_recent(text: str) -> bool:
    now = time.time()
    global _recent_texts
    _recent_texts = [(ts, t) for (ts, t) in _recent_texts if (now - ts) <= LLM_RECENT_TTL_SEC]
    for _, t in _recent_texts:
        if _similar(text.lower(), t.lower()) >= 96:
            return False
    _recent_texts.append((now, text))
    return True

def enqueue_user_text(text: str):
    global _last_enq_text, _last_enq_ts
    s = (text or "").strip()
    if not s: return
    now = time.time()
    if _last_enq_text and (now - _last_enq_ts) * 1000.0 < LLM_DEBOUNCE_WINDOW_MS:
        if _similar(s.lower(), _last_enq_text.lower()) >= LLM_DEBOUNCE_SIM:
            return
    if not _accept_against_recent(s): return
    _last_enq_text = s; _last_enq_ts = now
    try:
        while True:
            llm_queue.get_nowait()
            llm_queue.task_done()
    except queue.Empty:
        pass
    llm_queue.put(s)

# ===============================================================
# End-call Guards
# ===============================================================
END_CALL_STRICT = os.getenv("END_CALL_STRICT","1") not in ("0","false","False")
EXACT_END_PHRASE = os.getenv("EXACT_END_PHRASE","end call now").strip().lower()
END_CALL_PHRASES = [p.strip().lower() for p in os.getenv(
    "END_CALL_PHRASES",
    "goodbye,bye,that will be all,that's all,hang up,thanks that's it,thank you that's it,end call,please disconnect,quit,exit,done"
).split(",") if p.strip()]

def _end_call_checks(plow: str, original_text: str) -> Optional[str]:
    if END_CALL_STRICT:
        if EXACT_END_PHRASE and EXACT_END_PHRASE in plow:
            reply = "Understood. Ending the call now."
            store.conv.append("assistant", reply); store.enqueue_turn("assistant", reply)
            speak(reply, user_text_for_name_detect=original_text)
            store.set_stop("explicit_end_phrase")
            return reply
        if any(p in plow for p in END_CALL_PHRASES):
            reply = f"If you want me to end the call, please say exactly: {EXACT_END_PHRASE}. Otherwise, how can I help?"
            store.conv.append("assistant", reply); store.enqueue_turn("assistant", reply)
            speak(reply, user_text_for_name_detect=original_text)
            return reply
    return None

# ===============================================================
# Memory lookup (static/learned + optional fuzzy)
# ===============================================================
def _best_memory_match(norm: str) -> Optional[str]:
    """Return a reply from STATIC or LEARNED if we have it (exact or fuzzy)."""
    # Exact first
    if norm in store.STATIC_RESPONSES:
        return store.STATIC_RESPONSES[norm]
    if norm in store.LEARNED_PHRASES:
        rec = store.LEARNED_PHRASES.get(norm) or {}
        if rec.get("reply"):
            return rec["reply"]

    # Optional fuzzy fallback
    try:
        thresh = int(os.getenv("LLM_MEMORY_FUZZY_SIM", "0"))
    except Exception:
        thresh = 0
    if thresh <= 0:
        return None

    best = (None, 0, None)  # (key, score, reply)
    try:
        for k, v in store.STATIC_RESPONSES.items():
            s = fuzz.ratio(k, norm)
            if s > best[1]:
                best = (k, s, v)
        for k, rec in store.LEARNED_PHRASES.items():
            rep = (rec or {}).get("reply")
            if not rep:
                continue
            s = fuzz.ratio(k, norm)
            if s > best[1]:
                best = (k, s, rep)
    except Exception:
        return None
    return best[2] if best[1] >= thresh else None

# ===============================================================
# Conversation Helpers
# ===============================================================
def default_opener():
    return ("Hi there, this is Emma from Cloumen. We help small teams adopt smarter tech so they keep growing. "
            "We handle AI automation, web, and cloud/data support - how do you keep leads moving today?")

def generate_opener(context_text: str = "", hint_text: str = "") -> str:
    """
    Generate an opener without speaking or mutating conversation state.
    Used for pre-call preparation so we can start talking immediately when the call connects.
    """
    if not client:
        return shape_for_tts(default_opener())

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context_text = (context_text or "").strip()
    hint_text = (hint_text or "").strip()
    if context_text:
        messages.append({"role": "user", "content": context_text})
    if hint_text:
        messages.append({"role": "user", "content": "[FLOW HINT]\n" + hint_text})
    messages.append({"role": "user", "content": "[START_CALL]"})

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.6,
            top_p=0.9,
            presence_penalty=0.2,
            frequency_penalty=0.2,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "120")),
            messages=messages,
            timeout=12,
        )
        raw = (resp.choices[0].message.content or "").strip() or default_opener()
        return shape_for_tts(raw)
    except Exception as e:
        print(f"⚠️ DeepSeek opener generation error: {e}")
        return shape_for_tts(default_opener())

def _assemble_messages():
    msgs = [{"role":"system","content":SYSTEM_PROMPT}, *store.conv.load()]
    context = (_context_provider() or "").strip()
    # Send context once per unique context blob (new lead ⇒ new hash).
    if context:
        h = hashlib.sha256(context.encode("utf-8")).hexdigest()
        # Ensure per-conversation context injection: include conv key
        try:
            conv_key = getattr(store.conv, "key", "conv")
        except Exception:
            conv_key = "conv"
        context_key = f"ctxsent:{conv_key}:{h}"
        try: already = rds.get(context_key)
        except Exception: already = None
        if not already:
            msgs.append({"role":"user","content":context})
            try: rds.setex(context_key, 3600, "1")
            except Exception: pass

        
    hint = (_hint_provider() or "").strip()
    if hint:
        msgs.append({"role":"user","content":"[FLOW HINT]\n"+hint})
    return msgs

# ===============================================================
# Non-Streaming Reply
# ===============================================================
def deepseek_reply(prompt: str) -> str:
    norm = normalize_text(prompt)
    is_command = norm in SPECIAL_COMMANDS
    store.conv.append("user", prompt); store.enqueue_turn("user", prompt)
    plow = prompt.lower()

    g = _end_call_checks(plow, prompt)
    if g is not None: return g

    g = guards.location_guard(prompt)
    if g: return g
    g = guards.pitch_guard(prompt)
    if g: return g
    g = guards.price_guard(prompt)
    if g: return g

    # Memory first: static/learned, then cache, then LLM
    if not is_command:
        mem = _best_memory_match(norm)
        if mem:
            reply = shape_for_tts(mem)
            store.conv.append("assistant", reply)
            store.enqueue_turn("assistant", reply)
            store.enqueue_learn_async(norm, reply)
            speak(reply)
            return reply

    conv_key = getattr(store.conv, "key", "conv")
    cache_key = hashlib.sha256(f"{conv_key}|{norm}".encode()).hexdigest()
    try: cached = rds.get(cache_key)
    except Exception: cached = None
    if cached:
        reply = shape_for_tts(cached)
        store.conv.append("assistant", reply)
        store.enqueue_turn("assistant", reply)
        if not is_command:
            store.enqueue_learn_async(norm, reply)
        speak(reply)
        return reply

    if not client:
        reply = shape_for_tts("Sorry, my brain isn’t online right now. Could we try again shortly?")
        store.conv.append("assistant", reply)
        store.enqueue_turn("assistant", reply)
        if not is_command:
            store.enqueue_learn_async(norm, reply)
        speak(reply)
        return reply

    try:
        messages = _assemble_messages()
        resp = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.6, top_p=0.9,
            presence_penalty=0.2, frequency_penalty=0.2,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS","120")),
            messages=messages,
            timeout=12,
        )
        raw = (resp.choices[0].message.content or "").strip() or "Sorry, I didn’t catch that. Could you repeat?"
        reply = shape_for_tts(raw)
        try: rds.setex(cache_key, 86400, reply)
        except Exception: pass
        store.conv.append("assistant", reply)
        store.enqueue_turn("assistant", reply)
        if not is_command:
            store.enqueue_learn_async(norm, reply)
        speak(reply)
        return reply
    except Exception as e:
        print(f"⚠️ DeepSeek error: {e}")
        reply = shape_for_tts("I’m having a connection issue. Can we try again in a moment?")
        store.conv.append("assistant", reply)
        store.enqueue_turn("assistant", reply)
        if not is_command:
            store.enqueue_learn_async(norm, reply)
        speak(reply)
        return reply

# ===============================================================
# Streaming Reply (with Deduplication + Final Speak)
# ===============================================================
def deepseek_stream_and_speak(prompt: str) -> str:
    norm = normalize_text(prompt)
    is_command = norm in SPECIAL_COMMANDS
    store.conv.append("user", prompt); store.enqueue_turn("user", prompt)

    plow = prompt.lower()
    g = _end_call_checks(plow, prompt)
    if g is not None: return g

    g = guards.location_guard(prompt)
    if g: return g
    g = guards.pitch_guard(prompt)
    if g: return g
    g = guards.price_guard(prompt)
    if g: return g

    # Memory first: static/learned, then cache, then LLM
    if not is_command:
        mem = _best_memory_match(norm)
        if mem:
            reply = shape_for_tts(mem)
            store.conv.append("assistant", reply); store.enqueue_turn("assistant", reply)
            store.enqueue_learn_async(norm, reply); speak(reply); return reply
    conv_key = getattr(store.conv, "key", "conv")
    cache_key = hashlib.sha256(f"{conv_key}|{norm}".encode()).hexdigest()
    try: cached = rds.get(cache_key)
    except Exception: cached = None
    if cached:
        reply = shape_for_tts(cached)
        store.conv.append("assistant", reply); store.enqueue_turn("assistant", reply)
        if not is_command:
            store.enqueue_learn_async(norm, reply)
        speak(reply); return reply

    if not client:
        reply = shape_for_tts("Sorry, my brain isn’t online right now. Could we try again shortly?")
        store.conv.append("assistant", reply); store.enqueue_turn("assistant", reply)
        if not is_command:
            store.enqueue_learn_async(norm, reply)
        speak(reply); return reply

    messages = _assemble_messages()
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.6, top_p=0.9,
            presence_penalty=0.2, frequency_penalty=0.2,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS","120")),
            messages=messages,
            stream=True,
            timeout=12,
        )

        buffer, full = [], []
        last_emitted_tail = ""
        first_flush_done = False
        last_spoken = [""]

        def flush():
            nonlocal last_emitted_tail, first_flush_done
            if not buffer: return
            chunk = "".join(buffer).strip()
            if not chunk: return

            # Light clean-up (avoid "ok.", ".", "thanks.")
            if len(chunk) <= 3 and all(c in ".!?," for c in chunk):
                buffer.clear()
                return

            # Deduplicate: skip if >90% similar to last spoken
            sim = fuzz.partial_ratio(chunk, last_spoken[0]) if last_spoken[0] else 0
            if sim >= 90:
                buffer.clear()
                return

            if last_emitted_tail in {",", ";", ":", ")", '"', "”"} and chunk and chunk[0].isalnum():
                chunk = " " + chunk

            speak(chunk)
            store.publish_event({"type": "assistant_turn", "text": chunk})
            last_emitted_tail = chunk[-1] if chunk else last_emitted_tail
            last_spoken[0] = chunk
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

        # ✅ Final guaranteed flush
        flush()

        # ✅ Final reply synthesis (avoid replay)
        reply = shape_for_tts("".join(full).strip() or "Could you repeat that?")
        sim_final = fuzz.partial_ratio(reply, last_spoken[0]) if last_spoken[0] else 0
        if sim_final < 90:
            speak(reply)

        try: rds.setex(cache_key, 86400, reply)
        except Exception: pass
        store.conv.append("assistant", reply)
        store.enqueue_turn("assistant", reply)
        if not is_command:
            store.enqueue_learn_async(norm, reply)
        return reply

    except Exception as e:
        print(f"⚠️ DeepSeek stream error: {e}")
        reply = shape_for_tts("I’m having a connection issue. Can we try again in a moment?")
        store.conv.append("assistant", reply)
        store.enqueue_turn("assistant", reply)
        if not is_command:
            store.enqueue_learn_async(norm, reply)
        speak(reply)
        return reply


# ===============================================================
# LLM Worker Thread
# ===============================================================
_first_turn_done = False
INSTANT_BACKCHANNEL = os.getenv("INSTANT_BACKCHANNEL","1") not in ("0","false","False")

def llm_worker():
    global _first_turn_done
    while not store.STOP.is_set():
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

                print(f"🤖 CloumenAI: {reply}")
                store.publish_event({"type": "assistant_turn", "text": reply})
                _first_turn_done = True
        except Exception as e:
            print("⚠️ llm_worker error:", e)
        finally:
            try: llm_queue.task_done()
            except Exception: pass

# ===============================================================
# Warmup Function
# ===============================================================
def warmup_llm():
    if not client: return
    try:
        client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role":"system","content":"ping"}, {"role":"user","content":"hi"}],
            max_tokens=1,
            timeout=5
        )
    except Exception:
        pass
