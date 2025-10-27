import os, time, collections, queue
import numpy as np
from faster_whisper import WhisperModel
from core.audio import audio_queue, FRAME_SIZE, SAMPLE_RATE, FRAME_DURATION, vad, EnergyGate
from core.turn import commit_user_turn, touch_activity
from core.utils import word_count
from core import store
from core.tts import tts_interrupt, get_tts_started_ts

# Anti-phantom / finalize knobs
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

START_TRIGGER_MS     = int(os.getenv("START_TRIGGER_MS","180"))
START_TRIGGER_FRAMES = max(1, START_TRIGGER_MS // FRAME_DURATION)

MIN_VALID_WORDS      = int(os.getenv("MIN_VALID_WORDS","2"))
MIN_VALID_CHARS      = int(os.getenv("MIN_VALID_CHARS","6"))

BARGE_IN_TRIGGER_MS  = int(os.getenv("BARGE_IN_TRIGGER_MS","260"))
BARGE_IN_MIN_RMS_MULT= float(os.getenv("BARGE_IN_MIN_RMS_MULT","2.2"))
BARGE_IN_ECHO_MULT   = float(os.getenv("BARGE_IN_ECHO_MULT","2.5"))
from core.tts import tts_playing

DECODE = dict(
    language=os.getenv("LANGUAGE","en"),
    beam_size=1,
    without_timestamps=True,
    vad_filter=False,
    condition_on_previous_text=False,
    temperature=0.0,
    log_prob_threshold=-2.0,
    compression_ratio_threshold=2.8,
)

def make_model():
    print("🔄 Loading Faster-Whisper model...")
    m = WhisperModel(os.getenv("WHISPER_SIZE","small"), device=os.getenv("WHISPER_DEVICE","cuda"), compute_type="float16", device_index=0)
    try:
        _ = m.transcribe(np.zeros(3200, dtype=np.float32), language=os.getenv("LANGUAGE","en"), beam_size=1)
    except Exception as e:
        print("⚠️ warmup ignored:", e)
    print("✅ Whisper loaded!")
    return m

_FILLER_TOKENS = {"um","uh","hmm","mm","erm","uhm","ah","oh","yeah","yep","yup","okay","ok","okey","uh-huh","huh","hmm-mm","mmm","hmm-m"}
def _looks_like_filler(s: str) -> bool:
    if not s: return True
    import re
    t = re.sub(r"[^a-zA-Z\s'-]", " ", s.lower()).strip()
    if not t: return True
    toks = [w for w in t.split() if w]
    if not toks: return True
    if len(" ".join(toks)) <= 8 and all(w in _FILLER_TOKENS for w in toks): return True
    if len(toks) == 1 and toks[0] in _FILLER_TOKENS: return True
    return False

def _text_seems_valid(s: str) -> bool:
    if not s: return False
    if _looks_like_filler(s): return False
    if len(s.strip()) < MIN_VALID_CHARS: return False
    if word_count(s) < MIN_VALID_WORDS: return False
    return True

def stt_worker(model: WhisperModel):
    print("🎧 Ready! Start speaking… (Ctrl+C to stop)")
    pre_ring  = collections.deque(maxlen=PRE_FRAMES)
    post_ring = collections.deque(maxlen=POST_FRAMES)
    triggered = False; silence = 0; speech = 0; hold_left = 0
    utter = bytearray(); last_partial = ""; frames_since_partial = 0
    last_finalized_at = 0.0
    cooldown_until = 0.0

    gate = EnergyGate()

    # Echo-aware barge-in
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

    while not store.STOP.is_set():
        try:
            if not audio_queue:
                time.sleep(0.02)
                continue
            frame = audio_queue.popleft()
        except Exception:
            continue

        try:
            now = time.time()

            if now < cooldown_until:
                pre_ring.clear()
                triggered = False
                continue

            frame_rms = EnergyGate.rms(frame)

            vad_speech = vad.is_speech(frame, SAMPLE_RATE)
            energy_ok  = gate.loud_enough(frame)
            is_speech  = vad_speech and energy_ok

            # Echo-aware barge-in:
            if tts_playing.is_set():
                echo_rms = (float(os.getenv("ECHO_TRACK_DECAY","0.93")) * echo_rms) + ((1.0 - float(os.getenv("ECHO_TRACK_DECAY","0.93"))) * frame_rms)
                loud_vs_ambient = frame_rms >= (gate.baseline or 0.0) * BARGE_IN_MIN_RMS_MULT
                loud_vs_echo    = frame_rms >= max(1e-6, echo_rms) * float(os.getenv("BARGE_IN_ECHO_MULT","2.5"))
                if is_speech and loud_vs_ambient and loud_vs_echo:
                    barge_run += 1
                else:
                    barge_run = 0
                if barge_run >= max(1, int(os.getenv("BARGE_IN_TRIGGER_MS","260")) // FRAME_DURATION) and ((time.time() - get_tts_started_ts()) * 1000.0 >= int(os.getenv("TTS_BARGE_IN_PROTECT_MS","700"))):
                    tts_interrupt()
                    barge_run = 0

            if is_speech:
                touch_activity()

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
                if EARLY_FINALIZE_PUNCT and silence >= max(1, int(os.getenv("EARLY_SILENCE_MS","220")) // FRAME_DURATION):
                    frames_since_partial += 1
                    if frames_since_partial >= max(1, (200 // FRAME_DURATION)):
                        frames_since_partial = 0
                        txt = run_partial_for_punct()
                        if txt and txt[-1:] in ".?!" and len(txt) >= 12 and word_count(txt) >= 3:
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
                                    touch_activity()
                                    last_finalized_at = now
                                    cooldown_until = now + (int(os.getenv("STT_FINALIZE_COOLDOWN_MS","180")) / 1000.0)
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
                                    touch_activity()
                                    last_finalized_at = now
                                    cooldown_until = now + (int(os.getenv("STT_FINALIZE_COOLDOWN_MS","180")) / 1000.0)
                        triggered = False; silence = 0; speech = 0
                        pre_ring.clear(); post_ring.clear(); utter.clear()
                        last_partial = ""; frames_since_partial = 0
        except Exception as e:
            print("⚠️ stt_worker loop error:", e)

def run_audio_loop():
    from core.audio import run_audio_loop as _run
    _run(store.STOP)
