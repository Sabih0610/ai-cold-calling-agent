# core/stt.py
import os
import time
import collections
import concurrent.futures
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from core.audio import (
    audio_queue,
    FRAME_SIZE,
    SAMPLE_RATE,
    FRAME_DURATION,
    vad,
    EnergyGate,
)
from core.turn import commit_user_turn, touch_activity
from core.utils import word_count
from core import store, llm, backchannel
from core.tts import (
    tts_interrupt,
    get_tts_started_ts,
    tts_soft_interrupt,
    tts_playing,
)

# ---- First-turn arming (compat with live_voice_agent opener) ----
_SEEDED_RMS: Optional[float] = None


def arm_first_turn(seed_rms: Optional[float] = None):
    """Called from opener to seed baseline noise level for EnergyGate."""
    global _SEEDED_RMS
    _SEEDED_RMS = float(seed_rms) if (seed_rms and seed_rms > 0) else None


def reset_first_turn():
    global _SEEDED_RMS
    _SEEDED_RMS = None


def clear_audio_queue():
    """Hard flush any stale RX frames."""
    try:
        audio_queue.clear()
    except Exception:
        pass


# ---- Endpointing / anti-phantom knobs ----
END_SILENCE_MS = int(os.getenv("END_SILENCE_MS", "260"))
CONTINUE_WINDOW_MS = int(os.getenv("CONTINUE_WINDOW_MS", "160"))
PRE_ROLL_MS = int(os.getenv("PRE_ROLL_MS", "120"))
POST_ROLL_MS = int(os.getenv("POST_ROLL_MS", "120"))
MIN_TURN_MS = int(os.getenv("MIN_TURN_MS", "120"))

EARLY_FINALIZE_PUNCT = os.getenv("EARLY_FINALIZE_PUNCT", "1") not in (
    "0",
    "false",
    "False",
)
EARLY_SILENCE_MS = int(os.getenv("EARLY_SILENCE_MS", "260"))

ENABLE_STT_PARTIALS = os.getenv("STT_PARTIALS_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)
PARTIAL_INTERVAL_MS = int(os.getenv("STT_PARTIAL_INTERVAL_MS", "180"))
PARTIAL_WINDOW_MS = int(os.getenv("STT_PARTIAL_WINDOW_MS", "1400"))
PARTIAL_MIN_WORDS = int(os.getenv("STT_PARTIAL_MIN_WORDS", "2"))
MAX_STREAM_UTTER_MS = int(os.getenv("STT_MAX_STREAM_UTTER_MS", "9000"))

STT_DEBUG = os.getenv("STT_DEBUG", "0") not in ("0", "false", "False")

# Overall STT sensitivity
STT_MIN_VOICE_RMS = float(os.getenv("STT_MIN_VOICE_RMS", "0.004"))
MIN_VOICE_RMS = STT_MIN_VOICE_RMS

END_SILENCE_FRAMES = max(1, END_SILENCE_MS // FRAME_DURATION)
CONTINUE_WINDOW_FR = max(1, CONTINUE_WINDOW_MS // FRAME_DURATION)
PRE_FRAMES = max(1, PRE_ROLL_MS // FRAME_DURATION)
POST_FRAMES = max(1, POST_ROLL_MS // FRAME_DURATION)
MIN_TURN_FRAMES = max(1, MIN_TURN_MS // FRAME_DURATION)

# Trigger quicker when you start talking (first onset)
START_TRIGGER_MS = int(os.getenv("START_TRIGGER_MS", "180"))
START_TRIGGER_FRAMES = max(1, START_TRIGGER_MS // FRAME_DURATION)

MIN_VALID_WORDS = int(os.getenv("MIN_VALID_WORDS", "2"))
MIN_VALID_CHARS = int(os.getenv("MIN_VALID_CHARS", "4"))

# ---- Barge-in knobs ----
BARGE_IN_TRIGGER_MS = int(os.getenv("BARGE_IN_TRIGGER_MS", "140"))  # used as concept; we mainly use frames
TTS_BARGE_IN_PROTECT_MS = int(os.getenv("TTS_BARGE_IN_PROTECT_MS", "550"))

BARGE_IN_MIN_RMS_ABS = float(os.getenv("BARGE_IN_MIN_RMS_ABS", "0.006"))
BARGE_PRIORITY_SOFT = os.getenv("BARGE_PRIORITY_SOFT", "1") not in (
    "0",
    "false",
    "False",
)
BARGE_IN_DEBUG = os.getenv("BARGE_IN_DEBUG", "0") not in ("0", "false", "False")
BARGE_IN_MIN_GAP_MS = int(os.getenv("BARGE_IN_MIN_GAP_MS", "450"))
BARGE_SPEECH_CONFIRM_FRAMES = int(os.getenv("BARGE_SPEECH_CONFIRM_FRAMES", "3"))
BARGE_IN_ECHO_MULT = float(os.getenv("BARGE_IN_ECHO_MULT", "1.6"))

# ---- Faster-Whisper decode config (slim & fast) ----
DECODE = dict(
    language=os.getenv("LANGUAGE", "en"),
    beam_size=1,
    without_timestamps=True,
    vad_filter=False,
    condition_on_previous_text=False,
    temperature=0.0,
    word_timestamps=False,
)


def make_model() -> WhisperModel:
    print("🔄 Loading Faster-Whisper model...")
    m = WhisperModel(
        os.getenv("WHISPER_SIZE", "small"),
        device=os.getenv("WHISPER_DEVICE", "cuda"),
        compute_type="float16",
        device_index=0,
    )
    # Warmup small dummy audio so first real call is faster
    try:
        _ = m.transcribe(
            np.zeros(3200, dtype=np.float32),
            language=os.getenv("LANGUAGE", "en"),
            beam_size=1,
        )
    except Exception as e:
        print("⚠️ warmup ignored:", e)
    print("✅ Whisper loaded!")
    return m


# ---------- Text quality helpers ----------

_FILLER_TOKENS = {
    "um",
    "uh",
    "hmm",
    "mm",
    "erm",
    "uhm",
    "ah",
    "oh",
    "yeah",
    "yep",
    "yup",
    "okay",
    "ok",
    "okey",
    "uh-huh",
    "huh",
    "mmm",
    "hmm-mm",
    "hmm-m",
    "mmm-hmm",
}


def _looks_like_filler(s: str) -> bool:
    if not s:
        return True
    import re

    t = re.sub(r"[^a-zA-Z\s'-]", " ", s.lower()).strip()
    if not t:
        return True
    toks = [w for w in t.split() if w]
    if not toks:
        return True
    if len(" ".join(toks)) <= 8 and all(w in _FILLER_TOKENS for w in toks):
        return True
    if len(toks) == 1 and toks[0] in _FILLER_TOKENS:
        return True
    return False


def _text_seems_valid(s: str) -> bool:
    if not s:
        return False
    if _looks_like_filler(s):
        return False
    t = s.strip()
    if not t:
        return False

    wc = word_count(t)

    # Allow single-word answers only if they are useful (not pronouns/noise)
    if wc == 1:
        tok = t.lower()
        allowed = {
            "yes", "no", "yeah", "yep", "nope", "ok", "okay", "sure", "right",
            "correct", "absolutely", "ready", "go", "stop", "start", "done"
        }
        if tok in allowed:
            return True
        return False

    # For multi-word utterances, keep stricter filters
    if len(t) < MIN_VALID_CHARS:
        return False
    if wc < MIN_VALID_WORDS:
        return False
    return True


def stt_worker(model: WhisperModel):
    print("🎧 Ready! Start speaking… (Ctrl+C to stop)")

    pre_ring = collections.deque(maxlen=PRE_FRAMES)
    post_ring = collections.deque(maxlen=POST_FRAMES)

    triggered = False
    silence = 0
    speech = 0
    hold_left = 0

    utter = bytearray()
    last_finalized_at = 0.0
    cooldown_until = 0.0

    gate = EnergyGate()
    user_frames = 0

    last_partial_for_punct = ""
    frames_since_partial_probe = 0

    # Partial streaming
    frame_bytes = FRAME_SIZE * 2
    partial_interval_frames = max(1, PARTIAL_INTERVAL_MS // FRAME_DURATION)
    partial_emit_frames = 0
    last_partial_text = ""
    partial_executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=1)
        if ENABLE_STT_PARTIALS
        else None
    )
    partial_future: Optional[concurrent.futures.Future] = None
    partial_job_counter = 0
    partial_latest_job = 0
    max_stream_frames = (
        max(MIN_TURN_FRAMES, MAX_STREAM_UTTER_MS // FRAME_DURATION)
        if MAX_STREAM_UTTER_MS > 0
        else None
    )

    # Seed baseline from opener, if given
    try:
        if _SEEDED_RMS:
            gate.seed(_SEEDED_RMS)
    except Exception:
        pass

    # Simple barge-in state
    barge_run = 0
    echo_rms = 0.0
    ECHO_TRACK_DECAY = float(os.getenv("ECHO_TRACK_DECAY", "0.93"))
    last_barge_at = 0.0

    def _trigger_barge_in():
        """Cut TTS when the user talks over the agent."""
        nonlocal barge_run, last_barge_at
        last_barge_at = time.time()
        barge_run = 0
        try:
            if BARGE_PRIORITY_SOFT:
                tts_soft_interrupt(fade_ms=260)
            else:
                tts_interrupt()
        except Exception:
            try:
                tts_interrupt()
            except Exception:
                pass
        if BARGE_IN_DEBUG:
            print("[BARGE] 🔪 TTS cut by barge-in")

    # ---- Inner helpers ----

    def run_partial_for_punct() -> str:
        """Decode the full utter buffer for a 'punctuation probe'."""
        nonlocal last_partial_for_punct
        if len(utter) < 2 * FRAME_SIZE * MIN_TURN_FRAMES:
            return ""
        audio_np = np.frombuffer(utter, np.int16).astype(np.float32) / 32768.0
        try:
            segs, _ = model.transcribe(audio_np, **DECODE)
            txt = " ".join(s.text.strip() for s in segs).strip()
            if txt:
                last_partial_for_punct = txt
            return txt
        except Exception:
            return ""

    def _capture_tail_bytes() -> bytes:
        """Grab last PARTIAL_WINDOW_MS of audio from utter for partial STT."""
        if len(utter) < frame_bytes * 3:
            return b""
        samples = int(SAMPLE_RATE * PARTIAL_WINDOW_MS / 1000.0)
        tail_bytes = min(len(utter), samples * 2)
        if tail_bytes <= frame_bytes:
            return b""
        return bytes(utter[-tail_bytes:])

    def _partial_decode_job(job_id: int, buf: bytes):
        audio_np = np.frombuffer(buf, np.int16).astype(np.float32) / 32768.0
        if audio_np.size == 0:
            return job_id, ""
        try:
            segs, _ = model.transcribe(audio_np, **DECODE)
            text = " ".join(s.text.strip() for s in segs).strip()
            return job_id, text
        except Exception:
            return job_id, ""

    def _schedule_partial_job():
        nonlocal partial_future, partial_job_counter, partial_latest_job
        if not ENABLE_STT_PARTIALS or not partial_executor:
            return
        if partial_future and not partial_future.done():
            return
        buf = _capture_tail_bytes()
        if not buf:
            return
        partial_job_counter += 1
        job_id = partial_job_counter
        partial_latest_job = job_id
        partial_future = partial_executor.submit(_partial_decode_job, job_id, buf)

    def _process_partial_future():
        nonlocal partial_future, last_partial_text

        if not partial_future or not partial_future.done():
            return
        try:
            job_id, text = partial_future.result()
        except Exception:
            partial_future = None
            return
        partial_future = None
        if job_id != partial_latest_job:
            return

        clean = (text or "").strip()
        if not clean:
            return

        # === Partial streaming filters ===
        if clean == last_partial_text:
            return
        if _looks_like_filler(clean):
            return
        if len(clean) < 3 or word_count(clean) < PARTIAL_MIN_WORDS:
            return

        last_partial_text = clean
        if STT_DEBUG:
            print(f"[STT][partial] {clean}")
        try:
            llm.update_partial_transcript(clean, is_final=False)
        except Exception:
            pass

    def _clear_partial_state():
        nonlocal last_partial_text, partial_emit_frames, partial_future
        if last_partial_text:
            try:
                llm.update_partial_transcript("", is_final=True)
            except Exception:
                pass
        last_partial_text = ""
        partial_emit_frames = 0
        if partial_future and not partial_future.done():
            try:
                partial_future.cancel()
            except Exception:
                pass
        partial_future = None

    def _reset_after_finalize():
        nonlocal triggered, silence, speech, user_frames
        nonlocal last_partial_for_punct, frames_since_partial_probe
        nonlocal partial_emit_frames, last_partial_text, partial_future
        triggered = False
        silence = 0
        speech = 0
        user_frames = 0
        pre_ring.clear()
        post_ring.clear()
        utter.clear()
        last_partial_for_punct = ""
        frames_since_partial_probe = 0
        partial_emit_frames = 0
        last_partial_text = ""
        if partial_future and not partial_future.done():
            try:
                partial_future.cancel()
            except Exception:
                pass
        partial_future = None
        try:
            backchannel.reset_turn()
        except Exception:
            pass

    def _finalize_current(reason: str, min_frames: Optional[int] = None) -> bool:
        nonlocal last_finalized_at, cooldown_until
        target_frames = min_frames or MIN_TURN_FRAMES
        total_frames = len(utter) // (2 * FRAME_SIZE)
        if total_frames < target_frames:
            return False

        # Decode current utterance
        audio_np = np.frombuffer(utter, np.int16).astype(np.float32) / 32768.0
        try:
            segs, _ = model.transcribe(audio_np, **DECODE)
            full_text = " ".join(s.text.strip() for s in segs).strip()
        except Exception as e:
            print(f"⚠️ ASR error ({reason}): {e}")
            full_text = ""

        # Nothing recognized at all → just reset state, no user turn
        if not full_text:
            if STT_DEBUG:
                print(f"[STT finalize] empty text for {reason} (frames={total_frames})")
            try:
                llm.update_partial_transcript("", is_final=True)
            except Exception:
                pass
            _reset_after_finalize()
            _clear_partial_state()
            return False

        # Short / filler / low-content text → DROP IT
        if not _text_seems_valid(full_text):
            if STT_DEBUG:
                print(f"[STT drop] short/low-content ({reason}): {full_text!r}")
            try:
                llm.update_partial_transcript("", is_final=True)
            except Exception:
                pass
            _reset_after_finalize()
            _clear_partial_state()
            return False

        # 🔁 Backup barge-in: if TTS is still playing when we have a valid text,
        # kill TTS immediately so your answer doesn't keep talking over them.
        if tts_playing.is_set():
            if BARGE_IN_DEBUG:
                print("[BARGE] 🔕 cutting TTS on finalize")
            _trigger_barge_in()

        # Debounce “double finals”
        if (time.time() - last_finalized_at) * 1000.0 <= 150 and reason != "max_stream":
            return False

        # ✅ Valid final user turn
        print(f"🗣️ User ({reason}): {full_text}")
        commit_user_turn(full_text)
        touch_activity()
        last_finalized_at = time.time()
        cooldown_until = last_finalized_at + (
            int(os.getenv("STT_FINALIZE_COOLDOWN_MS", "120")) / 1000.0
        )
        try:
            llm.update_partial_transcript(full_text, is_final=True)
        except Exception:
            pass

        _reset_after_finalize()
        return True

    # ---- Main loop ----
    try:
        while not store.STOP.is_set():
            try:
                _process_partial_future()

                if not audio_queue:
                    time.sleep(0.02)
                    continue

                frame = audio_queue.popleft()
                now = time.time()

                # Cooldown after finalization
                if now < cooldown_until:
                    pre_ring.clear()
                    if triggered and user_frames:
                        user_frames = 0
                        try:
                            backchannel.reset_turn()
                        except Exception:
                            pass
                    triggered = False
                    continue

                frame_rms = EnergyGate.rms(frame)

                # Base detection: VAD + adaptive energy gate
                try:
                    vad_speech = vad.is_speech(frame, SAMPLE_RATE)
                except Exception:
                    vad_speech = frame_rms >= MIN_VOICE_RMS

                try:
                    energy_ok = gate.loud_enough(frame)
                except Exception:
                    energy_ok = frame_rms >= MIN_VOICE_RMS

                # 👇 Core speech flag for STT (never overridden by barge logic)
                is_speech = vad_speech and energy_ok and (frame_rms >= MIN_VOICE_RMS)

                # ---- Echo-aware barge-in while TTS is playing ----
                if tts_playing.is_set():
                    now_barge = time.time()

                    # Track recent RX loudness as "echo" baseline
                    echo_rms = (ECHO_TRACK_DECAY * echo_rms) + (
                        (1.0 - ECHO_TRACK_DECAY) * frame_rms
                    )

                    # Don’t barge in during very first chunk of TTS
                    allow_barge = (
                        (now_barge - get_tts_started_ts()) * 1000.0
                        >= TTS_BARGE_IN_PROTECT_MS
                    )

                    user_candidate = False
                    if allow_barge and is_speech:
                        # Baseline from gate (ambient line level)
                        baseline = getattr(gate, "baseline", None) or MIN_VOICE_RMS

                        # Threshold must be higher than:
                        # - ambient baseline
                        # - a hard absolute minimum
                        # - the recent echo level (so we don't barge on pure echo)
                        thr = max(
                            baseline * 1.3,
                            BARGE_IN_MIN_RMS_ABS,
                            echo_rms * BARGE_IN_ECHO_MULT if echo_rms > 0.0 else 0.0,
                        )
                        # Don't let threshold explode
                        thr = min(thr, 0.30)

                        if frame_rms >= thr:
                            user_candidate = True

                        if BARGE_IN_DEBUG:
                            print(
                                f"[BARGE] rms={frame_rms:.4f} "
                                f"echo={echo_rms:.4f} baseline={baseline:.4f} "
                                f"thr={thr:.4f} cand={user_candidate} run={barge_run}"
                            )

                    # Streak of "user" frames
                    if user_candidate:
                        barge_run += 1
                    else:
                        barge_run = 0

                    # After N consecutive user-like frames → cut TTS.
                    if allow_barge and barge_run >= BARGE_SPEECH_CONFIRM_FRAMES:
                        # Enforce minimum time gap between barges
                        if (
                            BARGE_IN_MIN_GAP_MS > 0
                            and last_barge_at > 0.0
                            and (now_barge - last_barge_at) * 1000.0
                            < BARGE_IN_MIN_GAP_MS
                        ):
                            if BARGE_IN_DEBUG:
                                print(
                                    "[BARGE] ⏱ gap guard – too soon after last barge"
                                )
                        else:
                            if BARGE_IN_DEBUG:
                                print(
                                    f"[BARGE] ✅ trigger (rms={frame_rms:.4f}, run={barge_run})"
                                )
                            _trigger_barge_in()

                if STT_DEBUG:
                    print(
                        f"[STT frame] rms={frame_rms:.6f} speech={is_speech} "
                        f"tts={tts_playing.is_set()}"
                    )

                if is_speech:
                    touch_activity()

                # ----- Not yet triggered: look for user start -----
                if not triggered:
                    if user_frames:
                        user_frames = 0
                        try:
                            backchannel.reset_turn()
                        except Exception:
                            pass
                    pre_ring.append(frame)
                    if is_speech:
                        speech += 1
                        if speech >= START_TRIGGER_FRAMES:
                            triggered = True
                            # backfill a little audio from before start
                            utter.extend(b"".join(pre_ring))
                            pre_ring.clear()
                            utter.extend(frame)
                            silence = 0
                            post_ring.clear()
                            frames_since_partial_probe = 0
                            last_partial_for_punct = ""
                            user_frames = 0
                            partial_emit_frames = 0
                            last_partial_text = ""
                    else:
                        if speech > 0:
                            speech = 0
                    continue

                # ----- Already in a user turn -----
                if is_speech:
                    if hold_left > 0:
                        hold_left = 0
                    silence = 0
                    post_ring.clear()
                    utter.extend(frame)
                    user_frames += 1

                    # Streaming partials (live STT while user is still talking)
                    if ENABLE_STT_PARTIALS:
                        partial_emit_frames += 1
                        if partial_emit_frames >= partial_interval_frames:
                            partial_emit_frames = 0
                            _schedule_partial_job()

                    # Hard cap on utterance length (force finalize)
                    if max_stream_frames and user_frames >= max_stream_frames:
                        if _finalize_current("max_stream"):
                            continue

                    # Optional backchannel (disabled by default via env)
                    try:
                        if (
                            os.getenv("BACKCHANNEL_ENABLE", "0")
                            not in ("0", "false", "False")
                            and not tts_playing.is_set()
                            and getattr(llm, "_first_turn_done", False)
                        ):
                            backchannel.maybe_ack(user_frames * FRAME_DURATION)
                    except Exception:
                        pass
                else:
                    # No speech this frame
                    partial_emit_frames = 0
                    if user_frames:
                        user_frames = 0
                        try:
                            backchannel.reset_turn()
                        except Exception:
                            pass
                    silence += 1
                    post_ring.append(frame)
                    utter.extend(frame)

                    # Early finalize using punctuation probe
                    if (
                        EARLY_FINALIZE_PUNCT
                        and silence >= max(1, EARLY_SILENCE_MS // FRAME_DURATION)
                    ):
                        frames_since_partial_probe += 1
                        if frames_since_partial_probe >= max(
                            1, (200 // FRAME_DURATION)
                        ):
                            frames_since_partial_probe = 0
                            txt = run_partial_for_punct()
                            if (
                                txt
                                and txt[-1:] in ".?!"
                                and len(txt) >= 12
                                and word_count(txt) >= 3
                            ):
                                if _finalize_current("early_punct"):
                                    continue

                # ----- Normal finalize after pause -----
                if silence >= END_SILENCE_FRAMES:
                    if hold_left == 0:
                        utter.extend(b"".join(post_ring))
                        post_ring.clear()
                        hold_left = CONTINUE_WINDOW_FR
                    else:
                        hold_left -= 1

                    if hold_left == 0:
                        if not _finalize_current("silence"):
                            _clear_partial_state()
                            _reset_after_finalize()

            except Exception as e:
                print("⚠️ stt_worker loop error:", e)
                continue
    finally:
        if partial_executor:
            partial_executor.shutdown(wait=False)


def run_audio_loop():
    """Wrapper so live_voice_agent can just call stt.run_audio_loop()."""
    src = os.getenv("AUDIO_SOURCE", "mic").lower()
    if src == "audiosocket":
        from core.audio import run_audiosocket_loop

        run_audiosocket_loop(store.STOP)
    else:
        from core.audio import run_audio_loop as _run

        _run(store.STOP)
