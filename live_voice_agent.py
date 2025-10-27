"""
Tiny runner that boots engines, wires modules, and exposes a Runner singleton
(other modules or api.py can import and control it).
"""

import os, time, threading, signal
from dotenv import load_dotenv

# Load env first so submodules see flags
load_dotenv()

from core import store, stt, tts, llm, turn
from agents import manager as agent_manager, voices as voice_catalog

# ----- Optional agent providers (default to no-op)
def _empty(): return ""
_context_provider = _empty
_hint_provider = _empty

def set_agent_providers(context_provider, hint_provider):
    global _context_provider, _hint_provider
    _context_provider = context_provider or _empty
    _hint_provider = hint_provider or _empty
    llm.set_providers(_context_provider, _hint_provider)

def _speak(text, user_text_for_name_detect=None):
    tts.speak(text, user_text_for_name_detect=user_text_for_name_detect)

class Runner:
    def __init__(self):
        self._threads = []
        self._running = False
        self._active_agent = None
        self._flow = None
        self._voice = None
        self._model = None

    @property
    def running(self): return self._running
    @property
    def active_agent(self): return self._active_agent

    def start(self, agent_id: str | None = None, use_script_opener: bool = True):
        if self._running:
            # If already running, just switch the agent/voice (hot swap)
            if agent_id:
                self.switch_agent(agent_id)
            return

        # 1) IO/Log worker
        t_io = threading.Thread(target=store.io_worker, daemon=True)
        t_io.start(); self._threads.append(t_io)

        # 2) TTS engine now (so we can speak opener)
        tts.init_engine()   # starts Piper with env-default voice

        # 3) Optional: activate an agent first (so TTS can switch voice)
        if agent_id:
            self.switch_agent(agent_id)
        else:
            # no agent → use empty providers
            set_agent_providers(None, None)

        # 4) Build STT model once
        self._model = stt.make_model()

        # 5) Start STT / LLM / Heartbeat
        t_stt = threading.Thread(target=stt.stt_worker, args=(self._model,), daemon=True)
        t_llm = threading.Thread(target=llm.llm_worker, daemon=True)
        t_hb  = threading.Thread(target=turn.heartbeat_worker, daemon=True)
        t_stt.start(); t_llm.start(); t_hb.start()
        self._threads += [t_stt, t_llm, t_hb]

        # 6) Warm LLM (non-blocking)
        threading.Thread(target=llm.warmup_llm, daemon=True).start()

        # 7) Small delay then speak opener
        time.sleep(0.2)
        if self._flow and use_script_opener:
            opener = self._flow.opener() or llm.default_opener()
        else:
            opener = llm.default_opener()
        store.conv.append("assistant", opener); store.enqueue_turn("assistant", opener)
        _speak(opener)

        # 8) Audio loop (in its own thread so API can control)
        t_audio = threading.Thread(target=stt.run_audio_loop, daemon=True)
        t_audio.start(); self._threads.append(t_audio)

        self._running = True

    def stop(self, reason="manual"):
        if not self._running: return
        store.set_stop(reason)
        try:
            tts.tts_shutdown()
        except Exception:
            pass
        try:
            store.close_session()
            store.conv.clear()
            # let IO drain
            deadline = time.time() + 2.0
            while time.time() < deadline and not store.io_queue.empty():
                time.sleep(0.05)
        except Exception:
            pass
        self._running = False

    def switch_agent(self, agent_id: str):
        """
        Hot-switch:
        - Load agent & flow
        - Restart TTS with agent voice
        - Install LLM providers (context once, hint per turn)
        """
        ag, flow, voice_entry = agent_manager.select_active(agent_id)
        self._active_agent = ag
        self._flow = flow
        self._voice = voice_entry
        if voice_entry:
            tts.use_voice(voice_entry)  # hot swap Piper
        # providers
        set_agent_providers(lambda: (ag.get("context") or ""), flow.hint if flow else None)

# Singleton
runner = Runner()

# Signals for CLI usage
def _handle_signal(signum, frame):
    store.set_stop(f"signal:{signum}")
for s in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
    try: signal.signal(s, _handle_signal)
    except Exception: pass

if __name__ == "__main__":
    # CLI usage: python -m live_voice_agent  (uses ACTIVE_AGENT_ID if set)
    agent_id = os.getenv("ACTIVE_AGENT_ID") or None
    runner.start(agent_id=agent_id, use_script_opener=True)
    try:
        while not store.STOP.is_set():
            time.sleep(0.2)
    finally:
        runner.stop("main_exit")
