"""
Tiny runner that boots engines, wires modules, and exposes a Runner singleton
(other modules or api.py can import and control it).
"""
#live_voice_agent.py
import os, time, threading, signal, json, queue
from dotenv import load_dotenv

# Load env first so submodules see flags
load_dotenv()

from core import store, stt, tts, llm, turn
from core import lead_context

# Telephony latch & helpers (PBX build). In local mode these fall back.
try:
    from core.audio import audiosocket_ready, prime_tx_silence, WIRE_IS_8K
except Exception:
    audiosocket_ready = None
    WIRE_IS_8K = None
    def prime_tx_silence(ms=0): pass

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

    # ---------------------------------------------
    # Opener loop: say opener on bridge; PRE-ARM STT (mutes while TTS plays)
    # ---------------------------------------------
    def _opener_on_socket_loop(self):
        sink = os.getenv("AUDIO_SINK", "audiosocket").lower()
        src  = os.getenv("AUDIO_SOURCE", "mic").lower()
        if sink != "audiosocket" or src != "audiosocket":
            print("[Opener] Telephony not active (sink/src). Opener loop idle.")
            return

        arm_guard_ms = int(os.getenv("ARM_WAIT_GUARD_MS", "120"))
        print("[Opener] Loop ready. Waiting for calls...")

        while self._running and not store.STOP.is_set():
            # Disarm for the next call
            stt.reset_first_turn()

            # Wait for PBX bridge
            audiosocket_ready.wait()
            if not self._running or store.STOP.is_set():
                break

            # Seed first turn baseline using wire-rate hint
            try:
                seed = 0.0027 if (WIRE_IS_8K is True) else 0.0038
                stt.arm_first_turn(seed_rms=seed)
            except Exception:
                stt.arm_first_turn()

            # Prime TX with a small silence so opener isn't clipped
            try:
                prime_tx_silence(int(os.getenv("ASOCK_PREROLL_MS", "200")))
            except Exception:
                pass

            # Build per-call context at the last moment (read dialer pointer)
            try:
                ctx_dir = os.getenv("CTX_DIR", "/var/tmp/dialer_ctx")
                next_path = os.path.join(ctx_dir, "next.json")
                lead_payload = None
                lead_id = campaign_id = None
                if os.path.exists(next_path):
                    with open(next_path, "r", encoding="utf-8") as f:
                        ptr = json.load(f)
                    context_path = ptr.get("context_path")
                    lead_id = ptr.get("lead_id")
                    campaign_id = ptr.get("campaign_id")
                    if context_path and os.path.exists(context_path):
                        with open(context_path, "r", encoding="utf-8") as f:
                            lead_payload = json.load(f)
                if lead_payload and lead_payload.get("lead_full_raw"):
                    lead_context.set_current_lead(lead_payload.get("lead_full_raw"))
                    # Rotate conversation + session per call
                    try: store.close_session()
                    except Exception: pass
                    try:
                        store.new_conversation(key=f"{campaign_id or 'x'}_{lead_id or 'y'}_{int(time.time())}")
                    except Exception:
                        pass
                    try: store.open_new_session()
                    except Exception: pass

                    # Install providers to include lead context + flow hint
                    agent_ctx = (self._active_agent.get("context") if self._active_agent else "") or ""
                    flow_titles = [s.get("title","") for s in (self._flow.sections if self._flow else [])]
                    def _context_provider():
                        return lead_context.build_context(script_context=agent_ctx, flow_titles=flow_titles)
                    llm.set_providers(_context_provider, (self._flow.hint if self._flow else None))
            except Exception as e:
                print(f"[Opener] Context wiring error: {e}")
            try:
                print("[Opener] Bridge detected. Generating opener via LLM...")
                # STT is already armed; ask LLM for a short opener grounded in lead
                llm.deepseek_stream_and_speak("[START_CALL]")
                print("[Opener] LLM opener sent.")
            except Exception as e:
                print(f"[Opener] ERROR while speaking opener: {e}")
            finally:
                print("[Opener] STT already armed for first user turn.")
                time.sleep(arm_guard_ms / 1000.0)

            # Wait for hangup before next cycle
            while audiosocket_ready.is_set() and self._running and not store.STOP.is_set():
                time.sleep(0.05)

            print("[Opener] Call ended. Waiting for the next call...")
            try:
                tts.tts_interrupt()
            except Exception:
                pass
            try:
                while True:
                    item = llm.llm_queue.get_nowait()
                    llm.llm_queue.task_done()
            except queue.Empty:
                pass
            try:
                llm._first_turn_done = False
            except Exception:
                pass

    def start(self, agent_id: str | None = None, use_script_opener: bool = True):
        if self._running:
            if agent_id:
                self.switch_agent(agent_id)
            return

        # 1) IO/Log worker
        t_io = threading.Thread(target=store.io_worker, daemon=True)
        t_io.start(); self._threads.append(t_io)

        # 2) TTS engine (no speech yet)
        tts.init_engine()

        # 3) Optional: activate an agent first (so TTS can switch voice)
        if agent_id:
            self.switch_agent(agent_id)
        else:
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

        # 7) Audio loop FIRST (so PBX can connect)
        t_audio = threading.Thread(target=stt.run_audio_loop, daemon=True)
        t_audio.start(); self._threads.append(t_audio)

        # Mark running before any opener logic
        self._running = True

        # 8) Opener policy
        if use_script_opener:
            t_op = threading.Thread(target=self._opener_on_socket_loop, daemon=True)
            t_op.start(); self._threads.append(t_op)
        else:
            # Local (non-PBX) mode: small settle, arm/seed, then speak
            time.sleep(0.2)
            stt.arm_first_turn(seed_rms=0.0032)
            # Try to build per-call context if dialer pointer exists, then generate LLM opener
            try:
                ctx_dir = os.getenv("CTX_DIR", "/var/tmp/dialer_ctx")
                next_path = os.path.join(ctx_dir, "next.json")
                if os.path.exists(next_path):
                    with open(next_path, "r", encoding="utf-8") as f:
                        ptr = json.load(f)
                    context_path = ptr.get("context_path")
                    if context_path and os.path.exists(context_path):
                        with open(context_path, "r", encoding="utf-8") as f:
                            lead_payload = json.load(f)
                        if lead_payload.get("lead_full_raw"):
                            lead_context.set_current_lead(lead_payload.get("lead_full_raw"))
                            store.new_conversation(f"local_{int(time.time())}")
                            store.open_new_session()
                            agent_ctx = (self._active_agent.get("context") if self._active_agent else "") or ""
                            flow_titles = [s.get("title","") for s in (self._flow.sections if self._flow else [])]
                            llm.set_providers(lambda: lead_context.build_context(agent_ctx, flow_titles), (self._flow.hint if self._flow else None))
            except Exception:
                pass
            llm.deepseek_stream_and_speak("[START_CALL]")

    def stop(self, reason="manual"):
        if not self._running: return
        store.set_stop(reason)
        try:
            tts.tts_shutdown()
        except Exception:
            pass
        try:
            # Clear any RX frames so STT doesn't print late "User:" lines
            try:
                stt.clear_audio_queue()
            except Exception:
                pass
            store.close_session()
            store.conv.clear()
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
    agent_id = os.getenv("ACTIVE_AGENT_ID") or None
    runner.start(agent_id=agent_id, use_script_opener=True)
    try:
        while not store.STOP.is_set():
            time.sleep(0.2)
    finally:
        runner.stop("main_exit")
