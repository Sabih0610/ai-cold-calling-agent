import os, json, asyncio, time
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from starlette.middleware.cors import CORSMiddleware
from typing import Optional

from live_voice_agent import runner
from agents import manager as agent_manager, voices as voice_catalog
from core import store, tts

API_KEY = os.getenv("API_KEY", "")
FRONT_ORIGIN = os.getenv("API_CORS_ORIGINS","*")

app = FastAPI(title="CloumenAI Voice Agent API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONT_ORIGIN] if FRONT_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def require_auth(key: Optional[str] = None):
    # On FastAPI, you'd normally use Header(...); keep simple:
    # Expect clients to send ?key=... or X-API-Key header
    from fastapi import Request
    async def _dep(request: Request):
        hdr = request.headers.get("authorization","")
        token = ""
        if hdr.lower().startswith("bearer "):
            token = hdr.split(" ",1)[1].strip()
        elif "x-api-key" in request.headers:
            token = request.headers.get("x-api-key","")
        else:
            token = request.query_params.get("key","")
        if API_KEY and token != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")
    return _dep

@app.get("/health")
async def health():
    return {"status":"ok", "running": runner.running, "agent": runner.active_agent["id"] if runner.active_agent else None}

# ---- Voices
@app.get("/voices", dependencies=[Depends(require_auth())])
async def list_voices():
    v = voice_catalog.list_voices()
    # return only id + label by default
    return [{"id":i["id"], "label":i["label"]} for i in v]

@app.post("/tts/preview", dependencies=[Depends(require_auth())])
async def tts_preview(voice_id: str = Form(...), text: str = Form(...)):
    import subprocess, tempfile, pathlib, shutil, json
    # Use Piper to synthesize to WAV file without touching live output stream.
    voice = voice_catalog.get(voice_id)
    if not voice:
        raise HTTPException(400, "voice_id not found")

    model = voice["model"]; config = voice["config"]
    piper_exe = os.getenv("PIPER_EXE","piper")
    if not shutil.which(piper_exe):
        raise HTTPException(500, "Piper not installed on PATH")
    # Write text to temp file
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tf:
        tf.write(text)
        text_path = tf.name
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    args = [piper_exe, "--model", model, "--config", config, "--input_text", text_path, "--output_file", wav_path]
    if os.getenv("PIPER_CUDA","0") not in ("0","false","False"):
        args.append("--cuda")
    try:
        subprocess.check_call(args)
        def _send():
            with open(wav_path, "rb") as f:
                yield from f
        return StreamingResponse(_send(), media_type="audio/wav")
    finally:
        try: os.unlink(text_path)
        except Exception: pass
        # leave wav for stream until consumed (auto-deleted by temp dirs on reboot)

# ---- Agents CRUD
@app.get("/agents", dependencies=[Depends(require_auth())])
async def list_agents():
    return agent_manager.list()

@app.post("/agents", dependencies=[Depends(require_auth())])
async def create_agent(name: str = Form(...), voice_id: str = Form(...), script_text: str = Form(default=""), script_file: UploadFile | None = File(default=None)):
    text = script_text
    if script_file:
        text = (await script_file.read()).decode("utf-8", errors="ignore")
    agent_id = agent_manager.create_from_script(name, text or "", voice_id)
    return {"agent_id": agent_id, "flow_preview_url": f"/agents/{agent_id}/flow.mmd"}

@app.get("/agents/{agent_id}", dependencies=[Depends(require_auth())])
async def get_agent(agent_id: str):
    return agent_manager.load(agent_id)

@app.get("/agents/{agent_id}/flow", dependencies=[Depends(require_auth())])
async def get_agent_flow(agent_id: str):
    ag = agent_manager.load(agent_id)
    with open(ag["flow_paths"]["json"],"r",encoding="utf-8") as f:
        return json.load(f)

@app.get("/agents/{agent_id}/flow.mmd", dependencies=[Depends(require_auth())], response_class=PlainTextResponse)
async def get_agent_mmd(agent_id: str):
    ag = agent_manager.load(agent_id)
    with open(ag["flow_paths"]["mmd"],"r",encoding="utf-8") as f:
        return f.read()

@app.post("/agents/{agent_id}/activate", dependencies=[Depends(require_auth())])
async def activate_agent(agent_id: str):
    runner.switch_agent(agent_id)
    return {"ok": True, "active_agent_id": agent_id}

@app.post("/agents/{agent_id}/deactivate", dependencies=[Depends(require_auth())])
async def deactivate_agent(agent_id: str):
    # switch to no-agent mode → providers empty
    from live_voice_agent import set_agent_providers
    set_agent_providers(None, None)
    return {"ok": True}

# ---- Run control
@app.post("/run/start", dependencies=[Depends(require_auth())])
async def run_start(agent_id: str | None = None):
    runner.start(agent_id=agent_id, use_script_opener=True)
    return {"running": True, "agent": runner.active_agent["id"] if runner.active_agent else None}

@app.post("/run/stop", dependencies=[Depends(require_auth())])
async def run_stop():
    runner.stop("api_stop")
    return {"running": False}

# ---- Live events (SSE)
@app.get("/events", dependencies=[Depends(require_auth())])
async def events():
    async def event_gen():
        # text/event-stream
        while True:
            try:
                ev = store.events.get(timeout=1.0)
                yield "event: message\n"
                yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
            except Exception:
                # send heartbeat to keep connection alive
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.01)
    return StreamingResponse(event_gen(), media_type="text/event-stream")
