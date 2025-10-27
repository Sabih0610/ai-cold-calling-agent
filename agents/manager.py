import os, json, uuid, time
from agents import parser
from agents.flow import Flow
from agents import voices as voice_catalog

AGENTS_DIR = os.getenv("AGENTS_DIR", os.path.join("agents","registry"))
os.makedirs(AGENTS_DIR, exist_ok=True)

def _agent_path(agent_id): return os.path.join(AGENTS_DIR, agent_id)

def list():
    out = []
    for name in os.listdir(AGENTS_DIR):
        p = os.path.join(AGENTS_DIR, name, "agent.json")
        if os.path.exists(p):
            with open(p,"r",encoding="utf-8") as f:
                ag = json.load(f); out.append(ag)
    return sorted(out, key=lambda a: a.get("updated_at",""), reverse=True)

def create_from_script(name: str, script_text: str, voice_id: str):
    agent_id = f"agt_{uuid.uuid4().hex[:6]}"
    root = _agent_path(agent_id)
    os.makedirs(root, exist_ok=True)

    # write script
    spath = os.path.join(root, "script.txt")
    with open(spath,"w",encoding="utf-8") as f: f.write(script_text or "")

    # parse + flow objects
    parsed = parser.parse(script_text)
    flow = Flow(parsed)
    flow_json = flow.to_json()
    flow_path = os.path.join(root, "flow.json")
    with open(flow_path,"w",encoding="utf-8") as f: json.dump(flow_json, f, indent=2)

    # mermaid
    mmd = flow.to_mermaid()
    mmd_path = os.path.join(root, "flow.mmd")
    with open(mmd_path,"w",encoding="utf-8") as f: f.write(mmd)

    # context paragraph (short): titles only for now
    titles = [s["title"] for s in parsed.get("sections",[])]
    context = "Script sections: " + " → ".join(titles)

    agent = {
        "id": agent_id,
        "name": name,
        "voice_id": voice_id,
        "context": context,
        "script_path": spath,
        "flow_paths": {"json": flow_path, "mmd": mmd_path},
        "updated_at": int(time.time())
    }
    with open(os.path.join(root, "agent.json"),"w",encoding="utf-8") as f:
        json.dump(agent, f, indent=2)
    return agent_id

def load(agent_id: str) -> dict:
    p = os.path.join(_agent_path(agent_id), "agent.json")
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

def select_active(agent_id: str):
    ag = load(agent_id)
    # rebuild flow from flow.json
    with open(ag["flow_paths"]["json"],"r",encoding="utf-8") as f:
        flow = Flow(json.load(f))
    voice_entry = voice_catalog.get(ag["voice_id"])
    return ag, flow, voice_entry
