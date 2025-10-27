"""
Deterministic parser: accepts either
- SECTION tags like [INTRO], [QUALIFY], [CLOSE]
- or "INTRO:" inline headers

Produces: {meta:{tone,style}, sections:[{id,title,items:[{type,text}]}], plan:[section_ids], opener}
"""
import re, uuid

def parse(script_text: str) -> dict:
    text = (script_text or "").replace("\r\n","\n").strip()
    sections = []
    plan = []
    opener = ""
    meta = {"tone":"", "style":""}

    # Try [SECTION] blocks first
    blocks = re.split(r"\n\s*(?=\[[A-Z0-9 _-]{3,}\]\s*)", "\n"+text)
    if len(blocks) > 1:
        for blk in blocks:
            m = re.match(r"\s*\[([A-Z0-9 _-]{3,})\]\s*(.*)", blk, flags=re.S)
            if not m: continue
            title = m.group(1).strip().title()
            body  = (m.group(2) or "").strip()
            items = []
            for line in body.split("\n"):
                line = line.strip()
                if not line: continue
                # naive ask/say detection
                typ = "ask" if line.endswith("?") else "say"
                items.append({"type":typ, "text":line})
                if not opener and typ == "say": opener = line
            sid = f"sec_{uuid.uuid4().hex[:6]}"
            sections.append({"id":sid, "title":title, "items":items})
            plan.append(sid)
    else:
        # Fallback to "TITLE:" lines
        cur = None
        for line in text.split("\n"):
            if ":" in line and line.strip().split(":")[0].isupper():
                title = line.split(":",1)[0].strip().title()
                cur = {"id":f"sec_{uuid.uuid4().hex[:6]}", "title":title, "items":[]}
                sections.append(cur); plan.append(cur["id"])
                tail = line.split(":",1)[1].strip()
                if tail:
                    typ = "ask" if tail.endswith("?") else "say"
                    cur["items"].append({"type":typ,"text":tail})
                    if not opener and typ=="say": opener=tail
            elif cur and line.strip():
                typ = "ask" if line.strip().endswith("?") else "say"
                cur["items"].append({"type":typ,"text":line.strip()})
                if not opener and typ=="say": opener=line.strip()

    return {"meta":meta, "sections":sections, "plan":plan, "opener":opener}
