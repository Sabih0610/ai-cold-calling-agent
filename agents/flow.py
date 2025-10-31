"""
Flow object that yields tiny per-turn hints and advances after bot replies.
Also exports flow.json and flow.mmd content helpers.
"""
#agents\flow.py
import json

class Flow:
    def __init__(self, parsed: dict):
        self.meta = parsed.get("meta",{})
        self.sections = parsed.get("sections",[])
        self.plan = parsed.get("plan",[])
        self._idx = 0
        self._opener = parsed.get("opener","")

    def opener(self) -> str:
        return self._opener

    def current_section(self):
        if 0 <= self._idx < len(self.plan):
            sid = self.plan[self._idx]
            for s in self.sections:
                if s["id"] == sid:
                    return s
        return None

    def hint(self) -> str:
        s = self.current_section()
        if not s: return ""
        # Tiny hint: "Stay in <title>; touch on first 1–2 lines"
        lines = [i["text"] for i in s.get("items",[])][:2]
        if not lines: return ""
        return f"Stay in '{s['title']}'. Mention: " + " | ".join(lines)

    def on_user_text(self, _text: str):
        # could choose to advance based on content; keep conservative: do nothing
        pass

    def after_bot_reply(self, _reply: str):
        # Simple advancement: move to next section if we already said something here
        if self._idx < len(self.plan)-1:
            self._idx += 1

    def to_json(self) -> dict:
        return {
            "meta": self.meta,
            "sections": self.sections,
            "plan": self.plan
        }

    def to_mermaid(self) -> str:
        lines = ["flowchart TD"]
        for i, sid in enumerate(self.plan):
            sec = next((s for s in self.sections if s["id"]==sid), None)
            if not sec: continue
            label = sec["title"].replace('"',"'")
            lines.append(f'  {sid}["{label}"]')
            if i < len(self.plan)-1:
                nxt = self.plan[i+1]
                lines.append(f"  {sid} --> {nxt}")
        return "\n".join(lines)
