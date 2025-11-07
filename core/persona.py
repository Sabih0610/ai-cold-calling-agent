import json
import os
import random
import threading
from typing import Any, Dict, Tuple

PROFILE_PATH = os.getenv(
    "AGENT_PROFILE_PATH",
    os.path.join(os.getcwd(), "data", "agent_profile.json"),
)

_LOCK = threading.Lock()
_PROFILE: Dict[str, Any] | None = None
_STAGE_MAP: Dict[str, Dict[str, Any]] | None = None


def _load() -> Dict[str, Any]:
    global _PROFILE
    if _PROFILE is not None:
        return _PROFILE
    with _LOCK:
        if _PROFILE is not None:
            return _PROFILE
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                _PROFILE = json.load(f)
        except FileNotFoundError:
            _PROFILE = {}
    return _PROFILE


def _stages() -> Dict[str, Dict[str, Any]]:
    global _STAGE_MAP
    if _STAGE_MAP is not None:
        return _STAGE_MAP
    data = _load()
    stage_map: Dict[str, Dict[str, Any]] = {}
    flow = (data.get("flow") or {}).get("stages") or []
    for stage in flow:
        if stage.get("id"):
            stage_map[stage["id"]] = stage
    _STAGE_MAP = stage_map
    return stage_map


def flow_plan() -> list[str]:
    data = _load()
    return (data.get("flow") or {}).get("plan") or []


def stage_for_turn(stage_idx: int) -> Tuple[str, Dict[str, Any]]:
    plan = flow_plan()
    if not plan:
        return "", {}
    idx = min(max(stage_idx, 0), len(plan) - 1)
    stage_id = plan[idx]
    return stage_id, _stages().get(stage_id) or {}


def stage_hint(stage_id: str) -> str:
    info = _stages().get(stage_id) or {}
    if not info:
        return ""
    lines = [f"Current stage: {info.get('title', stage_id)}"]
    goals = info.get("goals") or info.get("principles")
    if goals:
        lines.append("Goals for this stage:")
        lines.extend(f"- {g}" for g in goals)
    tactics = info.get("tactics") or info.get("behaviors")
    if tactics:
        lines.append("Keep in mind:")
        lines.extend(f"- {t}" for t in tactics)
    trigger = info.get("trigger")
    if trigger:
        lines.append(f"Transition trigger: {trigger}")
    return "\n".join(lines)


def stage_question(stage_id: str) -> str:
    info = _stages().get(stage_id) or {}
    qbank = info.get("question_bank") or info.get("questions") or []
    if not qbank:
        return ""
    return random.choice(qbank)


def persona_block() -> str:
    data = _load()
    if not data:
        return ""
    lines: list[str] = []
    meta = data.get("meta") or {}
    persona = data.get("persona") or {}
    objectives = data.get("objectives") or []
    behaviors = (data.get("behaviors") or {})
    guardrails = (data.get("guardrails") or {})

    if meta:
        lines.append(
            f"You are {persona.get('name', meta.get('agent_name','an agent'))} representing {meta.get('brand','Cloumen')}."
        )
    if objectives:
        lines.append("Objectives:")
        lines.extend(f"- {obj}" for obj in objectives)
    if persona:
        lines.append("Tone & persona directives:")
        for key, val in persona.items():
            if isinstance(val, str):
                lines.append(f"- {key.replace('_',' ').title()}: {val}")
    if behaviors:
        lines.append("Behavior expectations:")
        for key, val in behaviors.items():
            lines.append(f"- {key.replace('_',' ').title()}: {val}")
    if guardrails:
        lines.append("Guardrails:")
        for key, val in guardrails.items():
            lines.append(f"- {key.replace('_',' ').title()}: {val}")
    return "\n".join(lines).strip()


def _clean(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    return str(val)


def _first_nonempty(*vals: Any) -> str:
    for v in vals:
        s = _clean(v)
        if s:
            return s
    return ""


def _service_hint(lead: Dict[str, Any]) -> str:
    clues = []
    src = lead.get("source_row") or {}
    for key in (
        "Service",
        "Category",
        "Industry",
        "Vertical",
        "Primary Service",
        "Focus",
    ):
        val = src.get(key)
        if val:
            clues.append(str(val))
    if lead.get("industry"):
        clues.append(lead["industry"])
    if not clues:
        return ""
    return ", ".join(dict.fromkeys(clues))


def opener_brief(lead: Dict[str, Any] | None) -> str:
    if not lead:
        return ""
    name = _first_nonempty(
        lead.get("contact"),
        " ".join(filter(None, [lead.get("first_name"), lead.get("last_name")])),
    )
    company = _clean(lead.get("company"))
    city = _clean(lead.get("city"))
    state = _clean(lead.get("state"))
    industry = _clean(lead.get("industry"))
    role = _clean(lead.get("role_title") or lead.get("title"))
    website = _clean(lead.get("website"))
    service_hint = _service_hint(lead)

    lines = ["Opener personalization data:"]
    if name:
        lines.append(f"- Name: {name}")
    if company:
        lines.append(f"- Company: {company}")
    if city or state:
        lines.append(f"- Location: {', '.join(filter(None,[city,state]))}")
    if industry:
        lines.append(f"- Industry: {industry}")
    if role:
        lines.append(f"- Role: {role}")
    if website:
        lines.append(f"- Website: {website}")
    if service_hint:
        lines.append(f"- Service clues: {service_hint}")

    lines.append(
        "Instructions: weave at least one detail above into the first sentence, then connect Cloumen's value to their likely workflow before asking an open question."
    )
    return "\n".join(lines)


def lead_fact_snippet(lead: Dict[str, Any] | None) -> str:
    if not lead:
        return ""
    company = _clean(lead.get("company"))
    city = _clean(lead.get("city"))
    state = _clean(lead.get("state"))
    industry = _clean(lead.get("industry"))
    role = _clean(lead.get("role_title") or lead.get("title"))
    snippet_parts = []
    if company:
        snippet_parts.append(company)
    if city or state:
        snippet_parts.append(", ".join(filter(None, [city, state])))
    if industry:
        snippet_parts.append(industry)
    description = " | ".join(part for part in snippet_parts if part)
    if role:
        description = f"{description} — Role: {role}" if description else f"Role: {role}"
    return f"Lead facts: {description}" if description else ""


def template(name: str) -> str:
    data = _load()
    return (data.get("templates") or {}).get(name, "")


__all__ = [
    "persona_block",
    "opener_brief",
    "stage_for_turn",
    "stage_hint",
    "stage_question",
    "template",
    "lead_fact_snippet",
]
