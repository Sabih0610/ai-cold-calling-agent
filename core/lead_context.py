# core/lead_context.py
import json, threading

_LOCK = threading.Lock()
_CURRENT = None

def set_current_lead(lead_row: dict | None):
    """Call this right before you originate a call to this lead."""
    global _CURRENT
    with _LOCK:
        _CURRENT = dict(lead_row or {})

def _mask(s: str | None, keep_tail=4):
    if not s: return ""
    t = "".join(ch for ch in s if ch.isdigit())
    return f"+{t[:-keep_tail].replace(t[:-keep_tail],'*'*max(0,len(t)-keep_tail))}{t[-keep_tail:]}" if t else ""

def build_context(script_context: str = "", flow_titles: list[str] = None) -> str:
    """
    Returns a single string that includes both SCRIPT and LEAD blocks.
    Pass in your agent's short script_context (titles) and optional flow titles.
    """
    with _LOCK:
        L = dict(_CURRENT or {})

    # Human snapshot (avoid reciting raw contact details)
    parts = []
    bn = L.get("business_name") or L.get("company") or ""
    person = " ".join([L.get("first_name") or "", L.get("last_name") or ""]).strip()
    city = L.get("city") or ""
    state = L.get("state") or ""
    tz = L.get("tz") or L.get("tz_name") or ""
    phone_masked = _mask(L.get("phone_e164") or L.get("phone_raw") or "")
    email = L.get("email") or ""

    if bn: parts.append(f"Business: {bn}")
    if person: parts.append(f"Contact: {person}")
    if city or state: parts.append(f"Location: {city}{(', ' + state) if state else ''}")
    if tz: parts.append(f"Local TZ: {tz}")
    if phone_masked: parts.append(f"Phone: {phone_masked}")
    if email: parts.append("Email present (do not read aloud)")

    snapshot = " | ".join([p for p in parts if p])

    # Keep raw JSON for precise grounding (LLM sees it, voice won’t read it)
    # Strip obviously useless fields
    raw = {k: v for k, v in L.items() if v not in (None, "", [])}

    guidance = (
        "Use this to personalize naturally. Never recite phone/email/IDs. "
        "Prefer business and first name. If voicemail, leave a 10–12s message using the business name. "
        "If local time seems late/early, acknowledge politely and offer to schedule. "
        "If status hints 'DNC' or 'BAD_NUMBER', stop and mark appropriately."
    )

    # Optional flow section list (nice but not required)
    flows = " → ".join(flow_titles or [])

    blocks = []
    if script_context or flows:
        blocks.append("[SCRIPT CONTEXT]\n" + ("; ".join([s for s in [script_context, flows] if s]).strip()))
    blocks.append("[LEAD SNAPSHOT]\n" + (snapshot or "No visible fields"))
    blocks.append("[LEAD JSON]\n" + json.dumps(raw, ensure_ascii=False, indent=2))
    blocks.append("[LEAD GUIDANCE]\n" + guidance)
    return "\n\n".join(blocks)
