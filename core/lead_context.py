# core/lead_context.py
import json, threading, os

_LOCK = threading.Lock()
_CURRENT = {
    "lead": None,
    "campaign": None,
}
_PLAYBOOK = None
_PLAYBOOK_LOCK = threading.Lock()


def _load_playbook():
    global _PLAYBOOK
    if _PLAYBOOK is not None:
        return _PLAYBOOK
    with _PLAYBOOK_LOCK:
        if _PLAYBOOK is not None:
            return _PLAYBOOK
        path = os.getenv("SERVICE_PLAYBOOK_PATH", os.path.join("data", "service_playbook.json"))
        try:
            with open(path, "r", encoding="utf-8") as f:
                _PLAYBOOK = json.load(f)
        except Exception as exc:
            print(f"[LeadContext] Could not load service playbook ({path}): {exc}")
            _PLAYBOOK = {}
    return _PLAYBOOK


def _text_pool(lead_row: dict, campaign_row: dict | None) -> str:
    bits = []
    for val in (lead_row.get("company"), lead_row.get("contact"), lead_row.get("website"),
                lead_row.get("address"), lead_row.get("city"), lead_row.get("state")):
        if isinstance(val, str):
            bits.append(val.lower())
    source_row = lead_row.get("source_row") or {}
    if isinstance(source_row, dict):
        for val in source_row.values():
            if isinstance(val, str):
                bits.append(val.lower())
    if campaign_row:
        for key in ("name", "slug", "notes"):
            val = campaign_row.get(key)
            if isinstance(val, str):
                bits.append(val.lower())
    return " ".join(bits)


def _rank_profile(lead_row: dict, campaign_row: dict, playbook: dict) -> dict:
    profiles = playbook.get("profiles") or []
    if not profiles:
        return {}
    pool = _text_pool(lead_row, campaign_row)
    best = None
    best_score = 0
    for profile in profiles:
        score = 0
        for kw in profile.get("campaign_keywords", []):
            if kw.lower() in pool:
                score += 3
        for kw in profile.get("industry_keywords", []):
            if kw.lower() in pool:
                score += 2
        if score > best_score:
            best_score = score
            best = profile
    if best and best_score > 0:
        return best
    return playbook.get("default_profile") or {}


def _resolve_modules(profile: dict, playbook: dict) -> list[dict]:
    modules = playbook.get("modules") or {}
    resolved = []
    for key in profile.get("primary_modules", []):
        mod = modules.get(key)
        if mod:
            resolved.append(("Primary", mod))
    for key in profile.get("secondary_modules", []):
        mod = modules.get(key)
        if mod:
            resolved.append(("Secondary", mod))
    return resolved


def _render_discovery_block(lead_row: dict, campaign_row: dict, playbook: dict) -> str:
    profile = _rank_profile(lead_row, campaign_row or {}, playbook)
    if not profile:
        return ""
    modules = _resolve_modules(profile, playbook)
    lines = []
    label = profile.get("label", profile.get("id", "Discovery"))
    summary = profile.get("summary")
    lines.append(f"Profile: {label}")
    if summary:
        lines.append(summary)

    hooks = []
    for mod in modules:
        tier, module = mod
        hooks.extend(module.get("hook_openers", []) or [])
    hooks = hooks or profile.get("hook_openers", [])
    if hooks:
        lines.append("")
        lines.append("Attention hooks to open with:")
        for h in hooks[:3]:
            lines.append(f"- {h}")

    prompts = profile.get("discovery_prompts") or playbook.get("default_profile", {}).get("discovery_prompts") or []
    if prompts:
        lines.append("")
        lines.append("Priority questions:")
        for q in prompts:
            lines.append(f"- {q}")

    if modules:
        lines.append("")
        lines.append("Service angles to emphasize:")
        for tier, mod in modules:
            name = mod.get("name", "Service")
            value = mod.get("value_prop")
            lines.append(f"- {tier}: {name}")
            if value:
                lines.append(f"  Value: {value}")
            status_checks = mod.get("status_checks", [])
            if status_checks:
                lines.append("  Confirm they already have:")
                for chk in status_checks:
                    lines.append(f"    * {chk}")
            missing_value = mod.get("value_if_missing")
            if missing_value:
                lines.append(f"  If gap found: {missing_value}")
            for idx, question in enumerate(mod.get("discovery_questions", [])[:2], start=1):
                lines.append(f"  Ask {idx}: {question}")
            proof = mod.get("proof_points", [])
            if proof:
                lines.append(f"  Proof: {proof[0]}")
            upsell = mod.get("upsell_paths", [])
            if upsell:
                lines.append(f"  Upsell: {upsell[0]}")

    return "\n".join(lines).strip()

def set_current_lead(lead_row: dict | None, campaign_row: dict | None = None):
    """Call this right before you originate a call to this lead."""
    global _CURRENT
    with _LOCK:
        _CURRENT = {
            "lead": dict(lead_row or {}),
            "campaign": dict(campaign_row or {}) if campaign_row else None,
        }


def get_current_snapshot():
    with _LOCK:
        return {
            "lead": dict(_CURRENT.get("lead") or {}),
            "campaign": dict(_CURRENT.get("campaign") or {}) if _CURRENT.get("campaign") else None,
        }

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
        state = dict(_CURRENT or {})
        L = dict(state.get("lead") or {})
        campaign_row = state.get("campaign") or {}

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

    playbook = _load_playbook()
    if playbook:
        discovery = _render_discovery_block(raw, campaign_row, playbook)
        if discovery:
            blocks.append("[DISCOVERY GUIDE]\n" + discovery)
    return "\n\n".join(blocks)
