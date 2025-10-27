# ===============================================================
# services/name_detect.py
# Simple helper to extract or detect name from text input.
# ===============================================================
import re

COMMON_NAME_PATTERNS = [
    r"\bmy name is ([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
    r"\bi am ([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
    r"\bthis is ([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
    r"\b(?:it's|its) ([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
]

def detect_name_from_text(text: str):
    """Tries to extract a person’s name from conversation text."""
    if not text:
        return None
    t = text.strip()
    for pat in COMMON_NAME_PATTERNS:
        m = re.search(pat, t, flags=re.I)
        if m:
            name = m.group(1).strip().title()
            return name
    # fallback: if text is short and capitalized, maybe it's a name
    if len(t.split()) <= 2 and t.istitle():
        return t
    return None
