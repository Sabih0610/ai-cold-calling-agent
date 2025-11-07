import json
import os
import re
import threading
from collections import defaultdict
from functools import lru_cache


_LOCK = threading.Lock()
_DATA = None


def _playbook_path():
    return os.getenv(
        "COLD_CALL_PLAYBOOK_PATH",
        os.path.join(os.getcwd(), "data", "cold_calling_playbook.json"),
    )


def _load():
    global _DATA
    if _DATA is not None:
        return _DATA
    with _LOCK:
        if _DATA is not None:
            return _DATA
        path = _playbook_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                _DATA = json.load(f)
        except FileNotFoundError:
            _DATA = {"guidelines": [], "tips": [], "scripts": {}}
    return _DATA


def guidelines_block() -> str:
    data = _load()
    guidelines = data.get("guidelines") or []
    tips = data.get("tips") or []
    lines = []
    if guidelines:
        lines.append("Core cold-call guidelines:")
        lines.extend(f"- {g}" for g in guidelines)
    if tips:
        lines.append("")
        lines.append("Supporting tips to keep in mind on every call:")
        lines.extend(f"- {t}" for t in tips)
    return "\n".join(lines).strip()


@lru_cache(maxsize=1)
def guidelines_hash() -> str:
    import hashlib

    block = guidelines_block()
    if not block:
        return ""
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


def _format_script(script: str, replacements: dict | None = None) -> str:
    if not script:
        return ""
    tmpl = re.sub(r"\{\{(\w+)\}\}", r"{\1}", script)

    class _SafeDict(defaultdict):
        def __missing__(self, key):
            return f"[{key}]"

    safe = _SafeDict(str)
    if replacements:
        for k, v in replacements.items():
            if v is not None:
                safe[k] = str(v)
    return tmpl.format_map(safe)


def get_script(script_key: str, **replacements) -> str:
    if not script_key:
        return ""
    data = _load()
    script = (data.get("scripts") or {}).get(script_key)
    if not script:
        return ""
    return _format_script(script, replacements)


__all__ = ["guidelines_block", "guidelines_hash", "get_script"]
