#agents\voices.py
import os, json

CATALOG_PATH = os.path.join("agents","voices","catalog.json")
os.makedirs(os.path.dirname(CATALOG_PATH), exist_ok=True)
if not os.path.exists(CATALOG_PATH):
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump([
            { "id": "female_hfc", "label": "Female HFC Medium",
              "model": os.getenv("PIPER_FEMALE_MODEL", ""),
              "config": os.getenv("PIPER_FEMALE_CONFIG", "") },
            { "id": "male_bryce", "label": "Male Bryce Medium",
              "model": os.getenv("PIPER_MALE_MODEL", ""),
              "config": os.getenv("PIPER_MALE_CONFIG", "") }
        ], f, indent=2)

def list_voices():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get(voice_id: str) -> dict | None:
    for v in list_voices():
        if v.get("id") == voice_id:
            return v
    return None
