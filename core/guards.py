#core\guards.py
import os, re, time
from typing import Optional
from core import store
from core.tts import speak
from core.utils import normalize_text
from rapidfuzz import fuzz

PRICE_GUARD_COOLDOWN_SEC = float(os.getenv("PRICE_GUARD_COOLDOWN_SEC", "12"))
_LAST_PRICE_GUARD_TS = 0.0
PRICE_PATTERNS = [
    r"\bprice(s)?\b", r"\bpricing\b", r"\bcost(s)?\b", r"\bcharge(d|s)?\b",
    r"\brate(s)?\b", r"\bbudget(s)?\b", r"\bquote(s|d|ation)?\b",
    r"\bestimate(s|d)?\b", r"\bhow\s+much\b", r"\bfee(s)?\b",
    r"\bper\s+hour\b", r"\bper\s+project\b",
]
_PRICE_REGEXES = [re.compile(p, re.I) for p in PRICE_PATTERNS]

def price_guard(user_text: str) -> Optional[str]:
    global _LAST_PRICE_GUARD_TS
    t = (user_text or "").lower()
    now = time.time()
    if (now - _LAST_PRICE_GUARD_TS) < PRICE_GUARD_COOLDOWN_SEC:
        return None
    if not any(rx.search(t) for rx in _PRICE_REGEXES):
        return None
    reply = ("We tailor pricing to scope. "
             "If we come back with the most competitive price and a value-for-money outcome, "
             "would you be open to exploring it?")
    store.conv.append("assistant", reply); store.enqueue_turn("assistant", reply)
    store.enqueue_learn_async("pricing_guard", reply)
    speak(reply, user_text_for_name_detect=user_text)
    _LAST_PRICE_GUARD_TS = now
    return reply

LOCATION_KEYWORDS = {"where are you calling", "where are you based", "location", "from where", "which country", "which city"}
def location_guard(user_text: str) -> Optional[str]:
    t = (user_text or "").lower()
    if any(k in t for k in LOCATION_KEYWORDS):
        reply = ("We operate globally with engineering across timezones. "
                 "Would mornings or afternoons suit you better for a quick intro?")
        store.conv.append("assistant", reply); store.enqueue_turn("assistant", reply)
        speak(reply, user_text_for_name_detect=user_text)
        return reply
    return None

BUSINESS_PITCHES = {
    "ecommerce": "For e-commerce, we add AI chat for conversion and automate support. We also build dashboards for ROAS and inventory. Should we explore a quick win?",
    "restaurant": "For restaurants, we automate reservations and feedback, and improve local search. Would that help your bookings?",
    "clinic": "For clinics, we automate intake and reminders, plus clear reporting for patient flow. Is reducing no-shows a priority?",
    "salon": "For salons, we streamline booking and follow-ups, and boost local visibility. Want to try a quick tactic to lift repeat visits?",
    "real estate": "For real estate, we qualify leads with AI and automate follow-ups. Are you focused on lead quality or volume?",
    "logistics": "For logistics, we automate ops updates and build live dashboards. Which bottleneck slows your team most right now?",
    "retail": "For retail, we improve site speed and conversion and automate customer queries. Is conversion or retention more urgent?",
    "saas": "For SaaS, we speed up onboarding and add AI ticket deflection. Should we look at support load or trial conversion first?",
    "education": "For education, we automate admissions queries and build progress dashboards. Are you aiming for more enrollments?",
    "fitness": "For fitness, we streamline signups and reminders and improve discovery. Is churn reduction on your radar?",
    "legal": "For legal firms, we triage intake with AI and improve site speed/SEO. Are you targeting better-qualified leads?",
    "hotel": "For hotels, we improve direct booking UX and local search. Would lifting direct bookings help margins?",
    "construction": "For construction, we streamline lead intake and progress updates. Is visibility into costs and timelines a pain?",
}
ECOM_HINTS = ("ecommerce","e-commerce","e commerce","online store","marketplace","shopify","amazon","ebay","etsy","woocommerce")
FOOTWEAR_HINTS = ("shoe","shoes","footwear","sneaker","sneakers","boot","boots")

def pitch_guard(user_text: str) -> Optional[str]:
    t = (user_text or "").lower()
    t_norm = re.sub(r"[-_/]+", " ", t)
    is_ecom = any(h in t_norm for h in ECOM_HINTS)
    is_footwear = any(h in t_norm for h in FOOTWEAR_HINTS)
    if is_ecom or is_footwear:
        reply = BUSINESS_PITCHES["ecommerce"]
        store.conv.append("assistant", reply); store.enqueue_turn("assistant", reply)
        store.enqueue_learn_async("pitch_ecommerce", reply)
        speak(reply, user_text_for_name_detect=user_text)
        return reply
    for kw, msg in BUSINESS_PITCHES.items():
        if kw in t_norm:
            reply = msg
            store.conv.append("assistant", reply); store.enqueue_turn("assistant", reply)
            store.enqueue_learn_async(f"pitch_{kw}", reply)
            speak(reply, user_text_for_name_detect=user_text)
            return reply
    return None
