#core\guards.py
import os, re, time
from typing import Optional
from core import store
from core.tts import speak
from core.utils import normalize_text
from core import callplaybook
from core import persona
from core import flow_state
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

def _script_guard(script_key: str, user_text: str, replacements: Optional[dict] = None) -> Optional[str]:
    text = callplaybook.get_script(script_key, **(replacements or {}))
    if not text:
        return None
    store.conv.append("assistant", text); store.enqueue_turn("assistant", text)
    speak(text, user_text_for_name_detect=user_text)
    return text

_UNINTERESTED_PAT = re.compile(
    r"(not interested|no thanks|no thank you|not a fit|don't like it|do not like it|leave me alone|go away|no right to|not a good time)",
    re.I,
)
def uninterested_guard(user_text: str) -> Optional[str]:
    if not user_text or not _UNINTERESTED_PAT.search(user_text):
        return None
    return _script_guard("uninterested", user_text, {
        "name": "there",
        "agent": os.getenv("AGENT_NAME", "Emma"),
        "company": "Cloumen",
        "specialty": "AI-powered voice follow-up",
        "value": "keep leads warm between callbacks",
        "feature_list": "live agents, proactive callbacks, and real-time dashboards"
    })

_NO_TIME_PATTERNS = (
    "no time", "not a good time", "bad time", "another call",
    "heading into a meeting", "about to hop on", "running into a meeting",
    "driving", "busy", "swamped"
)
def time_guard(user_text: str) -> Optional[str]:
    t = (user_text or "").lower()
    if not any(p in t for p in _NO_TIME_PATTERNS):
        return None
    return _script_guard("no_time", user_text, {"time": "later today"})

_REFERRAL_PATTERNS = (
    "not a fit", "already have", "already working with", "we're good",
    "do not need", "don't need", "no need", "happy with", "stick with", "maybe later"
)
def referral_guard(user_text: str) -> Optional[str]:
    t = (user_text or "").lower()
    if not any(p in t for p in _REFERRAL_PATTERNS):
        return None
    return _script_guard("referral", user_text, {"name": "there", "pain": "missed follow-ups"})

_GATEKEEPER_PAT = re.compile(r"(who (are )?(you )?(trying|want)(ing)? (to )?(reach|speak)|who is this for|what is this regarding)", re.I)
def gatekeeper_guard(user_text: str) -> Optional[str]:
    if not user_text or not _GATEKEEPER_PAT.search(user_text):
        return None
    return _script_guard("gatekeeper", user_text, {
        "gatekeeper": "",
        "agent": os.getenv("AGENT_NAME", "Emma"),
        "company": "Cloumen",
        "prospect": "the business owner",
        "department": "operations or growth"
    })

_IDENTITY_PAT = re.compile(
    r"(who\s+are\s+you|why\s+(should\s+)?i\s+tell|why\s+do\s+i\s+tell|what('?| i)s\s+in\s+it\s+for\s+you|i\s+don't\s+know\s+you)",
    re.I,
)
def identity_guard(user_text: str) -> Optional[str]:
    if not user_text or not _IDENTITY_PAT.search(user_text):
        return None
    template = persona.template("identity_reassurance")
    stage_id, _ = persona.stage_for_turn(flow_state.current_stage_idx())
    follow_question = persona.stage_question(stage_id or "rapport")
    handled_before = flow_state.identity_handled()
    focus = os.getenv(
        "FOCUS_AREA",
        "marketing, web and app, cloud and DevOps, data and analytics, or automation",
    )
    if not handled_before:
        if template:
            text = template.replace("{{focus_area}}", focus)
        else:
            text = (
                "That is totally fair. I'm Emma with Cloumen. "
                f"We help with {focus}. I only ask a couple quick questions to see if it is worth a follow-up."
            )
        flow_state.mark_identity()
    else:
        text = ("Absolutely understand the caution. I'm here just to see if there's anything worth a quick follow-up later. "
                f"If it's okay, I'd love to know one small thing about how you handle {focus} so I don't make wrong assumptions.")
    if follow_question:
        text = f"{text} {follow_question}"
        flow_state.record_question(follow_question)
    store.conv.append("assistant", text); store.enqueue_turn("assistant", text)
    speak(text, user_text_for_name_detect=user_text)
    return text

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

# Services listing guard (when they ask "what do you do/offer?")
_SERVICES_PAT = re.compile(
    r"(what\s+services|what\s+do\s+you\s+(offer|provide|do)|services\s+do\s+you|tell\s+me\s+about\s+your\s+services|what\s+kind\s+of\s+services|what\s+companies\s+do\s+you\s+serve|who\s+do\s+you\s+work\s+with)",
    re.I,
)
SERVICES_REPLY = (
    "We help in six areas: marketing and growth, web and app builds, cloud and DevOps, "
    "data and analytics, automation and AI agents, and technology consulting. "
    "Which area matters most for you right now?"
)

def services_guard(user_text: str) -> Optional[str]:
    if not user_text or not _SERVICES_PAT.search(user_text):
        return None
    store.conv.append("assistant", SERVICES_REPLY); store.enqueue_turn("assistant", SERVICES_REPLY)
    speak(SERVICES_REPLY, user_text_for_name_detect=user_text)
    return SERVICES_REPLY

# Targeted service explainer when they mention a lane explicitly
def service_explainer_guard(user_text: str) -> Optional[str]:
    if not user_text:
        return None
    t = (user_text or "").lower()
    reply = None
    if "cloud" in t or "migration" in t or "devops" in t:
        reply = ("On cloud and DevOps, we handle planning, infra setup, migrations, automation, and monitoring "
                 "so your team can ship reliably. Want me to walk through what you already have?")
    elif "website" in t or "app" in t or "mobile" in t:
        reply = ("On web and app, we build and modernize sites and apps with speed, UX, and analytics baked in. "
                 "Should we talk about your current site stack and pain points?")
    elif "marketing" in t or "ads" in t or "leads" in t or "growth" in t:
        reply = ("For marketing and growth, we run paid and organic programs with clear tracking from click to lead. "
                 "Do you want help with campaigns, conversion, or reporting?")
    elif "data" in t or "dashboard" in t or "analytics" in t or "report" in t:
        reply = ("On data and analytics, we build pipelines and dashboards so you can see performance in real time. "
                 "Which metrics are hardest for you to see today?")
    elif "automation" in t or "ai" in t or "agent" in t:
        reply = ("For automation and AI, we build voice/chat agents and workflow automation to cut manual work. "
                 "Which process would you automate first?")
    elif "consult" in t or "strategy" in t or "roadmap" in t:
        reply = ("For tech consulting, we set a simple roadmap, prioritize the wins, and bring the delivery team to execute. "
                 "What initiative is stuck right now?")
    if reply:
        store.conv.append("assistant", reply); store.enqueue_turn("assistant", reply)
        speak(reply, user_text_for_name_detect=user_text)
        return reply
    return None

# Abuse / hostile guard: exit respectfully
_ABUSE_PAT = re.compile(r"(idiot|stupid|shut up|hate you|go away|leave me alone)", re.I)
def abuse_guard(user_text: str) -> Optional[str]:
    if not user_text or not _ABUSE_PAT.search(user_text):
        return None
    reply = "Understood. I will let you go now. Thanks for your time."
    store.conv.append("assistant", reply); store.enqueue_turn("assistant", reply)
    speak(reply, user_text_for_name_detect=user_text)
    try:
        store.set_stop("abuse_guard")
    except Exception:
        pass
    return reply

def voicemail_script(name: str = "", trigger: str = "", callback: str = "") -> str:
    return callplaybook.get_script(
        "voicemail",
        name=name or "there",
        trigger=trigger or "your latest update",
        callback=callback or os.getenv("CALLBACK_NUMBER", "+1-555-123-4567"),
    )
