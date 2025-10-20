import os
import json
import hashlib
import redis
from dotenv import load_dotenv
from openai import OpenAI

# =====================================================
# 🔧 Load environment variables
# =====================================================
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

# =====================================================
# 🧠 Initialize DeepSeek (OpenAI-compatible)
# =====================================================
client = OpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=DEEPSEEK_API_KEY
)

# =====================================================
# 🔗 Connect to Redis Cloud (from .env)
# =====================================================
try:
    r = redis.from_url(REDIS_URL, decode_responses=True)
    r.ping()
    print("✅ Connected to Redis Cloud")
except Exception as e:
    print("⚠️ Redis connection failed:", e)
    r = None

# =====================================================
# 💬 Persistent conversational memory
# =====================================================
conversation = [
    {
        "role": "system",
        "content": (
            "You are CloumenAI — an emotionally intelligent and natural conversational assistant "
            "representing Cloumen, a modern tech company. "
            "You speak with warmth, empathy, and confidence — like a human sales professional. "
            "Avoid robotic or scripted phrasing. "
            "Use subtle fillers like 'hmm', 'alright', 'I see', or small pauses ('...'). "
            "Be emotionally responsive: sound inspiring in motivational topics, calm in emotional ones, "
            "and confident in business contexts. "
            "Always keep sentences short, spoken-like, and genuine."
        ),
    }
]

# =====================================================
# ⚙️ Cache Utilities
# =====================================================
def cache_get(key: str):
    """Return cached response if found"""
    if not r:
        return None
    try:
        value = r.get(key)
        if value:
            print("🔁 Cache hit")
            return value
    except Exception as e:
        print("⚠️ Redis get error:", e)
    return None


def cache_set(key: str, value: str, ttl=86400):
    """Store new value in cache (default: 24h)"""
    if not r:
        return
    try:
        r.setex(key, ttl, value)
        print("💾 Cached new response")
    except Exception as e:
        print("⚠️ Redis set error:", e)

# =====================================================
# 🧠 DeepSeek Query Function
# =====================================================
def deepseek_reply(user_input: str):
    """Generates natural, emotional, cached AI replies"""
    # Cache key derived from user input
    cache_key = hashlib.sha256(user_input.encode()).hexdigest()

    # Try to fetch cached reply
    cached = cache_get(cache_key)
    if cached:
        return cached

    print("🚀 Cache miss → Querying DeepSeek...")

    # Append user message to conversation
    conversation.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.95,         # expressive & human
            top_p=0.95,
            presence_penalty=0.6,
            frequency_penalty=0.6,
            max_tokens=400,
            messages=conversation
        )

        reply = response.choices[0].message.content.strip()
        conversation.append({"role": "assistant", "content": reply})

        # Save to Redis
        cache_set(cache_key, reply)
        return reply

    except Exception as e:
        print(f"❌ DeepSeek API Error: {e}")
        return "Sorry, something went wrong while generating my response."

# =====================================================
# 💬 Interactive Mode (CLI Test)
# =====================================================
if __name__ == "__main__":
    print("🤖 CloumenAI (DeepSeek + Redis Cloud)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("🗣 You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Ending session.")
            break

        reply = deepseek_reply(user_input)
        print(f"🧠 CloumenAI: {reply}\n")
