# 🤖 AI Cold Calling Agent

An advanced **AI-powered voice agent** that listens, understands, and speaks like a human in real time.  
It combines **Whisper (Speech-to-Text)**, **DeepSeek (LLM)**, and **Piper (Text-to-Speech)** to deliver seamless, interactive voice conversations.  
The system supports **barge-in (interrupt while speaking)**, **emotionally adaptive tone**, and **short-term memory** via Redis.

---

## 🧠 Features

- 🎙️ **Real-time Speech Recognition** using **Faster-Whisper** (GPU accelerated)
- 💬 **DeepSeek Chat Model** for natural, context-aware dialogue (streamed responses)
- 🔊 **Piper Text-to-Speech** with natural human-like voices
- 🧠 **Session Memory and Learning**
  - Saves every session to `logs/users/`
  - Learns and promotes frequent phrases automatically
- 🪶 **Barge-in detection** (user can interrupt mid-speech)
- ⚙️ **Redis conversation memory** for contextual replies
- 🧍 **Automatic name detection** from speech
- 📜 **Structured JSON logging system** (user/session-based)
- ❤️ **Emotionally adaptive voice tone**

---

## 🧩 System Requirements

| Component | Minimum Requirement |
|------------|---------------------|
| **OS** | Windows 10/11 (x64) |
| **GPU** | NVIDIA GTX 1650 Ti or higher |
| **CUDA** | Version 12.4 installed and in PATH |
| **Python** | 3.10 – 3.12 |
| **Redis** | Local or Cloud instance |
| **Internet** | Required for DeepSeek API |

---

## 🛠️ Installation Guide

### 1️⃣ Clone or Download the Project

If not already on your system:
```bash
git clone https://github.com/Sabih0610/ai-cold-calling-agent.git
cd ai-cold-calling-agent
2️⃣ Create a Virtual Environment
Recommended for clean dependency isolation:

bash
Copy code
python -m venv winvenv
winvenv\Scripts\activate
You’ll see (winvenv) in your terminal — this means it’s active.

3️⃣ Install Required Packages
Install everything with:

bash
Copy code
pip install -r requirements.txt
If you don’t have requirements.txt, create one with:

txt
Copy code
numpy
sounddevice
webrtcvad
redis
python-dotenv
openai
rapidfuzz
faster-whisper
or install manually:

bash
Copy code
pip install numpy sounddevice webrtcvad redis python-dotenv openai rapidfuzz faster-whisper
4️⃣ Install and Configure Redis
Option A — Local Redis (Windows)
Use Memurai (Redis-compatible for Windows):
👉 https://www.memurai.com/download

or via WSL:

bash
Copy code
sudo apt install redis-server
sudo service redis-server start
Option B — Redis Cloud
Create a free account at Redis Cloud
Then copy your Redis connection string into .env (example below).

5️⃣ Install Piper TTS
Piper generates natural voices locally.
Download it from Piper Releases

Extract and place piper.exe in your project root:

makefile
Copy code
F:\ai cold calling agent\piper.exe
Download voice models:

Female: en_US-hfc_female-medium.onnx

Male: en_US-bryce-medium.onnx

and place them here:

bash
Copy code
voices/female_en/
voices/male_en/
6️⃣ Create and Configure the .env File
In your project root, create .env and paste this (edit as needed):

env
Copy code
# --- API & Services ---
DEEPSEEK_API_KEY=your_deepseek_api_key_here
REDIS_URL=redis://localhost:6379/0

# --- Whisper (STT) ---
WHISPER_SIZE=small
WHISPER_DEVICE=cuda
LANGUAGE=en

# --- Latency & Turn Detection ---
END_SILENCE_MS=550
CONTINUE_WINDOW_MS=350
MIN_TURN_MS=100
EARLY_SILENCE_MS=100
CALIBRATE_MS=100
ENERGY_FLOOR_MULT=1.1
ENERGY_MIN_RMS=0.0055
START_TRIGGER_MS=100
COALESCE_GRACE_MS=500
COALESCE_MAX_MS=1500
STT_FINALIZE_COOLDOWN_MS=180
EARLY_PUNCT_MIN_WORDS=4
BARGE_IN_TRIGGER_MS=100

# --- Conversation & LLM ---
CONV_TURNS=6
LLM_MAX_TOKENS=96
LLM_DEBOUNCE_WINDOW_MS=800
LLM_RECENT_TTL_SEC=2
EARLY_FINALIZE_PUNCT=1
🚀 Running the Agent
Once setup is done:

bash
Copy code
python live_voice_agent.py
You’ll see:

vbnet
Copy code
✅ Whisper loaded on cuda (float16)!
🎧 Ready! Start speaking… (Ctrl+C to stop)
🌐 Streaming DeepSeek reply…
🤖 CloumenAI: Hi there, how can I help today?
🧱 Folder Structure
bash
Copy code
ai-cold-calling-agent/
│
├── live_voice_agent.py          # Main AI voice agent
├── deepseek_text_agent.py       # Text-only version
├── static_responses.json        # Pre-learned responses
├── learned_phrases.json         # Auto-learned phrases
├── voices/                      # Piper voice models
├── logs/                        # Session & user logs
├── winvenv/ or venv/            # Virtual environment (ignored)
├── .env                         # Environment variables (ignored)
└── .gitignore                   # Git ignore rules
⚡ How It Works
🎙️ Speech Input (STT) → Microphone → Faster-Whisper

🧩 VAD (Voice Activity Detection) → Detects speech start/stop

🧠 Coalescer → Joins partial thoughts into complete sentences

💬 DeepSeek LLM → Generates contextual response (streamed instantly)

🔊 Piper TTS → Speaks response in real time

🚫 Barge-In → Stops TTS if user starts talking

🗂️ Redis Memory → Stores short-term conversational context

🧩 Key Environment Variables
Variable	Purpose
DEEPSEEK_API_KEY	Your DeepSeek API key
REDIS_URL	Redis connection string
WHISPER_SIZE	Whisper model size (tiny, small, etc.)
WHISPER_DEVICE	cuda for GPU or cpu
LANGUAGE	Speech language (default: en)
END_SILENCE_MS	How long silence lasts before finalizing speech
CONTINUE_WINDOW_MS	Grace period to merge pauses
MIN_TURN_MS	Minimum speaking duration for a valid turn
EARLY_SILENCE_MS	Silence threshold before partial commit
CALIBRATE_MS	Calibration window for ambient noise
ENERGY_FLOOR_MULT	Controls speech energy threshold
ENERGY_MIN_RMS	Minimum RMS amplitude to consider speech
START_TRIGGER_MS	Minimum speech duration before activation
COALESCE_GRACE_MS	Extend coalescing window for natural turns
COALESCE_MAX_MS	Max merge time for long sentences
STT_FINALIZE_COOLDOWN_MS	Cooldown between speech finalizations
EARLY_PUNCT_MIN_WORDS	Minimum words before punctuation triggers early stop
BARGE_IN_TRIGGER_MS	Time user must speak to interrupt TTS
LLM_MAX_TOKENS	Max words per LLM reply
CONV_TURNS	Max conversation memory length
LLM_DEBOUNCE_WINDOW_MS	Skip near-duplicate STT results
LLM_RECENT_TTL_SEC	Prevent same query repeat time
EARLY_FINALIZE_PUNCT	Enables early finalization on punctuation

🧠 Common Commands
Purpose	Command
Activate environment	winvenv\Scripts\activate
Run voice agent	python live_voice_agent.py
Stop agent	Ctrl + C
Commit & push code	git add . && git commit -m "update" && git push
Deactivate venv	deactivate

🧰 Troubleshooting
❌ Piper not found

Ensure piper.exe is in the project root or in PATH.

❌ CUDA not detected

Verify CUDA installation:

bash
Copy code
nvcc --version
and reinstall PyTorch GPU wheels if needed:

bash
Copy code
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
❌ Redis connection failed

Start Redis manually or verify your .env Redis URL.

⚠️ Slow response

Lower LLM_MAX_TOKENS or CONV_TURNS in .env.

💬 Whisper fallback to CPU

Ensure your GPU drivers and CUDA version are compatible.

🔐 Security Notes
.env and logs/ are excluded from GitHub via .gitignore.

Never commit or share your .env file publicly.

Keep your DEEPSEEK_API_KEY and Redis credentials secure.

💡 Future Enhancements
Emotion-based voice modulation

CRM or lead-management integration

Multilingual support (EN, NL, DE)

Conversation analytics dashboard

WebSocket real-time browser interface
