# 🤖 Multi-modal AI Voice Assistant

> A production-ready, multi-modal AI assistant supporting **text, voice, and image** inputs
> with **RAG**, **tool calling**, **streaming TTS**, and support for multiple LLM backends.

**Author:** Vu Tuan Hung

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Multi-LLM Support** | DeepSeek, OpenAI GPT-4, Anthropic Claude, Google Gemini, Local (Qwen via llama.cpp) |
| 📄 **RAG Pipeline** | Upload PDFs → auto-indexed via ChromaDB + HuggingFace Embeddings → semantic retrieval |
| 🎙️ **Voice I/O** | Whisper-based STT + Kokoro neural TTS with real-time audio streaming |
| 🖼️ **Vision** | Multi-modal image understanding (screenshot, webcam, file upload) |
| 🔧 **Tool Calling** | DuckDuckGo search, clipboard extraction, screenshot/webcam context capture |
| 🔀 **Intent Router** | Auto-routes queries to RAG or general LLM based on keyword + UI signals |
| 💬 **Streaming UI** | Live token streaming with animated audio wave in Streamlit chat interface |

---

## Overview
<img width="2490" height="1197" alt="image" src="https://github.com/user-attachments/assets/72dac57b-12be-42db-9add-fc1aeb9b3bc5" />

---

## 🛠️ Tech Stack

- **LLM Backends:** OpenAI, Anthropic, Google Gemini, DeepSeek, **Qwen (llama.cpp / Ollama)**
- **RAG:** LangChain + **ChromaDB** + `sentence-transformers/all-MiniLM-L6-v2`
- **STT:** OpenAI **Whisper** (via `faster-whisper`)
- **TTS:** **Kokoro** neural TTS (streaming)
- **Vision:** Multi-modal image-to-text via vision-capable LLMs
- **Web UI:** **Streamlit** with custom CSS chat bubbles & live streaming
- **Tool Calling:** DuckDuckGo Search, Clipboard, Screenshot, Webcam
- **Orchestration:** Custom `ToolLoop` + `ConversationContext` with memory

---

## 🔍 RAG Pipeline Detail

- **Ingest:** PDFs uploaded via UI → saved to `RAG_DOCUMENTS_DIR`
- **Chunk:** Split with configurable `chunk_size` / `chunk_overlap`
- **Embed:** `sentence-transformers/all-MiniLM-L6-v2` (CPU-friendly)
- **Store:** Persisted in **ChromaDB** vector store
- **Retrieve:** **Top-K** similarity search at query time
- **Generate:** **LLM** answers grounded strictly on retrieved chunks with source citation

---

## 🚀 Quickstart

### Option A — Local (No Docker)

**1. Clone & Install**
```bash
git clone https://github.com/vutuanhungkkk/Mmva-me.git
cd Mmva-me
pip install -r requirements.txt
```

**2. Run the App**
```bash
streamlit run app.py
```

---

### Option B — Docker (Recommended) 🐳

> ✅ No dependency headaches — everything is packaged inside the container.

#### System Requirements

| Requirement | Minimum |
|---|---|
| OS | Windows 10/11 or Ubuntu 20.04+ |
| GPU | NVIDIA GPU (VRAM ≥ 6GB) |
| CUDA | 12.6+ |
| RAM | ≥ 16GB |
| Docker | Docker Desktop (Win/Mac) or Docker Engine (Linux) |
| NVIDIA Driver | ≥ 530 |

#### Step 1 — Install Prerequisites

- **Docker Desktop:** https://www.docker.com/products/docker-desktop
- **NVIDIA Container Toolkit** (for GPU support):
```bash
# Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Step 2 — Clone the Repository
```bash
git clone https://github.com/vutuanhungkkk/Mmva-me.git
cd ai-voice-assistant
```

#### Step 3 — Add Your Model Files

Create a `models/` folder and place your `.gguf` model file inside:
```
models/
└── your-model.gguf     ← Download from HuggingFace
```

> 📥 Recommended: [Qwen2.5 GGUF on HuggingFace](https://huggingface.co/)  
> ⚠️ Model files are **not included** in the repo (too large). You must download them separately.

#### Step 4 — Configure API Keys

```bash
# Copy the example env file
cp .env.example .env
```

Open `.env` and fill in your keys:
```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx
GOOGLE_API_KEY=xxxxxxxxxxxx
```

> 💡 API keys can also be entered directly in the sidebar UI at runtime.

#### Step 5 — Build & Run

```bash
# Build the Docker image (first time: ~10–15 minutes)
docker compose up --build -d

# View logs
docker compose logs -f
```

#### Step 6 — Open in Browser

```
http://localhost:8501
```

#### Useful Docker Commands

```bash
# Stop the container
docker compose down

# Restart after code changes
docker compose restart

# Rebuild after dependency changes
docker compose build --no-cache && docker compose up -d

# Check running containers
docker compose ps
```

---

## 🎮 Usage Guide

### 💬 Text Chat
Type your message in the input box and press **Send**.

### 🎙️ Voice Input
Click **Start Recording** → speak → click **Stop** → click **Transcribe & Send**.  
Whisper STT automatically transcribes your speech.

### 🖼️ Image Input
Upload an image (JPG / PNG / WebP) → type a question (optional) → click **Send Image**.  
The vision LLM will describe and answer questions about the image.

### 📄 PDF / RAG
Upload a PDF → it is automatically indexed into ChromaDB → ask questions grounded in the document.

---

## 📁 Project Structure

```
ai-voice-assistant/
├── app.py                  ← Streamlit UI
├── assistant/
│   ├── core.py             ← VoiceAssistant orchestration
│   ├── speech.py           ← Whisper STT
│   ├── config.py           ← Environment & paths
│   └── ...
├── models/                 ← Place .gguf model files here (not in repo)
├── documents/              ← PDF upload directory
├── vector_db/              ← ChromaDB persistent store
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example            ← Copy to .env and fill in your API keys
```

---
