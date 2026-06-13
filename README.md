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

## 🚀 Quickstart

### Clone & Install

```bash
git clone https://github.com/your-username/ai-voice-assistant.git
cd ai-voice-assistant
pip install -r requirements.txt

Run the App: streamlit run app.py

### Usage Guide
💬 Text Chat
Type your message and press Send.

🎙️ Voice Input
Record → Stop → Transcribe & Send (Whisper STT auto-transcribes).

🖼️ Image Input
Upload an image → ask a question → vision LLM describes and answers.

📄 PDF / RAG
Upload a PDF → it's indexed into ChromaDB → ask questions grounded in the document.

