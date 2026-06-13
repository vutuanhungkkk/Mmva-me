"""
Streamlit web interface for the Multi-modal AI Voice Assistant.
Run with: streamlit run app.py
"""

import os
import io
import base64
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
import streamlit as st

# ─────────────────────────────────────────────
# Page config — must be the FIRST Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — light background, dark text + Audio Wave Animation
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F5F7FA;
        color: #1A1A2E;
        font-family: 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #1A1A2E !important;
    }
    .user-bubble {
        background: #4F8EF7;
        color: #FFFFFF;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        margin: 6px 0 6px auto; 
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 2px 6px rgba(79,142,247,0.25);
        display: inline-block;  
        max-width: 80%;        
        word-wrap: break-word; 
        float: right;        
        clear: both;    
    }
    .assistant-bubble {
        background: #FFFFFF;
        color: #1A1A2E;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 18px;
        margin: 6px auto 6px 0;
        font-size: 15px;
        line-height: 1.6;
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
        display: inline-block;
        max-width: 80%;      
        word-wrap: break-word; 
        float: left;   
        clear: both;     
    }
    .role-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin-bottom: 3px;
        text-transform: uppercase;
    }
    .user-label   { color: #4F8EF7; text-align: right; clear: both;}
    .assistant-label { color: #6C757D; }
    
    /* AUDIO WAVE CSS */
    .audio-wave {
        display: inline-flex;
        align-items: center;
        gap: 2px;
        margin-right: 8px;
        height: 14px;
    }
    .audio-wave span {
        display: block;
        width: 3px;
        height: 100%;
        background-color: #4F8EF7;
        animation: wave-animation 1.2s infinite ease-in-out;
    }
    .audio-wave span:nth-child(1) { animation-delay: -1.2s; }
    .audio-wave span:nth-child(2) { animation-delay: -1.1s; }
    .audio-wave span:nth-child(3) { animation-delay: -1.0s; }
    .audio-wave span:nth-child(4) { animation-delay: -0.9s; }
    @keyframes wave-animation {
        0%, 40%, 100% { transform: scaleY(0.3); }
        20% { transform: scaleY(1); }
    }
    
    .stTextInput > div > div > input {
        background: #FFFFFF;
        color: #1A1A2E;
        border: 1.5px solid #C9D6E3;
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 15px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4F8EF7;
        box-shadow: 0 0 0 3px rgba(79,142,247,0.15);
    }
    .stButton > button {
        background: #4F8EF7;
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        padding: 10px 22px;
        font-size: 15px;
        font-weight: 600;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #3A7BD5; }
    .stSelectbox label, .stRadio label { color: #1A1A2E !important; }
    .stAlert { border-radius: 10px; }
    hr { border-color: #E0E0E0; }
    audio { width: 100%; margin-top: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        # List of chat turns:
        # {"role": "user"|"assistant", "content": str, "audio_b64": str|None}
        "messages": [],

        "assistant": None,
        "provider": "deepseek",
        "api_key": "",
        "assistant_ready": False,

        # ── Two-phase processing flags ──
        # Holds the pending user turn while we wait for the LLM
        # Format: {"prompt": str, "image_bytes": bytes|None, "image_b64": str|None}
        "pending_input": None,

        # True while the assistant is generating a response
        "is_processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────
# Provider metadata
# ─────────────────────────────────────────────
PROVIDERS = {
    "deepseek":  {"label": "DeepSeek",          "needs_key": True,  "env": "DEEPSEEK_API_KEY"},
    "openai":    {"label": "OpenAI",             "needs_key": True,  "env": "OPENAI_API_KEY"},
    "anthropic": {"label": "Anthropic Claude",   "needs_key": True,  "env": "ANTHROPIC_API_KEY"},
    "google":    {"label": "Google Gemini",      "needs_key": True,  "env": "GOOGLE_API_KEY"},
    "local":     {"label": "Local (Qwen)", "needs_key": False, "env": None},
}


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🤖 AI Voice Assistant")
        st.markdown("---")

        st.markdown("### 🧠 LLM Provider")
        provider_labels = {k: v["label"] for k, v in PROVIDERS.items()}
        selected_key = st.selectbox(
            "Choose provider",
            options=list(provider_labels.keys()),
            format_func=lambda k: provider_labels[k],
            index=list(provider_labels.keys()).index(st.session_state.provider),
            key="provider_select",
        )
        st.session_state.provider = selected_key

        meta = PROVIDERS[selected_key]
        if meta["needs_key"]:
            existing = os.environ.get(meta["env"], "")
            api_key = st.text_input(
                f"{meta['label']} API Key",
                value=st.session_state.api_key or existing,
                type="password",
                placeholder=f"Paste your {meta['label']} API key…",
                key="api_key_input",
            )
            st.session_state.api_key = api_key
        else:
            st.info("No API key required for local models.")
            st.session_state.api_key = ""

        # TTS Provider di chuyển lên trên nút Apply & Start
        st.markdown("---")
        st.markdown("### 🔊 TTS Provider")
        st.info("Kokoro TTS (default)")
        st.markdown("---")

        if st.button("🚀 Apply & Start Assistant", use_container_width=True):
            _init_assistant()

        if st.session_state.assistant_ready:
            st.success("✅ Assistant is ready!")
        else:
            st.warning("⚠️ Configure and start the assistant.")

        # Thông tin app và Author ở Sidebar
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("**App:** Multi-modal AI Voice Assistant")
        st.markdown("**Author:** Vu Tuan Hung")


# ─────────────────────────────────────────────
# Assistant initialisation
# ─────────────────────────────────────────────
def _init_assistant() -> None:
    provider = st.session_state.provider
    meta = PROVIDERS[provider]

    if meta["needs_key"] and st.session_state.api_key:
        os.environ[meta["env"]] = st.session_state.api_key

    os.environ["LLM_PROVIDER"] = provider
    os.environ["TTS_PROVIDER"] = "kokoro"

    try:
        import importlib
        import assistant.config as cfg
        importlib.reload(cfg)

        from assistant.core import VoiceAssistant
        st.session_state.assistant = VoiceAssistant()
        st.session_state.assistant_ready = True
        st.success("Assistant initialised successfully!")
    except Exception as exc:
        st.session_state.assistant_ready = False
        st.error(f"Failed to initialise assistant: {exc}")


# ─────────────────────────────────────────────
# Chat history renderer
# ─────────────────────────────────────────────
def render_chat_history() -> None:
    for msg in st.session_state.messages:
        role      = msg["role"]
        content   = msg["content"]
        audio_b64 = msg.get("audio_b64")

        if role == "user":
            st.markdown('<p class="role-label user-label">You</p>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="user-bubble">{content}</div>'
                '<div style="clear:both"></div>',
                unsafe_allow_html=True,
            )
            if msg.get("image_bytes"):
                st.image(msg["image_bytes"], width=220)
        else:
            st.markdown('<p class="role-label assistant-label">🤖 Assistant</p>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="assistant-bubble">{content}</div>'
                '<div style="clear:both"></div>',
                unsafe_allow_html=True,
            )
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                st.audio(audio_bytes, format="audio/wav")

# ─────────────────────────────────────────────
# Phase 1 — Queue the user's input, rerun immediately
# ─────────────────────────────────────────────
def _queue_user_input(prompt: str, image_bytes: Optional[bytes] = None) -> None:
    if not st.session_state.assistant_ready or not st.session_state.assistant:
        st.error("Please initialise the assistant first (sidebar → Apply & Start).")
        return

    user_display = prompt or ""
    if image_bytes:
        user_display = f"[Image attached]\n{prompt}" if prompt else "[Image attached]"

    image_b64: Optional[str] = None
    if image_bytes:
        image_b64 = (
            "data:image/jpeg;base64,"
            + base64.b64encode(image_bytes).decode("utf-8")
        )

    st.session_state.messages.append({
        "role":        "user",
        "content":     user_display,
        "image_bytes": image_bytes,   
    })

    st.session_state.pending_input = {
        "prompt":    prompt,
        "image_b64": image_b64,
    }
    st.session_state.is_processing = True

    st.rerun()


# ─────────────────────────────────────────────
# Phase 2 — Call LLM, show live text sync, then play audio
# ─────────────────────────────────────────────
def _process_pending_input() -> None:
    pending = st.session_state.pending_input
    if pending is None:
        return

    prompt    = pending["prompt"]
    image_b64 = pending["image_b64"]
    assistant = st.session_state.assistant

    st.session_state.pending_input = None
    st.session_state.is_processing = False

    # Build an empty container for inserting the streaming UI
    stream_container = st.empty()

    try:
        response_text, audio_b64 = _call_llm_and_capture_audio(
            assistant, prompt, image_b64, stream_container
        )
    except Exception as exc:
        response_text = f"⚠️ Error: {exc}"
        audio_b64     = None
        stream_container.empty()

    # Once stream is completely over, we remove the container 
    # and append it officially to history.
    stream_container.empty()

    st.session_state.messages.append({
        "role":      "assistant",
        "content":   response_text,
        "audio_b64": audio_b64,       
    })

    # Display the static completed final output
    st.rerun()


# ─────────────────────────────────────────────
# LLM call + Live UI Text Sync + Audio capture
# ─────────────────────────────────────────────
def _call_llm_and_capture_audio(
    assistant,
    prompt: str,
    img_context: Optional[str],
    ui_container,  # Receive UI placeholder for realtime streaming
) -> tuple[str, Optional[str]]:
    
    # ── Initial State: Thinking ──
    ui_container.markdown(
        '<p class="role-label assistant-label">🤖 Assistant</p>'
        '<div class="assistant-bubble" style="color:#6C757D; font-style:italic;">🤔 Thinking...</div>'
        '<div style="clear:both"></div>',
        unsafe_allow_html=True
    )

    audio_chunks: list[bytes] = []

    # ── Monkey-patch stream_speak to intercept text & UI ──
    original_stream_speak = assistant.tts_provider.stream_speak

    def capturing_stream_speak(chunks_iter):
        collected: list[str] = []

        def tee(it):
            first_chunk = True
            for chunk in it:
                if first_chunk and chunk.strip():
                    first_chunk = False
                
                collected.append(chunk)
                current_text = "".join(collected)
                
                # Update UI container real-time with speaking animation & parsed text
                wave_html = '<div class="audio-wave"><span></span><span></span><span></span><span></span></div>'
                
                ui_container.markdown(
                    f'<p class="role-label assistant-label">🤖 Assistant</p>'
                    f'<div class="assistant-bubble">'
                    f'<div style="margin-bottom:8px;">{wave_html}</div>'
                    f'<div>{current_text}</div>'
                    f'</div><div style="clear:both"></div>',
                    unsafe_allow_html=True
                )
                yield chunk

        # Play audio through speakers as normal while streaming the UI
        original_stream_speak(tee(chunks_iter))

        # Re-synthesize to bytes for the offline/in-browser historical player
        full_text = "".join(collected)
        if full_text and hasattr(assistant.tts_provider, "synthesize_to_bytes"):
            try:
                wav = assistant.tts_provider.synthesize_to_bytes(full_text)
                if wav:
                    audio_chunks.append(wav)
            except Exception:
                pass  

    assistant.tts_provider.stream_speak = capturing_stream_speak
    try:
        response_text = assistant.run_once(
            prompt=prompt,
            image_b64=img_context,
        )
    finally:
        assistant.tts_provider.stream_speak = original_stream_speak

    audio_b64: Optional[str] = None
    if audio_chunks:
        combined  = b"".join(audio_chunks)
        audio_b64 = base64.b64encode(combined).decode("utf-8")

    return response_text, audio_b64


# ─────────────────────────────────────────────
# Input mode: Text
# ─────────────────────────────────────────────
def render_text_input() -> None:
    disabled = st.session_state.is_processing
    st.markdown(
        """
        <style>
        [data-testid="stTextArea"] textarea {
            background: #FFFFFF;
            color: #1A1A2E;
            border: 1.5px solid #C9D6E3;
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 15px;
            font-family: 'Segoe UI', sans-serif;
            min-height: 46px !important;
            max-height: 300px !important;
            overflow-y: auto !important;
            resize: none !important;
            line-height: 1.6;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        [data-testid="stTextArea"] textarea:focus {
            border-color: #4F8EF7 !important;
            box-shadow: 0 0 0 3px rgba(79,142,247,0.15) !important;
        }
        </style>
        <script>
        window.addEventListener('load', function() {
            function autoResize(el) {
                el.style.height = 'auto';
                el.style.height = Math.min(el.scrollHeight, 300) + 'px';
            }
            function attachListeners() {
                const textareas = document.querySelectorAll(
                    '[data-testid="stTextArea"] textarea'
                );
                textareas.forEach(function(ta) {
                    if (!ta.dataset.autoResize) {
                        ta.dataset.autoResize = 'true';
                        ta.addEventListener('input', function() {
                            autoResize(this);
                        });
                    }
                });
            }
            attachListeners();
            const observer = new MutationObserver(attachListeners);
            observer.observe(document.body, { childList: true, subtree: true });
        });
        </script>
        """,
        unsafe_allow_html=True,
    )

    with st.form(key="text_form", clear_on_submit=True):
        col1, col2 = st.columns([8, 1])
        with col1:
            user_text = st.text_area(  
                "Message",
                placeholder="Type your message here…" if not disabled
                            else "Waiting for response…",
                label_visibility="collapsed",
                disabled=disabled,
                height=46,              
            )
        with col2:
            st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Send", disabled=disabled)

    if submitted and user_text and user_text.strip():
        _queue_user_input(user_text.strip())


# ─────────────────────────────────────────────
# Audio helper utilities
# ─────────────────────────────────────────────
def _detect_audio_format(raw_bytes: bytes) -> str:
    if len(raw_bytes) < 4:
        return "unknown"
    if raw_bytes[:4] == b"RIFF":
        return "wav"
    if raw_bytes[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if raw_bytes[:4] == b"OggS":
        return "ogg"
    return "unknown"


def _read_wav_with_soundfile(wav_bytes: bytes):
    try:
        buf = io.BytesIO(wav_bytes)
        audio_data, sample_rate = sf.read(buf, dtype="float32")
        return audio_data, sample_rate
    except Exception as e:
        print(f"[Audio] soundfile read failed: {e}")
        return None, None


def _write_wav_with_soundfile(audio_data: np.ndarray, sample_rate: int) -> bytes:
    if audio_data.ndim == 2:
        audio_data = audio_data.mean(axis=1)
    if sample_rate != 16000:
        duration = len(audio_data) / sample_rate
        target_length = int(duration * 16000)
        audio_data = np.interp(
            np.linspace(0, len(audio_data) - 1, target_length),
            np.arange(len(audio_data)),
            audio_data,
        )
        sample_rate = 16000

    out_buf = io.BytesIO()
    sf.write(out_buf, audio_data, sample_rate, format="WAV", subtype="PCM_16")
    result = out_buf.getvalue()
    return result


def _convert_audio_to_wav(raw_bytes: bytes) -> bytes | None:
    fmt = _detect_audio_format(raw_bytes)
    if fmt == "wav":
        audio_data, sample_rate = _read_wav_with_soundfile(raw_bytes)
        if audio_data is not None:
            return _write_wav_with_soundfile(audio_data, sample_rate)
        return None
    return None


def _get_suffix_for_format(fmt: str) -> str:
    mapping = {
        "wav":     ".wav",
        "webm":    ".webm",
        "ogg":     ".ogg",
        "unknown": ".webm",
    }
    return mapping.get(fmt, ".webm")


def _save_audio_to_tempfile(raw_bytes: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        prefix="voice_input_",
    )
    tmp.write(raw_bytes)
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────
# Input mode: Voice
# ─────────────────────────────────────────────
def render_audio_input() -> None:
    st.markdown("#### 🎙️ Voice Input")

    disabled = st.session_state.is_processing
    if disabled:
        st.info("⏳ Waiting for the assistant to finish…")
        return

    st.info(
        "Click **Start Recording**, speak your question, "
        "click **Stop**, then hit **Transcribe & Send**."
    )

    audio_value = st.audio_input("Record your question", key="audio_recorder")

    if audio_value is not None:
        raw_bytes = audio_value.getvalue()
        fmt = _detect_audio_format(raw_bytes)

        st.caption(f"Detected format: `{fmt}` | Size: `{len(raw_bytes):,}` bytes")
        wav_bytes = _convert_audio_to_wav(raw_bytes)

        if wav_bytes:
            st.audio(wav_bytes, format="audio/wav")
        else:
            st.caption(
                f"ℹ️ Format `{fmt}` — playback skipped on Windows (no ffmpeg). "
                "Recording is valid. Click **Transcribe & Send** to proceed."
            )

        if st.button("📤 Transcribe & Send", key="transcribe_btn"):
            with st.spinner("Transcribing audio…"):
                tmp_path: str | None = None
                try:
                    from assistant.speech import wav_to_text

                    transcript: str = ""
                    if wav_bytes is not None:
                        audio_buffer = io.BytesIO(wav_bytes)
                        transcript = wav_to_text(audio_buffer)
                    else:
                        suffix = _get_suffix_for_format(fmt)
                        tmp_path = _save_audio_to_tempfile(raw_bytes, suffix)
                        transcript = wav_to_text(tmp_path)

                    if transcript and transcript.strip():
                        st.markdown(f"**Transcript:** _{transcript.strip()}_")
                        _queue_user_input(transcript.strip())
                    else:
                        st.warning(
                            "Could not transcribe audio. "
                            "Please speak clearly and try again."
                        )

                except Exception as exc:
                    st.error(f"Transcription error: {exc}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception: pass


# ─────────────────────────────────────────────
# Input mode: Image
# ─────────────────────────────────────────────
def render_image_input() -> None:
    st.markdown("#### 🖼️ Image Input")

    disabled = st.session_state.is_processing
    if disabled:
        st.info("⏳ Waiting for the assistant to finish…")
        return

    uploaded = st.file_uploader(
        "Upload an image (JPG / PNG / WebP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="img_uploader",
    )
    if uploaded:
        st.image(uploaded, caption="Uploaded image", use_column_width=True)

    img_prompt = st.text_input(
        "Question about the image (optional)",
        placeholder="Describe or ask about this image…",
        key="img_prompt_input",
    )

    if st.button("📤 Send Image", key="send_img_btn"):
        if uploaded is None:
            st.warning("Please upload an image first.")
            return
        img_bytes = uploaded.read()
        prompt    = img_prompt.strip() or "Please describe this image."
        _queue_user_input(prompt, image_bytes=img_bytes)

# ─────────────────────────────────────────────
# Input mode: PDF RAG
# ─────────────────────────────────────────────
def render_pdf_input() -> None:
    st.markdown("#### 📄 PDF Document (RAG)")

    disabled = st.session_state.is_processing
    if disabled:
        st.info("⏳ Waiting for the assistant to finish...")
        return

    uploaded_pdf = st.file_uploader(
        "Upload a PDF document to chat with",
        type=["pdf"],
        key="pdf_uploader",
    )

    pdf_prompt = st.text_input(
        "Question about the document",
        placeholder="E.g., Summarize this document...",
        key="pdf_prompt_input",
    )

    if st.button("📤 Process PDF & Ask", key="send_pdf_btn"):
        if uploaded_pdf is None:
            st.warning("Please upload a PDF first.")
            return
        
        # Save the uploaded PDF to the RAG documents directory
        import os
        from assistant.config import RAG_DOCUMENTS_DIR
        
        os.makedirs(RAG_DOCUMENTS_DIR, exist_ok=True)
        file_path = os.path.join(RAG_DOCUMENTS_DIR, uploaded_pdf.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
            
        st.success(f"Successfully saved {uploaded_pdf.name}.")
        
        if st.session_state.assistant:
            with st.spinner("Indexing PDF into RAG vectorstore..."):
                st.session_state.assistant.rebuild_rag()
                
        prompt = pdf_prompt.strip() or "Please summarize the provided document."
        routed_prompt = f"[INTENT: DOC_QA] {prompt}"
        _queue_user_input(routed_prompt)


# ─────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────
def main() -> None:
    render_sidebar()

    # ── Header ──
    st.markdown(
        "<h1 style='color:#1A1A2E;font-size:2rem;margin-bottom:0'>"
        "🤖 AI Voice Assistant</h1>",
        unsafe_allow_html=True,
    )
    # Thêm thông tin App và Author: Vu Tuan Hung vào tiêu đề
    st.markdown(
        "<p style='color:#6C757D;font-size:15px;margin-top:4px'>"
        "Multi-modal assistant — text, voice &amp; image inputs · Kokoro TTS<br>"
        "<b>Author: Vu Tuan Hung</b></p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Chat history (Nằm trong scrollable container) ──
    chat_container = st.container(height=500)
    with chat_container:
        if not st.session_state.messages and not st.session_state.is_processing:
            st.markdown(
                "<div style='text-align:center;padding:40px;color:#B0B8C1'>"
                "<p style='font-size:3rem'>💬</p>"
                "<p style='font-size:16px'>No messages yet. Start a conversation!</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            render_chat_history()
            
        # ── Processor Phase 2 (Phải ở bên trong container để text sinh ra nằm trong khung) ──
        if st.session_state.is_processing:
            _process_pending_input()
            return

    st.markdown("---")

    # ── Input mode selector ──
    col_title, col_radio = st.columns([2, 8])

    with col_title:
        st.markdown("**Choose Input Mode**")

    with col_radio:
        input_mode = st.radio(
            "Input mode",
            options=["💬 Text", "🎙️ Voice", "🖼️ Image", "📄 PDF"],
            horizontal=True,
            label_visibility="collapsed",
            key="input_mode_radio",
        )
    st.markdown("")

    if input_mode == "💬 Text":
        render_text_input()
    elif input_mode == "🎙️ Voice":
        render_audio_input()
    elif input_mode == "🖼️ Image":
        render_image_input()
    else:
        render_pdf_input()

    # ── Nút Clear Conversation được đưa ra bên ngoài và nằm bên dưới widgets nhập liệu ──
    st.markdown("<div style='margin-top:15px'></div>", unsafe_allow_html=True)
    col_gap, col_clear = st.columns([8, 2])
    with col_clear:
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_input = None
            st.session_state.is_processing = False
            if st.session_state.assistant:
                st.session_state.assistant.conversation_context.forget()
            st.rerun()

if __name__ == "__main__":
    main()