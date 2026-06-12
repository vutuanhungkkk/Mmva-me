"""Core orchestration logic for the voice assistant."""
from __future__ import annotations
import io
import time
import threading
from typing import Any, Dict, List, Optional
from pathlib import Path
import speech_recognition as sr
from rich.panel import Panel
import os
import warnings

os.environ["LLAMA_CPP_LOG_LEVEL"] = "error" 
os.environ["GGML_LOG_LEVEL"] = "error"    
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*__path__.*")


from .config import (
    SYSTEM_MESSAGE,
    ENABLE_TOOL_CALLING,
    SIMPLE_TOOLS,
    RAG_ENABLED,
    RAG_VECTOR_DB_DIR,
    RAG_NUM_RETRIEVED,
    RAG_DOCUMENTS_DIR,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    TTS_PROVIDER,
)
from .context import EnhancedConversationContext, ContextProviderRegistry, MCPContextProvider
from .providers.llm import get_llm_provider, LLMProvider
from .providers.tts import get_tts_provider, TTSProvider
from .speech import wav_to_text, extract_prompt
from .tools import (
    ToolLoop,
    ToolRegistry,
    capture_screenshot_context_tool,
    capture_webcam_context_tool,
    duckduckgo_search,
    duckduckgo_search_tool,
    extract_clipboard_text_tool,
    process_search_results,
)
from .tools.vision_tools import set_llm_provider
from .utils import console, log, save_log


class VoiceAssistant:
    """Main voice assistant orchestrator."""

    def __init__(self) -> None:
        self.llm_provider: LLMProvider = get_llm_provider()
        self.tts_provider: TTSProvider = get_tts_provider()
        set_llm_provider(self.llm_provider)

        self.conversation_context = EnhancedConversationContext()
        self.context_provider_registry = ContextProviderRegistry()
        self.context_provider_registry.register(MCPContextProvider())

        self.tool_registry = ToolRegistry()
        self._register_builtin_tools()

        self.tool_loop = ToolLoop(
            llm_provider=self.llm_provider,
            tool_registry=self.tool_registry,
            tools_enabled=ENABLE_TOOL_CALLING,
        )

        self.convo: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_MESSAGE}]
        self.recognizer = sr.Recognizer()
        # RAG/vectorstore (optional)
        self.vectorstore = None
        if RAG_ENABLED:
            self._load_vectorstore()
            # If vectorstore is missing or empty, attempt to build from documents
            existing_count = self._vectorstore_count()
            if not self.vectorstore or existing_count == 0:
                try:
                    from .rag_builder import build_vectorstore

                    docs_dir = Path(RAG_DOCUMENTS_DIR)
                    if not docs_dir.exists():
                        log(f"RAG documents directory not found: {RAG_DOCUMENTS_DIR}", title="RAG", style="bold yellow")
                    else:
                        pdf_count = len(list(docs_dir.rglob("*.pdf")))
                        if pdf_count == 0:
                            log(f"No PDF files found in {RAG_DOCUMENTS_DIR}; skipping RAG build.", title="RAG", style="bold yellow")
                        else:
                            log(
                                f"Building RAG vector DB from {RAG_DOCUMENTS_DIR} ({pdf_count} PDF file(s))",
                                title="RAG",
                                style="bold blue",
                            )
                            build_vectorstore(RAG_DOCUMENTS_DIR, RAG_VECTOR_DB_DIR, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)
                            # Try loading again
                            self._load_vectorstore()
                except Exception as exc:  # noqa: BLE001 - best-effort
                    log(f"Automatic RAG build failed: {exc}", title="RAG", style="bold yellow")

        if not ENABLE_TOOL_CALLING:
            log("Tool calling disabled via ASSISTANT_DISABLE_TOOLS.", title="TOOLS", style="bold yellow")
        if SIMPLE_TOOLS:
            log("Simple tools mode enabled. Vision tools disabled.", title="TOOLS", style="bold blue")

    def _register_builtin_tools(self) -> None:
        """Register built-in tools with the registry."""
        self.tool_registry.register(
            name="extract_clipboard_text",
            description="Extract the latest textual content from the user's clipboard.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: extract_clipboard_text_tool(),
        )

        # Only register duckduckgo_search if RAG is NOT enabled
        if not RAG_ENABLED:
            self.tool_registry.register(
                name="duckduckgo_search",
                description="Perform a DuckDuckGo search and return the most relevant results.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to run on DuckDuckGo.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default 5).",
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["query"],
                },
                handler=lambda query, max_results=5: duckduckgo_search_tool(query=query, max_results=max_results),
            )

        if SIMPLE_TOOLS:
            return

        self.tool_registry.register(
            name="capture_screenshot_context",
            description=(
                "Capture a screenshot on the user's machine (macOS supported) and "
                "describe it for additional conversation context."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "user_prompt": {
                        "type": "string",
                        "description": "The user's current request to guide the screenshot analysis.",
                    }
                },
            },
            handler=lambda user_prompt="": capture_screenshot_context_tool(user_prompt=user_prompt),
        )
        self.tool_registry.register(
            name="capture_webcam_context",
            description="Capture a webcam photo and describe it for additional conversation context.",
            parameters={
                "type": "object",
                "properties": {
                    "user_prompt": {
                        "type": "string",
                        "description": "The user's current request to guide the webcam analysis.",
                    }
                },
            },
            handler=lambda user_prompt="": capture_webcam_context_tool(user_prompt=user_prompt),
        )

    def _load_vectorstore(self) -> None:
        try:
            # Import lazily to avoid hard dependency at startup
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_chroma import Chroma

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )

            self.vectorstore = Chroma(persist_directory=RAG_VECTOR_DB_DIR, embedding_function=embeddings)
            count = None
            try:
                count = getattr(self.vectorstore, "_collection").count()
            except Exception:
                pass
            log(f"RAG vectorstore loaded (dir={RAG_VECTOR_DB_DIR}) count={count}", title="RAG", style="bold green")
        except Exception as exc:  # noqa: BLE001 - non-fatal
            self.vectorstore = None
            log(f"Could not load RAG vectorstore: {exc}", title="RAG", style="bold yellow")

    def _vectorstore_count(self) -> int:
        """Best-effort count of indexed chunks in the active vectorstore."""
        if not self.vectorstore:
            return 0
        try:
            count = getattr(self.vectorstore, "_collection").count()
            return int(count or 0)
        except Exception:
            return 0

    def _retrieve_and_process(self, query: str) -> str:
        """Retrieve top documents for `query` and return a single joined text block."""
        if not self.vectorstore:
            return ""

        docs = []
        try:
            # langchain VectorStore API: similarity_search
            docs = self.vectorstore.similarity_search(query, k=RAG_NUM_RETRIEVED)
        except Exception:
            try:
                # Some wrappers expose a `as_retriever()` interface
                retr = self.vectorstore.as_retriever(search_kwargs={"k": RAG_NUM_RETRIEVED})
                docs = retr.get_relevant_documents(query)  # type: ignore[attr-defined]
            except Exception:
                docs = []

        if not docs:
            return ""

        parts: List[str] = []
        for i, d in enumerate(docs, start=1):
            meta = getattr(d, "metadata", {}) or {}
            src = meta.get("source") or meta.get("file") or meta.get("path") or "unknown"
            parts.append(f"[DOC {i}] Source: {src}\n{getattr(d, 'page_content', str(d))}\n")

        return "\n".join(parts)

    def llm_prompt(self, prompt: str, img_context: Optional[str] = None, use_rag: bool = False) -> str:
        """Run the user's prompt through the LLM and stream the response to TTS."""
        base_prompt = prompt
        context = self.conversation_context.get_context()
        provider_context = self.context_provider_registry.gather(
            prompt=prompt, conversation_history=self.conversation_context.history
        )

        if context:
            prompt = f"Previous conversation:\n{context}\n\nCurrent user prompt: {prompt}"
        if provider_context:
            prompt = f"{prompt}\n\nAdditional context from providers:\n{provider_context}"

        if use_rag and RAG_ENABLED and self.vectorstore:
            log("Executing vector similarity search for context injection.", title="RAG", style="dim")
            retrieved = self._retrieve_and_process(base_prompt)

            if retrieved:
                rag_instruction = (
                    "===== RELEVANT DOCUMENTS =====\n"
                    f"{retrieved}\n"
                    "==============================\n\n"
                    "Instruction: Read the above documents carefully. "
                    "DO NOT use external knowledge.\n"
                    "If the answer is not in the documents, clearly state: "
                    "'This information is not available in the provided documents'.\n"
                    "Always cite which document [DOC X] your answer comes from.\n\n"
                    f"Based ONLY on the documents above, answer:\n"
                    f"USER QUERY: {prompt}"
                )
                final_prompt = rag_instruction
            else:
                final_prompt = (
                    f"{prompt}\n\n"
                    "[Note: No relevant documents were found in the knowledge base. "
                    "You may answer using general knowledge.]"
                )
        else:
            general_system_message = (
                "You are a multi-modal AI voice assistant. Your user may or may not have "
                "attached a photo for context (either a screenshot or a webcam capture). "
                "Any photo has already been processed into a highly detailed text prompt "
                "that will be attached to their transcribed voice prompt. Generate the most "
                "useful and factual response possible, carefully considering all previous "
                "generated text in your response before adding new tokens to the response. "
                "Do not expect or request images, just use the context if added. "
                "Use all of the context of this conversation so your response is relevant "
                "to the conversation. Make your responses clear and concise, avoiding any verbosity."
            )
            self.convo[0] = {"role": "system", "content": general_system_message}
            final_prompt = prompt

        # ── Build message payload ──
        if img_context and self.llm_provider.supports_vision:
            self.convo.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": final_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": img_context},
                    },
                ],
            })
        else:
            if img_context and not self.llm_provider.supports_vision:
                final_prompt = f"{final_prompt}\n\n[User attached an image but vision is not supported]"
            self.convo.append({"role": "user", "content": final_prompt})

        # ── Stream LLM response → TTS ──
        full_response_text = ""

        def speakable_chunks():
            nonlocal full_response_text
            for chunk in self.tool_loop.stream(self.convo):
                full_response_text += chunk
                yield chunk

        self.tts_provider.stream_speak(speakable_chunks())

        if not use_rag:
            from .config import SYSTEM_MESSAGE
            self.convo[0] = {"role": "system", "content": SYSTEM_MESSAGE}

        self.conversation_context.add_exchange(base_prompt, full_response_text)
        return full_response_text

    def llm_prompt_with_image(self, prompt: str, image_b64: str) -> str:
        """
        Run a prompt that includes a base64-encoded image through the LLM.
        The image is passed as img_context so vision-capable providers can
        handle it; for text-only providers the image description is skipped.

        Args:
            prompt:     The user's text question.
            image_b64:  A data-URI string  e.g. 'data:image/jpeg;base64,<bytes>'

        Returns:
            The assistant's full response text.
        """
        return self.llm_prompt(prompt=prompt, img_context=image_b64)

    def speak(self, text: str) -> None:
        """Speak text using the configured TTS provider with fallback to OpenAI."""
        if self.tts_provider.speak(text):
            return

        if TTS_PROVIDER != "openai":
            log("Falling back to OpenAI TTS.", title="TTS", style="bold yellow")
            from .providers.tts.openai_tts import OpenAITTSProvider
            try:
                fallback = OpenAITTSProvider()
                if fallback.speak(text):
                    return
            except Exception:
                pass

        log("Unable to synthesise speech for the assistant response.", title="TTS", style="bold red")

    def callback(self, recognizer: sr.Recognizer, audio: sr.AudioData) -> None:
        """Audio callback for background listening."""
        wav_data = io.BytesIO(audio.get_wav_data())
        prompt_text = wav_to_text(wav_data)
        log(f"Heard: {prompt_text!r}", title="DEBUG", style="dim")
        clean_prompt = prompt_text.strip()

        if not clean_prompt:
            return

        log(f"USER: {clean_prompt}", title="USER INPUT", style="bold green")
        response = self._handle_command(clean_prompt)
        log(f"ASSISTANT: {response}", title="ASSISTANT RESPONSE", style="bold magenta")

    def _handle_command(self, clean_prompt: str) -> str:
        """Route a recognised prompt to memory commands, search, or the LLM."""
        lowered = clean_prompt.lower()
        if lowered.startswith("remember "):
            self.conversation_context.remember(clean_prompt[9:])
            return "I've remembered that information."
        if lowered == "forget context":
            return self.conversation_context.forget()
        if lowered.startswith("search "):
            search_query = clean_prompt[7:]
            # If RAG is enabled, only use local retrieval (no web search)
            if RAG_ENABLED:
                if self.vectorstore:
                    retrieved = self._retrieve_and_process(search_query)
                    if retrieved:
                        return self.llm_prompt(
                            prompt=(
                                "USER QUERY: " + search_query + "\n\n"
                                "You must ONLY answer based on the retrieved documents provided below. "
                                "Do not use general knowledge. If the answer is not in the documents, say so explicitly."
                            ),
                            img_context=None,
                        )
                return self.llm_prompt(
                    prompt=f"Unable to find relevant documents for: {search_query}",
                    img_context=None,
                )
            # If RAG is NOT enabled, use web search
            search_results = duckduckgo_search(search_query)
            processed_results = process_search_results(search_results)
            return self.llm_prompt(
                prompt=(
                    "Based on the following search results, answer the query: "
                    f"{search_query}\n\n{processed_results}"
                ),
                img_context=None,
            )
        return self.llm_prompt(prompt=clean_prompt, img_context=None)


    def run_once(
        self,
        prompt: str,
        image_b64: Optional[str] = None,
        audio_wav: Optional[io.BytesIO] = None,
    ) -> str:
        """
        Process a single user turn coming from any input modality.

        Args:
            prompt:     Text prompt (required; may be empty string for audio-only).
            image_b64:  Optional base64 data-URI image string.
            audio_wav:  Optional BytesIO containing WAV audio to transcribe first.

        Returns:
            The assistant's text response.
        """
        # 1. Transcribe audio if provided (overrides/supplements text prompt)
        if audio_wav is not None:
            from .speech import wav_to_text, extract_prompt
            transcribed = wav_to_text(audio_wav)
            log(f"Transcribed audio: {transcribed!r}", title="STT", style="dim")
            clean = transcribed.strip()
            if clean:
                prompt = clean
            elif not prompt:
                log("No speech detected and no text prompt provided.", title="STT", style="bold yellow")
                return "I didn't catch that. Could you please repeat?"

        if not prompt:
            return "Please provide a question or command."

        log(f"USER: {prompt}", title="USER INPUT", style="bold green")
        response = self._handle_command_with_image(prompt, image_b64)
        log(f"ASSISTANT: {response}", title="ASSISTANT RESPONSE", style="bold magenta")
        return response


    def _handle_command_with_image(
        self, clean_prompt: str, image_b64: Optional[str] = None
    ) -> str:
        """
        Routes memory commands, handles search, determines intent for RAG,
        and finally passes the payload to the LLM.
        """
        lowered = clean_prompt.lower()

        if lowered.startswith("remember "):
            self.conversation_context.remember(clean_prompt[9:])
            return "I've remembered that information."

        if lowered == "forget context":
            return self.conversation_context.forget()

        if lowered.startswith("search ") and image_b64 is None:
            return self._handle_command(clean_prompt)
            
        # ── Intent Routing Execution ──
        intent = self.determine_intent(clean_prompt)
        
        # Clean the hidden marker sent from the UI (if any)
        clean_prompt = clean_prompt.replace("[INTENT: DOC_QA]", "").strip()

        # Route to RAG block if DOC_QA intent is detected
        if intent == "DOC_QA":
            return self.llm_prompt(prompt=clean_prompt, img_context=image_b64, use_rag=True)
            
        # Fallback to standard conversational LLM stream (no unnecessary RAG retrieval)
        return self.llm_prompt(prompt=clean_prompt, img_context=image_b64, use_rag=False)

    def start_listening(self) -> None:
        """Load a single test command from command.txt and run it once (CLI mode)."""
        command_file = Path(__file__).resolve().parent / "command.txt"

        if not command_file.exists():
            log(
                f"Test command file not found: {command_file}",
                title="ACTION",
                style="bold yellow",
            )
            return

        try:
            clean_prompt = command_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            log(f"Could not read test command file: {e}", title="ACTION", style="bold red")
            save_log()
            return

        if not clean_prompt:
            log("Test command file is empty.", title="ACTION", style="bold yellow")
            return

        console.print(
            Panel(
                f"Loaded test command from {command_file.name}.",
                border_style="bold magenta",
                title="INSTRUCTIONS",
            )
        )

        # ── Use the unified run_once() entry point ──
        response = self.run_once(prompt=clean_prompt)
        log(f"ASSISTANT: {response}", title="ASSISTANT RESPONSE", style="bold magenta")
        save_log()

    def rebuild_rag(self) -> None:
        """Dynamically rebuild the vector store and reload it into memory."""
        try:
            from .rag_builder import build_vectorstore
            
            docs_dir = Path(RAG_DOCUMENTS_DIR)
            if not docs_dir.exists():
                log(f"RAG documents directory missing: {RAG_DOCUMENTS_DIR}", title="RAG", style="bold yellow")
                return
                
            log("Rebuilding RAG vector DB with new documents...", title="RAG", style="bold blue")
            build_vectorstore(RAG_DOCUMENTS_DIR, RAG_VECTOR_DB_DIR, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)
            
            # Reload the newly built vectorstore into context
            self._load_vectorstore()
            log("RAG vectorstore updated successfully.", title="RAG", style="bold green")
        except Exception as exc: 
            log(f"Dynamic RAG rebuild failed: {exc}", title="RAG", style="bold red")

    def determine_intent(self, prompt: str) -> str:
        """
        Classify the user intent to either trigger RAG (DOC_QA) or normal chat (GENERAL).
        This acts as the primary Intent Router.
        """
        # 1. Explicit UI marker check
        if "[INTENT: DOC_QA]" in prompt:
            log("Intent Router: Forced DOC_QA from UI.", title="ROUTER", style="dim")
            return "DOC_QA"
            
        # 2. Keyword heuristic check for voice and text inputs
        lower_prompt = prompt.lower()
        rag_keywords = ["pdf", "document", "the file", "report", "context"]
        if any(kw in lower_prompt for kw in rag_keywords):
            log("Intent Router: Keyword match, routed to DOC_QA.", title="ROUTER", style="dim")
            return "DOC_QA"
            
        # Default behavior is general chat
        log("Intent Router: Routed to GENERAL chat.", title="ROUTER", style="dim")
        return "GENERAL"

def main() -> None:
    """Main entry point for the voice assistant."""
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="faster_whisper")

    assistant = VoiceAssistant()
    assistant.start_listening()


if __name__ == "__main__":
    main()
