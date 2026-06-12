"""Local LLM provider using llama-cpp-python (ollama.cpp).

Replaces LM Studio with local inference via llama.cpp bindings.
Supports GPU acceleration via n_gpu_layers parameter.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base import LLMProvider
from ...config import (
    OLLAMA_CPP_MODEL_PATH,
    OLLAMA_CPP_N_GPU_LAYERS,
    OLLAMA_CPP_CONTEXT_SIZE,
    OLLAMA_CPP_CLIP_MODEL_PATH,
    ModelCatalog,
)
from ...utils import log


class LocalProvider(LLMProvider):
    """Local LLM provider using llama-cpp-python.
    """

    def __init__(self) -> None:
        if not OLLAMA_CPP_MODEL_PATH:
            raise RuntimeError(
                "OLLAMA_CPP_MODEL_PATH environment variable is required "
            )

        try:
            import llama_cpp
            from llama_cpp import Llama
            self._Llama = Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python package not installed. "
            )

        try:
            # Initialize the model
            self._llm = Llama(
                model_path=OLLAMA_CPP_MODEL_PATH,
                n_gpu_layers=OLLAMA_CPP_N_GPU_LAYERS,
                n_ctx=OLLAMA_CPP_CONTEXT_SIZE,
                verbose=False,
                chat_handler=llama_cpp.llama_chat_format.Qwen25VLChatHandler(
                    clip_model_path=OLLAMA_CPP_CLIP_MODEL_PATH
                ),
                logits_all=True,
                n_threads=8,  # Adjust based on your CPU cores
            )
            log(
                f"Loaded local model from {OLLAMA_CPP_MODEL_PATH} "
                f"(GPU layers: {OLLAMA_CPP_N_GPU_LAYERS})",
                title="LLM",
                style="bold blue",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model from {OLLAMA_CPP_MODEL_PATH}: {exc}"
            ) from exc

        # Infer model name from file path
        model_name = OLLAMA_CPP_MODEL_PATH.split("/")[-1].replace(".gguf", "")
        self._model_name = model_name

        self._catalog = ModelCatalog({
            "conversation": [model_name],
            "vision": [model_name],  # Local models typically don't support vision
            "structured": [model_name],
        })

    # ------------------------------------------------------------------
    # LLMProvider interface properties
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        """Return the underlying Llama instance."""
        return self._llm

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def supports_tools(self) -> bool:
        """Local models can support structured outputs with proper prompting."""
        return True

    # ------------------------------------------------------------------
    # Core completion method
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        capability: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Execute a chat completion using the local llama.cpp model.

        Args:
            capability: One of 'conversation', 'vision', or 'structured'.
            messages:   OpenAI-style message list
            **kwargs:   Optional parameters:
            - max_tokens (int): Max output tokens (default 1024)
            - temperature (float): Sampling temperature (default 0.7)
            - top_p (float): Nucleus sampling (default 0.95)

        Returns:
            An OpenAI-compatible response wrapper.

        Raises:
            RuntimeError: When the model fails to generate.
        """
        try:
            # Convert messages to chat format
            formatted_messages = self._format_messages(messages)

            # Extract generation parameters
            max_tokens = kwargs.get("max_tokens", 1024)
            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 0.95)

            # Call the model
            response = self._llm.create_chat_completion(
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            log(
                f"Generated response ({response['usage']['completion_tokens']} tokens)",
                title="LLM",
                style="bold blue",
            )

            # Wrap in OpenAI-compatible format
            return self._wrap_response(response)

        except Exception as exc:
            log(
                f"Error during completion: {exc}",
                title="LLM",
                style="bold red",
            )
            raise RuntimeError(f"Local LLM completion failed: {exc}") from exc

    def stream_chat_completion(
        self,
        capability: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Stream chat completion from local model."""

        # --- Wrapper classes to mimic OpenAI streaming delta format ---
        class FakeDelta:
            def __init__(self, content: str) -> None:
                self.content = content
                self.tool_calls = None  # Required by ToolLoop

        class FakeChoice:
            def __init__(self, content: str) -> None:
                self.delta = FakeDelta(content)

        class FakeChunk:
            def __init__(self, content: str) -> None:
                self.choices = [FakeChoice(content)]

        # --------------------------------------------------------------

        try:
            formatted_messages = self._format_messages(messages)

            max_tokens = kwargs.get("max_tokens", 1024)
            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 0.95)

            response = self._llm.create_chat_completion(
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            )

            for chunk in response:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield FakeChunk(content)  # ✅ yield object, not str

        except Exception as exc:
            log(
                f"Error during streaming: {exc}",
                title="LLM",
                style="bold red",
            )
            raise RuntimeError(f"Local LLM streaming failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _format_messages(self, messages):
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "tool":
                tool_name = msg.get("name", "tool")
                formatted.append({
                    "role": "user",
                    "content": f"[Tool result from '{tool_name}']: {content}",
                })
            elif isinstance(content, list):
                cleaned_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            cleaned_parts.append(part)
                        elif part.get("type") == "image_url":
                            cleaned_parts.append(part)
                formatted.append({"role": role, "content": cleaned_parts})
            else:
                formatted.append({
                    "role": role,
                    "content": str(content),
                })
        return formatted

    def _wrap_response(self, response: Dict[str, Any]) -> Any:
        """Wrap llama.cpp response in OpenAI-compatible format."""

        # Inner wrapper classes to mimic OpenAI response structure
        class WrappedMessage:
            def __init__(self, content: str) -> None:
                self.role = "assistant"
                self.content = content
                self.tool_calls = None

            def model_dump(self) -> Dict[str, Any]:
                return {
                    "role": self.role,
                    "content": self.content,
                    "tool_calls": [],
                }

        class WrappedChoice:
            def __init__(self, message: WrappedMessage) -> None:
                self.message = message
                self.finish_reason = "stop"

        class WrappedResponse:
            def __init__(self, message: WrappedMessage) -> None:
                self.choices = [WrappedChoice(message)]

        # Extract text from llama.cpp response
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = ""

        message = WrappedMessage(content)
        return WrappedResponse(message)
