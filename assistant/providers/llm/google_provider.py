"""Google (Gemini) LLM provider implementation.

Replaces the Anthropic/Claude provider with Google Gemini API.
Maintains OpenAI-compatible response format for seamless integration.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base import LLMProvider
from ...config import (
    GOOGLE_API_KEY,
    GEMINI_PREFERRED_CHAT_MODEL,
    GEMINI_PREFERRED_VISION_MODEL,
    GEMINI_PREFERRED_TOOL_MODEL,
    DEFAULT_MODEL_CATALOG,
)
from ...utils import log


class GoogleProvider(LLMProvider):
    """Google Gemini API provider.

    Provides chat completion capabilities using Google's Gemini models.
    Converts between OpenAI-style message format and Gemini's native format.

    Setup:
        1. pip install google-generativeai
        2. Set GOOGLE_API_KEY environment variable
        3. Set LLM_PROVIDER=google in your configuration
    """

    def __init__(self) -> None:
        # Validate API key presence before attempting connection
        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY environment variable is required "
                "when LLM_PROVIDER=google"
            )

        try:
            import google.generativeai as genai

            # Configure the SDK with the provided API key
            genai.configure(api_key=GOOGLE_API_KEY)
            self._genai = genai
            log(
                "Initialized Google Gemini provider",
                title="LLM",
                style="bold blue",
            )
        except ImportError:
            raise RuntimeError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )

        # Load model catalog for capability-based model selection
        self._catalog = DEFAULT_MODEL_CATALOG

    # ------------------------------------------------------------------
    # LLMProvider interface properties
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        """Return the underlying google.generativeai module reference."""
        return self._genai

    @property
    def supports_vision(self) -> bool:
        """Gemini Pro Vision and Gemini 1.5+ support image inputs."""
        return True

    @property
    def supports_tools(self) -> bool:
        """Gemini 1.5+ supports function calling (tool use)."""
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
        """Execute a chat completion request using the Gemini API.

        Iterates through available models for the requested capability,
        falling back to the next model on failure.

        Args:
            capability: One of 'chat', 'vision', or 'tools'.
                        Used to select the appropriate model from the catalog.
            messages:   OpenAI-style message list
                        [{"role": "user"|"assistant"|"system", "content": "..."}]
            **kwargs:   Optional overrides:
            - max_tokens (int): Maximum tokens to generate.
            - temperature (float): Sampling temperature 0.0–1.0.
            - tools (list): OpenAI-style tool/function definitions.

        Returns:
            An OpenAI-compatible response wrapper with `.choices[0].message`.

        Raises:
            RuntimeError: When all candidate models fail.
        """
        last_error: Optional[Exception] = None

        # Resolve model list for the requested capability
        models: List[str] = self._catalog.get(capability) or [
            GEMINI_PREFERRED_CHAT_MODEL
        ]

        # Separate system instruction from conversational messages
        system_instruction, chat_messages = self._extract_system(messages)

        for model_name in models:
            try:
                # Build GenerativeModel with optional system instruction
                model_kwargs: Dict[str, Any] = {"model_name": model_name}
                if system_instruction:
                    model_kwargs["system_instruction"] = system_instruction

                # Attach tool declarations when caller provides them
                if "tools" in kwargs:
                    gemini_tools = self._convert_tools(kwargs["tools"])
                    if gemini_tools:
                        model_kwargs["tools"] = gemini_tools

                model = self._genai.GenerativeModel(**model_kwargs)

                # Build generation config from caller overrides
                gen_config = self._build_generation_config(kwargs)

                # Convert messages to Gemini Content format
                gemini_history, last_user_parts = self._convert_messages(
                    chat_messages
                )

                # Start a chat session with historical turns
                chat_session = model.start_chat(history=gemini_history)

                # Send the final user turn and receive the response
                response = chat_session.send_message(
                    last_user_parts,
                    generation_config=gen_config,
                )

                log(
                    f"Model '{model_name}' served the '{capability}' request.",
                    title="MODEL",
                    style="bold blue",
                )

                # Normalise to an OpenAI-compatible wrapper
                return self._wrap_response(response)

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log(
                    f"Model '{model_name}' unavailable ({exc}). "
                    "Trying fallback...",
                    title="MODEL",
                    style="bold yellow",
                )

        raise RuntimeError(
            f"All models failed for capability '{capability}'."
        ) from last_error

    # ------------------------------------------------------------------
    # Message format conversion helpers
    # ------------------------------------------------------------------

    def _extract_system(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Separate the system prompt from the rest of the message list.

        Gemini does not use a 'system' role in the conversation history;
        instead it is provided as `system_instruction` at model creation.

        Args:
            messages: Full OpenAI-style message list including system role.

        Returns:
            A tuple of (system_instruction_text, remaining_messages).
            system_instruction_text is None when no system message exists.
        """
        system_parts: List[str] = []
        remaining: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                # Accumulate multiple system messages into one block
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle multipart content blocks
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            system_parts.append(part["text"])
                else:
                    system_parts.append(str(content))
            else:
                remaining.append(msg)

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return system_instruction, remaining

    def _convert_messages(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], Any]:
        """Convert OpenAI-style messages to Gemini history + final turn.

        Gemini's `start_chat(history=...)` expects all turns EXCEPT the
        last user message.  The last user message is sent via
        `chat_session.send_message(...)`.

        Role mapping:
            OpenAI 'user'      -> Gemini 'user'
            OpenAI 'assistant' -> Gemini 'model'
            OpenAI 'tool'      -> Gemini 'user'  (tool result as user turn)

        Args:
            messages: Filtered message list (system messages already removed).

        Returns:
            (history, last_user_parts) where history is fed to start_chat
            and last_user_parts is sent to send_message.
        """
        history: List[Dict[str, Any]] = []

        # Always ensure the conversation ends with a user turn
        # Walk backwards to find the last user/tool message index
        last_user_index = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") in ("user", "tool"):
                last_user_index = i
                break

        # Fallback: if no user turn found, treat the last message as user
        if last_user_index == -1:
            last_user_index = len(messages) - 1

        # Build history from all messages before the last user turn
        for msg in messages[:last_user_index]:
            role = msg.get("role", "user")
            gemini_role = "model" if role == "assistant" else "user"
            parts = self._build_parts(msg)
            history.append({"role": gemini_role, "parts": parts})

        # Extract the final user turn's parts for send_message
        last_parts = self._build_parts(messages[last_user_index])

        return history, last_parts

    def _build_parts(self, msg: Dict[str, Any]) -> List[Any]:
        """Extract content parts from a single message dict.

        Handles three content shapes:
            1. Plain string  -> single text part
            2. List of dicts -> multipart (text + image_url blocks)
            3. Tool result   -> formatted text part

        Args:
            msg: A single OpenAI-style message dictionary.

        Returns:
            List of Gemini-compatible part objects (str or Blob/Part).
        """
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Tool result messages: wrap with context label
        if role == "tool":
            tool_name = msg.get("name", "tool")
            return [f"[Tool result from '{tool_name}']: {content}"]

        # Multipart content (e.g., text + image)
        if isinstance(content, list):
            parts: List[Any] = []
            for block in content:
                if not isinstance(block, dict):
                    parts.append(str(block))
                    continue

                block_type = block.get("type", "text")

                if block_type == "text":
                    parts.append(block.get("text", ""))

                elif block_type == "image_url":
                    # Attempt inline base64 image conversion
                    image_part = self._convert_image_block(block)
                    if image_part is not None:
                        parts.append(image_part)
                    else:
                        # Fallback: pass URL as text when conversion fails
                        url = block.get("image_url", {}).get("url", "")
                        parts.append(f"[Image URL: {url}]")

            return parts

        # Plain string content
        return [str(content)]

    def _convert_image_block(self, block: Dict[str, Any]) -> Optional[Any]:
        """Convert an OpenAI image_url block to a Gemini inline image part.

        Supports base64-encoded data URIs in the format:
            data:<mime_type>;base64,<encoded_data>

        Args:
            block: OpenAI image_url content block.

        Returns:
            A google.generativeai types.Part object, or None on failure.
        """
        try:
            from google.generativeai import types as genai_types

            url_info = block.get("image_url", {})
            url = url_info.get("url", "") if isinstance(url_info, dict) else str(url_info)

            # Only handle base64 data URIs; skip plain HTTP URLs
            if not url.startswith("data:"):
                return None

            # Parse  "data:<mime>;base64,<data>"
            header, encoded = url.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]

            import base64
            image_bytes = base64.b64decode(encoded)

            return genai_types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            )
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # Tool format conversion helpers
    # ------------------------------------------------------------------

    def _convert_tools(
        self, tools: List[Dict[str, Any]]
    ) -> List[Any]:
        """Convert OpenAI-style tool definitions to Gemini FunctionDeclarations.

        OpenAI format:
            {"type": "function",
             "function": {"name": ..., "description": ..., "parameters": {...}}}

        Gemini format (via SDK helper):
            genai.protos.FunctionDeclaration(
                name=..., description=..., parameters=Schema(...)
            )

        Args:
            tools: List of OpenAI-style tool definition dicts.

        Returns:
            List of Gemini Tool objects, or empty list on failure.
        """
        try:
            from google.generativeai import protos

            declarations: List[Any] = []

            for tool in tools:
                if tool.get("type") != "function":
                    continue

                func = tool.get("function", {})
                name = func.get("name", "")
                description = func.get("description", "")
                parameters = func.get("parameters", {})

                # Build the parameter schema for Gemini
                schema = self._build_parameter_schema(parameters)

                declarations.append(
                    protos.FunctionDeclaration(
                        name=name,
                        description=description,
                        parameters=schema,
                    )
                )

            if not declarations:
                return []

            # Wrap declarations inside a Tool container
            return [self._genai.protos.Tool(function_declarations=declarations)]

        except Exception as exc:  # noqa: BLE001
            log(
                f"Failed to convert tools for Gemini: {exc}",
                title="MODEL",
                style="bold yellow",
            )
            return []

    def _build_parameter_schema(
        self, parameters: Dict[str, Any]
    ) -> Any:
        """Recursively convert a JSON Schema dict to a Gemini Schema object.

        Handles:
            - Primitive types: string, integer, number, boolean
            - Object types with nested properties
            - Array types with item schemas

        Args:
            parameters: JSON Schema dict (OpenAI function parameters format).

        Returns:
            A google.generativeai.protos.Schema object.
        """
        from google.generativeai import protos

        # Map JSON Schema type strings to Gemini Type enum values
        type_map: Dict[str, Any] = {
            "string":  protos.Type.STRING,
            "integer": protos.Type.INTEGER,
            "number":  protos.Type.NUMBER,
            "boolean": protos.Type.BOOLEAN,
            "object":  protos.Type.OBJECT,
            "array":   protos.Type.ARRAY,
        }

        schema_type_str = parameters.get("type", "object")
        schema_type = type_map.get(schema_type_str, protos.Type.OBJECT)

        schema_kwargs: Dict[str, Any] = {"type_": schema_type}

        # Include top-level description when present
        if "description" in parameters:
            schema_kwargs["description"] = parameters["description"]

        # Recursively convert nested object properties
        if schema_type_str == "object" and "properties" in parameters:
            nested: Dict[str, Any] = {}
            for prop_name, prop_schema in parameters["properties"].items():
                nested[prop_name] = self._build_parameter_schema(prop_schema)
            schema_kwargs["properties"] = nested

            # Preserve required field list
            if "required" in parameters:
                schema_kwargs["required"] = parameters["required"]

        # Recursively convert array item schema
        elif schema_type_str == "array" and "items" in parameters:
            schema_kwargs["items"] = self._build_parameter_schema(
                parameters["items"]
            )

        return protos.Schema(**schema_kwargs)

    # ------------------------------------------------------------------
    # Generation configuration
    # ------------------------------------------------------------------

    def _build_generation_config(
        self, kwargs: Dict[str, Any]
    ) -> Any:
        """Build a Gemini GenerationConfig from caller-supplied kwargs.

        Recognised keys:
            max_tokens  (int)   -> mapped to max_output_tokens
            temperature (float) -> passed through directly
            top_p       (float) -> passed through directly
            top_k       (int)   -> passed through directly

        Args:
            kwargs: Keyword arguments forwarded from chat_completion.

        Returns:
            A google.generativeai.types.GenerationConfig instance.
        """
        config_kwargs: Dict[str, Any] = {}

        if "max_tokens" in kwargs:
            config_kwargs["max_output_tokens"] = int(kwargs["max_tokens"])

        if "temperature" in kwargs:
            config_kwargs["temperature"] = float(kwargs["temperature"])

        if "top_p" in kwargs:
            config_kwargs["top_p"] = float(kwargs["top_p"])

        if "top_k" in kwargs:
            config_kwargs["top_k"] = int(kwargs["top_k"])

        return self._genai.types.GenerationConfig(**config_kwargs)

    # ------------------------------------------------------------------
    # Response normalisation
    # ------------------------------------------------------------------

    def _wrap_response(self, response: Any) -> Any:
        """Wrap a Gemini response in an OpenAI-compatible structure.

        The rest of the codebase expects:
            response.choices[0].message.content       -> str
            response.choices[0].message.tool_calls    -> list | None
            response.choices[0].message.model_dump()  -> dict

        This method bridges Gemini's native response to that interface.

        Args:
            response: Raw response object from Gemini SDK.

        Returns:
            A WrappedResponse instance mimicking OpenAI's response shape.
        """

        # --- Inner wrapper classes ----------------------------------------

        class WrappedMessage:
            """Mimics openai.types.chat.ChatCompletionMessage."""

            def __init__(
                self,
                content: str,
                tool_calls: Optional[List[Dict[str, Any]]] = None,
            ) -> None:
                self.role = "assistant"
                self.content = content
                self.tool_calls = tool_calls  # None when no tools were called

            def model_dump(self) -> Dict[str, Any]:
                """Serialise to dict (mirrors Pydantic's model_dump)."""
                return {
                    "role": self.role,
                    "content": self.content,
                    "tool_calls": self.tool_calls or [],
                }

        class WrappedChoice:
            """Mimics openai.types.chat.Choice."""

            def __init__(self, message: WrappedMessage) -> None:
                self.message = message
                # Gemini doesn't expose finish_reason in the same way;
                # default to 'stop' for compatibility
                self.finish_reason = "tool_calls" if message.tool_calls else "stop"

        class WrappedResponse:
            """Mimics openai.types.chat.ChatCompletion."""

            def __init__(
                self,
                content: str,
                tool_calls: Optional[List[Dict[str, Any]]] = None,
            ) -> None:
                self.choices = [WrappedChoice(WrappedMessage(content, tool_calls))]

        # --- Extract text and function calls from Gemini response ----------

        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        # response.candidates is a list; we use the first (best) candidate
        candidates = getattr(response, "candidates", [])
        if not candidates:
            # Fallback: try response.text for simple text-only responses
            fallback_text = getattr(response, "text", "") or ""
            return WrappedResponse(fallback_text)

        content_obj = getattr(candidates[0], "content", None)
        parts = getattr(content_obj, "parts", []) if content_obj else []

        for part in parts:
            # Text part
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)

            # Function call part (tool use)
            elif hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                # Gemini returns arguments as a MapComposite (dict-like) object
                try:
                    arguments = json.dumps(dict(fc.args))
                except Exception:  # noqa: BLE001
                    arguments = str(fc.args)

                tool_calls.append(
                    {
                        # Gemini does not provide a call ID; generate a stable one
                        "id": f"call_{fc.name}_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": arguments,
                        },
                    }
                )

        combined_text = "".join(text_parts)
        return WrappedResponse(
            combined_text,
            tool_calls if tool_calls else None,
        )