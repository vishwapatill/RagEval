"""
pip install google-genai     # for GoogleGenAILLM
pip install ollama           # for OllamaLLM
"""

from __future__ import annotations

from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from .interfaces import LLM

T = TypeVar("T", bound=BaseModel)


# ═══════════════════════════════════════════════════════════════════════════
# Google GenAI (Gemini / Gemma)
# ═══════════════════════════════════════════════════════════════════════════


class GoogleGenAILLM(LLM):
    """Wraps the google-genai SDK.

    Use one instance with model="gemma-4-12b-it" for generation and a
    SEPARATE instance with model="gemini-2.5-flash" for judging, so the
    judge isn't grading its own generator.

    pip install google-genai
    """

    def __init__(self, model: str, api_key: str):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Plain text completion."""
        from google.genai import types

        config = types.GenerateContentConfig()
        if system_prompt:
            config.system_instruction = system_prompt

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        return response.text

    def invoke_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """Structured output — returns a validated Pydantic instance.

        The google-genai SDK natively accepts a Pydantic class as
        response_schema. The model is constrained to emit valid JSON
        matching the schema, and response.parsed gives back the
        validated object directly.
        """
        from google.genai import types

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_model,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        # response.parsed is the Pydantic instance when response_schema
        # was a Pydantic model.
        return response.parsed


# ═══════════════════════════════════════════════════════════════════════════
# Ollama (local models)
# ═══════════════════════════════════════════════════════════════════════════


class OllamaLLM(LLM):
    """Wraps the ollama Python SDK for local model inference.

    Uses Ollama's constrained decoding — the model's token generation
    is restricted at inference time to only produce valid JSON matching
    the schema.  No parsing failures, no markdown fences.

    pip install ollama
    # Then pull a model: ollama pull gemma4:12b-it-qat
    """

    def __init__(
        self,
        model: str = "gemma4:12b-it-qat",
        host: Optional[str] = None,
        temperature: float = 0.0,
    ):
        import ollama
        self._model = model
        self._temperature = temperature
        self._client = ollama.Client(host=host) if host else ollama.Client()

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Plain text completion."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat(
            model=self._model,
            messages=messages,
            options={"temperature": self._temperature},
        )
        return response.message.content

    def invoke_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """Structured output — returns a validated Pydantic instance.

        Ollama's format parameter accepts a JSON schema dict.  We
        generate it from the Pydantic model via model_json_schema(),
        then validate the response back into the Pydantic class.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat(
            model=self._model,
            messages=messages,
            format=response_model.model_json_schema(),
            options={"temperature": self._temperature},
        )
        return response_model.model_validate_json(response.message.content)

