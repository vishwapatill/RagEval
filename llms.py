"""
LLM implementations. Add a new model backend by subclassing LLM and
implementing invoke(prompt) -> str.
"""

from __future__ import annotations

from .interfaces import LLM


class GoogleGenAILLM(LLM):
    """Wraps the google-genai SDK. Use one instance with
    model="gemma-4-12b-it" for generation and a SEPARATE instance with
    model="gemini-2.5-flash" for judging, so the judge isn't grading its
    own generator."""

    def __init__(self, model: str, api_key: str):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def invoke(self, prompt: str) -> str:
        response = self._client.models.generate_content(model=self._model, contents=prompt)
        return response.text
