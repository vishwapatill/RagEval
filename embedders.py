"""
Embedder implementations. Add a new embedding backend by subclassing
Embedder and implementing embed_documents() / embed_query().
"""

from __future__ import annotations
from typing import List

from .interfaces import Embedder


class HuggingFaceEmbedder(Embedder):
    """Local embedding model via sentence-transformers - runs fine on your
    4050 or even CPU at this scale."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from langchain_huggingface import HuggingFaceEmbeddings
        self._model = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_query(text)


# To add a hosted embedding API instead (e.g. Gemini embeddings), subclass
# Embedder the same way - embed_documents(texts) -> List[List[float]] and
# embed_query(text) -> List[float] is the entire contract. Nothing else in
# the pipeline needs to know or care which one you're using.
