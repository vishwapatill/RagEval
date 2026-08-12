"""
Retriever implementations. Add a different vector store / similarity metric
by subclassing Retriever and implementing index() / retrieve().
"""

from __future__ import annotations
from typing import List

from .interfaces import Retriever, Chunk, Embedder
from .utils import cosine_similarity


class InMemoryCosineRetriever(Retriever):
    """Plain numpy cosine-similarity search. Fine for a few thousand chunks
    (your 15-18 test docs won't get anywhere close). Swap for a Chroma/FAISS
    subclass later without touching Evaluator or anything upstream."""

    def __init__(self, embedder: Embedder):
        self._embedder = embedder
        self._chunks: List[Chunk] = []
        self._vectors: List[List[float]] = []

    def index(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("Chunker produced zero chunks - nothing to index.")
        self._chunks = chunks
        self._vectors = self._embedder.embed_documents([c.text for c in chunks])

    def retrieve(self, query: str, k: int) -> List[Chunk]:
        if not self._chunks:
            raise RuntimeError("Call .index(chunks) before .retrieve().")
        q_vec = self._embedder.embed_query(query)
        sims = [cosine_similarity(q_vec, v) for v in self._vectors]
        top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return [self._chunks[i] for i in top_idx]
