"""
Core abstractions for the RAG chunking-comparison project.

Every concrete Parser / Chunker / Embedder / Retriever / LLM implementation
inherits from the ABCs below and implements the abstract methods. Swap any
stage of the pipeline by writing a new subclass - nothing else changes.

    Parser    : source (file path)      -> Document
    Chunker   : Document                -> List[Chunk]
    Embedder  : text                    -> vector
    Retriever : List[Chunk] (+Embedder) -> retrieve(query) -> List[Chunk]
    LLM       : prompt                  -> text  (used for both generation and judging)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Page:
    page_no: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    source_id: str                       # e.g. filename or path
    pages: List[Page]
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Parser-specific rich objects go here rather than forcing every parser
    # into a lossy plain-text-only format. Example: DoclingParser stashes
    # the full structured DoclingDocument in metadata["docling_document"]
    # so DoclingHybridChunker can read layout/table/heading structure back
    # out - the flat `pages` text alone would lose exactly what you're
    # trying to test.


@dataclass
class Chunk:
    text: str
    chunk_no: int
    page_numbers: List[int] = field(default_factory=list)
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Abstract base classes - inherit from these to add a new implementation
# --------------------------------------------------------------------------

class Parser(ABC):
    """Turns a raw source (PDF path, etc.) into a Document."""

    @abstractmethod
    def parse(self, source: str) -> Document:
        ...


class Chunker(ABC):
    """Turns a Document into a list of Chunks."""

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        ...


class Embedder(ABC):
    """Turns text into vectors. Two methods because some embedding models
    (e.g. BGE-style) genuinely embed queries differently from documents -
    if yours doesn't, just implement embed_query as a call to embed_documents."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        ...


class Retriever(ABC):
    """Indexes chunks (via an Embedder) and retrieves the top-k for a query.
    Swap this to change vector store / similarity metric without touching
    anything else - e.g. an in-memory numpy version for small experiments,
    a Chroma/FAISS version once your chunk counts grow."""

    @abstractmethod
    def index(self, chunks: List[Chunk]) -> None:
        ...

    @abstractmethod
    def retrieve(self, query: str, k: int) -> List[Chunk]:
        ...


class LLM(ABC):
    """Anything that can answer a prompt - used for both RAG answer
    generation and as the evaluation judge. Pass two different instances
    to Evaluator if you want the judge to be a different model from the
    generator (recommended, to avoid self-preference bias)."""

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        ...
