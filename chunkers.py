"""
Chunker implementations. Add a new chunking strategy by subclassing Chunker
and implementing .chunk(document) -> List[Chunk].
"""

from __future__ import annotations
from typing import List

from .interfaces import Chunker, Document, Chunk


class FixedSizeChunker(Chunker):
    """Blind N-character splitting, no separator awareness at all."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Chunk]:
        chunks = []
        n = 0
        step = max(self.chunk_size - self.overlap, 1)
        for page in document.pages:
            text = page.content
            for i in range(0, len(text), step):
                piece = text[i:i + self.chunk_size]
                if piece.strip():
                    chunks.append(Chunk(
                        text=piece, chunk_no=n, page_numbers=[page.page_no],
                        source_id=document.source_id,
                    ))
                    n += 1
        return chunks


class RecursiveChunker(Chunker):
    """LangChain's RecursiveCharacterTextSplitter - tries separators in order
    (paragraph, line, sentence, char) but is still blind to actual document
    structure like headings or table boundaries."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    def chunk(self, document: Document) -> List[Chunk]:
        chunks = []
        n = 0
        for page in document.pages:
            for piece in self._splitter.split_text(page.content):
                chunks.append(Chunk(
                    text=piece, chunk_no=n, page_numbers=[page.page_no],
                    source_id=document.source_id,
                ))
                n += 1
        return chunks


class DoclingHybridChunker(Chunker):
    """Structure- and token-aware chunking that respects section/table
    boundaries and injects heading-hierarchy context into each chunk via
    contextualize(). Requires a Document produced by DoclingParser (reads
    the structured DoclingDocument back out of document.metadata).

    Only needs docling-core, not the full docling package - useful if you
    ever want to run this step somewhere lighter than where you parse.
    """

    def __init__(self, tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2", max_tokens: int = 512):
        from docling_core.transforms.chunker import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from transformers import AutoTokenizer

        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(tokenizer_model),
            max_tokens=max_tokens,
        )
        self._chunker = HybridChunker(tokenizer=tokenizer)

    def chunk(self, document: Document) -> List[Chunk]:
        dl_doc = document.metadata.get("docling_document")
        if dl_doc is None:
            raise ValueError(
                "DoclingHybridChunker needs a Document produced by DoclingParser "
                "(expects document.metadata['docling_document'])."
            )

        chunks = []
        for i, dl_chunk in enumerate(self._chunker.chunk(dl_doc=dl_doc)):
            text = self._chunker.contextualize(chunk=dl_chunk)  # prepends heading path
            pages = sorted({
                prov.page_no
                for item in dl_chunk.meta.doc_items
                for prov in item.prov
            }) if dl_chunk.meta.doc_items else []
            chunks.append(Chunk(
                text=text, chunk_no=i, page_numbers=pages,
                source_id=document.source_id,
                metadata={"headings": dl_chunk.meta.headings},
            ))
        return chunks
