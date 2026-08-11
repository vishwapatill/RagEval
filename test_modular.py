"""
Smoke test for the modular rag_eval package - no real APIs/models needed.
Run: python3 tests/test_modular.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_eval import (
    Document, Page, Chunk,
    Parser, Chunker, Embedder, LLM,
    InMemoryCosineRetriever, RAGPipeline,
)


# ---- Mock implementations, each inheriting from the real ABCs ----

class MockParser(Parser):
    """Pretends to parse a PDF - just hands back canned pages."""
    DOCS = {
        "doc_a.pdf": [
            (12, "Q3 automotive revenue was $45.2M, up 8% year over year. " + "filler text " * 10),
        ],
        "doc_b.pdf": [
            (5, "unrelated filler content about something else entirely. " * 5),
        ],
    }

    def parse(self, source: str) -> Document:
        pages = [Page(page_no=pn, content=text) for pn, text in self.DOCS[source]]
        return Document(source_id=source, pages=pages)


class MockFixedSizeChunker(Chunker):
    def __init__(self, size=60):
        self.size = size

    def chunk(self, document: Document):
        chunks = []
        n = 0
        for page in document.pages:
            for i in range(0, len(page.content), self.size):
                piece = page.content[i:i + self.size]
                chunks.append(Chunk(text=piece, chunk_no=n, page_numbers=[page.page_no], source_id=document.source_id))
                n += 1
        return chunks


class MockEmbedder(Embedder):
    """Bag-of-words hashing so cosine similarity reflects real word overlap."""
    VOCAB = ["revenue", "automotive", "q3", "45", "2m", "growth", "8", "unrelated", "filler", "text"]

    def _vec(self, text):
        text = text.lower()
        return [text.count(w) for w in self.VOCAB]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class MockLLM(LLM):
    def invoke(self, prompt: str) -> str:
        if "Respond with ONLY a JSON object" in prompt:
            return '{"faithfulness": 0.9, "answer_relevancy": 0.85, "context_adequacy": 0.8, "reasoning": "grounded"}'
        return "Q3 automotive revenue was $45.2M, up 8% year over year."


def test_abstract_classes_enforce_implementation():
    """A subclass that forgets to implement the abstract method must fail to instantiate."""
    try:
        class BrokenChunker(Chunker):
            pass
        BrokenChunker()
        raise AssertionError("Expected TypeError - ABC did not enforce implementation!")
    except TypeError:
        print("ABC enforcement OK: incomplete subclass correctly refused to instantiate")


def test_full_pipeline():
    queries = [
        {
            "Query": "What was Q3 automotive revenue?",
            "Ground_Truths": [
                {"content": "Q3 automotive revenue was $45.2M", "page_number": 12},
                {"content": "up 8% year over year", "page_number": 12},
            ],
            "Summary_of_ground_truths": "Q3 automotive revenue was $45.2M, up 8% YoY.",
        }
    ]

    pipeline = RAGPipeline(
        name="MockFixedSize",
        parser=MockParser(),
        chunker=MockFixedSizeChunker(size=60),
        retriever=InMemoryCosineRetriever(embedder=MockEmbedder()),
    )
    pipeline.build(sources=["doc_a.pdf", "doc_b.pdf"])

    results = pipeline.evaluate(queries, llm=MockLLM(), judge_llm=MockLLM(), top_k=3)

    print("=== Summary ===")
    for k, v in results["summary"].items():
        print(f"  {k}: {v}")

    assert results["summary"]["recall_at_3"] == 1.0, "expected both facts to be found"
    assert 0 < results["summary"]["precision_at_3"] <= 1.0
    assert results["summary"]["faithfulness"] == 0.9
    assert results["summary"]["judge_errors"] == 0
    print("full pipeline test OK")


def test_swap_chunker_without_touching_anything_else():
    """The whole point: change ONE line (the chunker), rerun, done."""

    class OneBigChunkChunker(Chunker):
        """Alternative strategy: one chunk per page, no splitting at all."""
        def chunk(self, document: Document):
            return [
                Chunk(text=p.content, chunk_no=i, page_numbers=[p.page_no], source_id=document.source_id)
                for i, p in enumerate(document.pages)
            ]

    queries = [{
        "Query": "What was Q3 automotive revenue?",
        "Ground_Truths": [{"content": "Q3 automotive revenue was $45.2M", "page_number": 12}],
        "Summary_of_ground_truths": "Q3 automotive revenue was $45.2M.",
    }]

    pipeline = RAGPipeline(
        name="OneBigChunk",
        parser=MockParser(),                      # <- unchanged
        chunker=OneBigChunkChunker(),              # <- only this line changed
        retriever=InMemoryCosineRetriever(embedder=MockEmbedder()),  # <- unchanged
    )
    pipeline.build(sources=["doc_a.pdf", "doc_b.pdf"])
    results = pipeline.evaluate(queries, llm=MockLLM(), judge_llm=MockLLM(), top_k=2)

    assert results["summary"]["recall_at_2"] == 1.0
    print("chunker-swap test OK - same pipeline code, different strategy, zero other changes")


if __name__ == "__main__":
    test_abstract_classes_enforce_implementation()
    test_full_pipeline()
    test_swap_chunker_without_touching_anything_else()
    print("\nALL CHECKS PASSED")
