"""
RAGPipeline - convenience composer showing the intended usage pattern:
swap the `parser` or `chunker` argument to test a different combination,
everything else in the run_all_pipelines() workflow stays identical.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .interfaces import Parser, Chunker, Retriever, LLM
from .evaluator import Evaluator, compare_pipelines


@dataclass
class RAGPipeline:
    name: str
    parser: Parser
    chunker: Chunker
    retriever: Retriever  # construct with your chosen Embedder, not yet indexed

    def build(self, sources: List[str]) -> None:
        """Parse + chunk every source, index all resulting chunks into the retriever."""
        chunks = []
        for source in sources:
            document = self.parser.parse(source)
            chunks.extend(self.chunker.chunk(document))
        self.retriever.index(chunks)

    def evaluate(
        self,
        queries: List[dict],
        llm: LLM,
        judge_llm: Optional[LLM] = None,
        top_k: int = 5,
        run_generation: bool = True,
    ) -> Dict[str, Any]:
        evaluator = Evaluator(self.retriever, llm, judge_llm, top_k)
        return evaluator.evaluate(queries, run_generation)


def run_all_pipelines(
    pipelines: List[RAGPipeline],
    sources_by_pipeline: Dict[str, List[str]],
    queries: List[dict],
    llm: LLM,
    judge_llm: Optional[LLM] = None,
    top_k: int = 5,
):
    """Builds and evaluates every pipeline, returns a compare_pipelines() DataFrame.
    sources_by_pipeline lets Docling get raw PDF paths while vanilla pipelines
    get the same paths too - both Parsers take a file path, they just do
    different things with it internally."""
    all_results = {}
    for pipeline in pipelines:
        pipeline.build(sources_by_pipeline[pipeline.name])
        all_results[pipeline.name] = pipeline.evaluate(queries, llm, judge_llm, top_k)
    return compare_pipelines(all_results), all_results
