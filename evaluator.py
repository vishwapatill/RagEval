"""
Evaluator - the only class that knows about queries and ground truth.
Takes an already-indexed Retriever plus an LLM (and optionally a separate
judge LLM), and turns a list of queries into retrieval + generation metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .interfaces import Retriever, LLM, Chunk
from .metrics import recall_at_k, precision_at_k, reciprocal_rank, page_hit
from .utils import invoke_with_retry, parse_json_response


_GEN_PROMPT_TEMPLATE = """Answer the question using ONLY the context below. \
If the answer is not present in the context, say "Not found in context." \
Do not use outside knowledge.

Context:
{context}

Question: {query}

Answer:"""

_JUDGE_PROMPT_TEMPLATE = """You are scoring a RAG system's output. Be strict and base every \
score only on the material given below - do not reward answers that happen to be \
correct from outside knowledge.

Question: {query}

Retrieved Context:
{context}

Generated Answer:
{answer}

Reference Summary (what a correct answer should cover):
{summary}

Score each from 0.0 to 1.0:
1. faithfulness: Is the generated answer fully supported by the retrieved context, with no hallucinated facts?
2. answer_relevancy: Does the generated answer actually address the question asked?
3. context_adequacy: Does the retrieved context contain enough information to produce the reference summary?

Respond with ONLY a JSON object, no markdown fences, no other text:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_adequacy": 0.0, "reasoning": "one sentence"}}"""


@dataclass
class QueryResult:
    query: str
    retrieved_chunks: List[Chunk]
    recall_at_k: Optional[float]
    precision_at_k: Optional[float]
    reciprocal_rank: Optional[float]
    page_hit: Optional[bool]
    generated_answer: Optional[str] = None
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_adequacy: Optional[float] = None
    judge_reasoning: Optional[str] = None
    judge_error: Optional[str] = None


class Evaluator:
    """
    Usage:
        evaluator = Evaluator(retriever=my_already_indexed_retriever,
                               llm=gemma, judge_llm=gemini_flash, top_k=5)
        results = evaluator.evaluate(queries)
        print(results["summary"])
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        judge_llm: Optional[LLM] = None,
        top_k: int = 5,
        fuzzy_threshold: int = 85,
    ):
        self.retriever = retriever
        self.llm = llm
        self.judge_llm = judge_llm or llm  # pass a different model to avoid self-preference bias
        self.top_k = top_k
        self.fuzzy_threshold = fuzzy_threshold

    def generate_answer(self, query: str, retrieved: List[Chunk]) -> str:
        context = "\n\n".join(c.text for c in retrieved)
        prompt = _GEN_PROMPT_TEMPLATE.format(context=context, query=query)
        return invoke_with_retry(self.llm, prompt)

    def judge(self, query: str, context: str, answer: str, summary: str) -> dict:
        prompt = _JUDGE_PROMPT_TEMPLATE.format(query=query, context=context, answer=answer, summary=summary)
        raw = invoke_with_retry(self.judge_llm, prompt)
        return parse_json_response(raw)

    def evaluate(self, queries: List[dict], run_generation: bool = True) -> Dict[str, Any]:
        """
        queries: [{"Query": str,
                   "Ground_Truths": [{"content": str, "page_number": int}],
                   "Summary_of_ground_truths": str}, ...]
        run_generation=False skips all LLM calls, giving retrieval-only
        metrics fast and free - use this as a first pass before spending API
        quota on generation + judging.
        """
        results: List[QueryResult] = []

        for q in queries:
            query_text = q["Query"]
            ground_truths = q["Ground_Truths"]
            summary = q.get("Summary_of_ground_truths", "")

            retrieved = self.retriever.retrieve(query_text, self.top_k)

            qr = QueryResult(
                query=query_text,
                retrieved_chunks=retrieved,
                recall_at_k=recall_at_k(retrieved, ground_truths, self.fuzzy_threshold),
                precision_at_k=precision_at_k(retrieved, ground_truths, self.fuzzy_threshold),
                reciprocal_rank=reciprocal_rank(retrieved, ground_truths, self.fuzzy_threshold),
                page_hit=page_hit(retrieved, ground_truths),
            )

            if run_generation:
                context = "\n\n".join(c.text for c in retrieved)
                try:
                    qr.generated_answer = self.generate_answer(query_text, retrieved)
                    judge_result = self.judge(query_text, context, qr.generated_answer, summary)
                    qr.faithfulness = judge_result.get("faithfulness")
                    qr.answer_relevancy = judge_result.get("answer_relevancy")
                    qr.context_adequacy = judge_result.get("context_adequacy")
                    qr.judge_reasoning = judge_result.get("reasoning")
                except Exception as e:  # noqa: BLE001
                    qr.judge_error = str(e)

            results.append(qr)

        return {"summary": self._aggregate(results), "per_query": results}

    @staticmethod
    def _avg(values: List[Optional[float]]) -> Optional[float]:
        clean = [v for v in values if v is not None]
        return sum(clean) / len(clean) if clean else None

    def _aggregate(self, results: List[QueryResult]) -> Dict[str, Any]:
        recall = self._avg([r.recall_at_k for r in results])
        precision = self._avg([r.precision_at_k for r in results])
        f1 = None
        if recall is not None and precision is not None and (recall + precision) > 0:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "num_queries": len(results),
            f"recall_at_{self.top_k}": recall,
            f"precision_at_{self.top_k}": precision,
            f"f1_at_{self.top_k}": f1,
            "mrr": self._avg([r.reciprocal_rank for r in results]),
            "page_hit_rate": self._avg([1.0 if r.page_hit else 0.0 for r in results if r.page_hit is not None]),
            "faithfulness": self._avg([r.faithfulness for r in results]),
            "answer_relevancy": self._avg([r.answer_relevancy for r in results]),
            "context_adequacy": self._avg([r.context_adequacy for r in results]),
            "judge_errors": sum(1 for r in results if r.judge_error is not None),
        }


def compare_pipelines(results_by_pipeline: Dict[str, Dict[str, Any]]):
    """results_by_pipeline: {"Fixed-size": evaluator.evaluate(...) result, ...}
    Returns a pandas DataFrame, one row per pipeline."""
    import pandas as pd
    rows = {name: res["summary"] for name, res in results_by_pipeline.items()}
    return pd.DataFrame(rows).T
