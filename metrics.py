"""
Retrieval metrics. Pure functions - work on any List[Chunk] and ground-truth
list, regardless of which pipeline produced the chunks. Ground truth is
anchored to source content + page number, never to a chunk ID, which is what
makes these comparable across pipelines that chunk completely differently.
"""

from __future__ import annotations
from typing import List, Optional

from .interfaces import Chunk
from .utils import is_fact_in_text


def recall_at_k(retrieved: List[Chunk], ground_truths: List[dict], fuzzy_threshold: int = 85) -> Optional[float]:
    """Of all required facts, what fraction appear anywhere across the top-k chunks."""
    if not ground_truths:
        return None
    combined_text = " ".join(c.text for c in retrieved)
    found = sum(1 for gt in ground_truths if is_fact_in_text(gt["content"], combined_text, fuzzy_threshold))
    return found / len(ground_truths)


def precision_at_k(retrieved: List[Chunk], ground_truths: List[dict], fuzzy_threshold: int = 85) -> float:
    """Of the k retrieved chunks, what fraction contain at least one required fact."""
    if not retrieved:
        return 0.0
    relevant = sum(
        1 for c in retrieved
        if any(is_fact_in_text(gt["content"], c.text, fuzzy_threshold) for gt in ground_truths)
    )
    return relevant / len(retrieved)


def reciprocal_rank(retrieved: List[Chunk], ground_truths: List[dict], fuzzy_threshold: int = 85) -> float:
    """1/rank of the first chunk containing any required fact. 0 if none do."""
    for i, c in enumerate(retrieved):
        if any(is_fact_in_text(gt["content"], c.text, fuzzy_threshold) for gt in ground_truths):
            return 1.0 / (i + 1)
    return 0.0


def page_hit(retrieved: List[Chunk], ground_truths: List[dict]) -> Optional[bool]:
    """Diagnostic, not a scoring metric: did retrieval land on the right page
    at all? Cross-reference with recall to separate chunking failures (right
    page, missed fact) from retrieval/embedding failures (wrong page)."""
    gt_pages = {gt["page_number"] for gt in ground_truths if gt.get("page_number") is not None}
    if not gt_pages:
        return None
    retrieved_pages = set()
    for c in retrieved:
        retrieved_pages.update(c.page_numbers)
    return bool(gt_pages & retrieved_pages)
