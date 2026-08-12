"""Shared helpers - fact matching, cosine similarity, retry wrapper, JSON parsing."""

from __future__ import annotations
import json
import re
import time
from typing import Any, List

import numpy as np
from rapidfuzz import fuzz


def extract_text(response: Any) -> str:
    """LangChain-style responses often carry .content; plain strings pass through."""
    if hasattr(response, "content"):
        return response.content
    return str(response)


def invoke_with_retry(llm, prompt: str, max_retries: int = 3, base_delay: float = 2.0) -> str:
    """Wraps any LLM.invoke() with exponential backoff - free-tier rate limits
    (Gemma/Gemini) get hit hard once you're running many queries x 3 pipelines."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return extract_text(llm.invoke(prompt))
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is a retry wrapper
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(np.dot(a_arr, b_arr) / denom) if denom else 0.0


def is_fact_in_text(fact: str, text: str, fuzzy_threshold: int = 85) -> bool:
    """Strict on numbers (45.2 vs 54.2 is NOT a fuzzy match, and 8 vs 18 is
    NOT a substring match), fuzzy on surrounding prose (a markdown table, a
    raw text dump, and a reformatted LLM answer all describe the same fact
    differently - token_set_ratio tolerates reordering and extra words,
    which plain partial_ratio does not)."""
    fact_norm = fact.lower().strip()
    text_norm = text.lower()

    numbers = re.findall(r"\d[\d,\.]*", fact_norm)
    if numbers:
        clean_numbers = [n.replace(",", "").rstrip(".") for n in numbers]
        clean_text = text_norm.replace(",", "")
        for n in clean_numbers:
            if not n:
                continue
            # word-boundary check: "8" must not match inside "18" or "2018"
            if not re.search(rf"(?<!\d){re.escape(n)}(?!\d)", clean_text):
                return False

    return fuzz.token_set_ratio(fact_norm, text_norm) >= fuzzy_threshold


def parse_json_response(raw: str) -> dict:
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise