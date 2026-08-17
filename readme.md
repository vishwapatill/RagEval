# RagParserEval

A modular framework to evaluate how **document parsing** affects **RAG retrieval quality**.

Most RAG evaluations test the final generated answer — mixing retrieval quality with LLM quality. RagParserEval isolates the retrieval stage: given a query and ground-truth answer, did the retriever pull back chunks that contain the right information?

The core idea is simple: swap the parser, keep everything else identical, and measure what changes.

```
PDF ──→ Parser ──→ Chunker ──→ Embedder ──→ Retriever ──→ LLM Judge
         ▲                                                    │
     (swap this)                                        PASS / FAIL
```

## What it does

- **Parse** a PDF with any supported parser (PDFPlumber, Docling, Marker, PyMuPDF4LLM, MinerU, Unstructured, Docling VLM)
- **Chunk** the parsed output with pluggable chunkers (fixed-size, recursive, Docling HybridChunker)
- **Embed** chunks and queries with HuggingFace sentence-transformers
- **Retrieve** top-k chunks via cosine similarity
- **Judge** retrieval quality with a local LLM (Ollama) or cloud LLM (Gemini) using structured Pydantic output — returns `PASS` or `FAIL` per query with per-chunk relevance tagging
- **Compare** pipelines side by side with precision@k, recall@k, and hit rate

## Project structure

```
rag_eval/
├── interfaces.py        # Abstract base classes (Parser, Chunker, Embedder, Retriever, LLM)
├── parsers.py           # 7 parser implementations across 4 tiers
├── chunkers.py          # FixedSizeChunker, RecursiveChunker, DoclingHybridChunker
├── embedders.py         # HuggingFace sentence-transformer embedder
├── llms.py              # GoogleGenAILLM, OllamaLLM (both with structured output)
├── retrieval_evaluator.py  # Lexical eval (precision, recall, MRR) + LLM judge
├── metrics.py           # recall@k, precision@k, reciprocal rank, page hit
└── utils.py             # Retry logic, JSON parsing helpers

data_set/                # Ground truth from open_ragbench (see below)
├── queries.json
├── answers.json
├── qrels.json
└── pdf_urls.json

pdfs/                    # Downloaded PDFs (not committed)
test_file.ipynb          # Full evaluation notebook: PDFPlumber vs Docling
```

## Quick start

### 1. Install

```bash
git clone https://github.com/vishwapatill/RagParserEval.git
cd RagParserEval
pip install -r requirement.txt
```

For GPU acceleration on Windows, disable torch.compile before running anything:

```python
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
```

### 2. Pick a parser and run

```python
from rag_eval.parsers import get_parser
from rag_eval.chunkers import RecursiveChunker
from rag_eval.embedders import HuggingFaceEmbedder

parser = get_parser("docling", use_gpu=True)
chunker = RecursiveChunker()
embedder = HuggingFaceEmbedder()

doc = parser.parse("pdfs/2401.03305v2.pdf")
chunks = chunker.chunk(doc)
embeddings = [embedder.embed_query(c.text) for c in chunks]
```

### 3. Evaluate with LLM judge

```python
from rag_eval.llms import OllamaLLM, RetrievalJudgement

judge = OllamaLLM(model="qwen2.5:3b", temperature=0.0)

verdict = judge.invoke_structured(
    prompt="Evaluate the following RAG retrieval...",
    response_model=RetrievalJudgement,
    system_prompt=RETRIEVAL_JUDGE_SYSTEM_PROMPT,
)

print(verdict.result)  # "PASS" or "FAIL"
print(verdict.reason)  # "The retrieved chunks contain..."
```

### 4. Compare pipelines

```python
from rag_eval.retrieval_evaluator import compare_retrievers

df = compare_retrievers({
    "PDFPlumber + Recursive": pdfplumber_results,
    "Docling + Recursive": docling_results,
})
```

## Supported parsers

| Parser | Type | License | GPU needed |
|---|---|---|---|
| PDFPlumber | Rule-based | MIT | No |
| PyMuPDF4LLM | Rule-based + layout | AGPL | No |
| Unstructured | Hybrid (rules + YOLOX) | Apache-2.0 | Optional |
| Marker | Pipeline (Surya VLM) | GPL-3.0 | Recommended |
| MinerU | Pipeline (DocLayout-YOLO) | Apache-2.0* | Recommended |
| Docling | Pipeline (RT-DETRv2) | MIT | Optional |
| Docling VLM | End-to-end VLM (SmolDocling) | Apache-2.0 | Yes |

Install only the parsers you need — each has its own dependency tree.

## Ground truth dataset

Evaluation queries and answers come from [Vectara's open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench/tree/main/pdf/arxiv) (CC-BY-NC-4.0), which provides query-answer-document mappings over arXiv PDFs.

The `data_set/` folder contains:

- `queries.json` — questions with type and source metadata
- `answers.json` — ground-truth answers per query
- `qrels.json` — maps each query to its source document
- `pdf_urls.json` — download URLs for the source PDFs

## LLM judge

The evaluation uses a local LLM (via Ollama) or a cloud LLM (via Google GenAI) as a retrieval judge. The judge receives the query, ground-truth answer, and retrieved chunks, then returns a structured `PASS`/`FAIL` verdict with per-chunk relevance tagging.

Both LLM classes support Pydantic structured output — the model is constrained to emit valid JSON matching the schema, so there are no parsing failures or markdown fence issues.

## Adding a new parser

Subclass `Parser` and implement `.parse() -> Document`. Everything downstream works automatically:

```python
from rag_eval.interfaces import Parser, Document, Page

class MyParser(Parser):
    def parse(self, source: str) -> Document:
        # your parsing logic
        return Document(source_id=source, pages=[Page(page_no=1, content="...")])
```

The same pattern applies to chunkers, embedders, and LLMs — all are pluggable via abstract base classes.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally (for the LLM judge)
- NVIDIA GPU recommended for Docling/Marker/MinerU parsers

## License

MIT

## Acknowledgements

- [open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench) by Vectara for the evaluation dataset
- [Docling](https://github.com/docling-project/docling) by IBM Research
- [Marker](https://github.com/VikParuchuri/marker) by Datalab
- [Ollama](https://ollama.com/) for local LLM inference