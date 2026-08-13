
from typing import Any, List
from rag_chunking_evaluator import Chunk


class FixedSizeChunkerAdapter:
    """document = {"pages": [{"text": str, "page_number": int}, ...]}"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: dict) -> List[Chunk]:
        chunks = []
        for page in document["pages"]:
            text, page_num = page["text"], page["page_number"]
            step = self.chunk_size - self.overlap
            for i in range(0, len(text), step):
                piece = text[i:i + self.chunk_size]
                if piece.strip():
                    chunks.append(Chunk(text=piece, page_numbers=[page_num]))
        return chunks


class RecursiveChunkerAdapter:
    """document = {"pages": [{"text": str, "page_number": int}, ...]}"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )

    def chunk(self, document: dict) -> List[Chunk]:
        chunks = []
        for page in document["pages"]:
            for piece in self.splitter.split_text(page["text"]):
                chunks.append(Chunk(text=piece, page_numbers=[page["page_number"]]))
        return chunks


class DoclingChunkerAdapter:
    """document = path to the PDF file (str). Docling parses AND chunks in
    one call here, using its own structure-aware HybridChunker."""

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", max_tokens: int = 512):
        from docling.document_converter import DocumentConverter
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from transformers import AutoTokenizer

        self.converter = DocumentConverter()
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(embedding_model_name),
            max_tokens=max_tokens,
        )
        self.chunker = HybridChunker(tokenizer=tokenizer)

    def chunk(self, document: str) -> List[Chunk]:
        dl_doc = self.converter.convert(document).document
        chunks = []
        for dl_chunk in self.chunker.chunk(dl_doc=dl_doc):
            # contextualize() prepends the heading hierarchy - this is the
            # feature you specifically want to be testing
            text = self.chunker.contextualize(chunk=dl_chunk)
            pages = sorted({
                prov.page_no
                for item in dl_chunk.meta.doc_items
                for prov in item.prov
            }) if dl_chunk.meta.doc_items else []
            chunks.append(Chunk(text=text, page_numbers=pages, metadata={"headings": dl_chunk.meta.headings}))
        return chunks


# ============================================================================
# 2. EMBEDDER  - LangChain embedding classes already satisfy the interface
#    RAGEvaluator needs (.embed_documents / .embed_query), no adapter needed.
# ============================================================================

def build_embedder():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Runs fine on your 4050 or even CPU for this scale - see earlier discussion.


# ============================================================================
# 3. LLM ADAPTER  - wraps Gemma (generation) / Gemini (judge) behind .invoke()
# ============================================================================

class GoogleGenAILLM:
    """Minimal .invoke()-compatible wrapper around the google-genai SDK.
    Use one instance with model="gemma-4-12b-it" for generation and a
    SEPARATE instance with model="gemini-2.5-flash" for judging, so the
    judge isn't grading its own generator (see earlier self-preference-bias note)."""

    def __init__(self, model: str, api_key: str):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def invoke(self, prompt: str) -> str:
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return response.text


# ============================================================================
# 4. PUTTING IT TOGETHER - one pipeline run
# ============================================================================

def run_one_pipeline(pipeline_name: str, chunker, documents: list, queries: list, api_key: str):
    from rag_chunking_evaluator import RAGEvaluator

    evaluator = RAGEvaluator(
        chunker=chunker,
        embedder=build_embedder(),
        llm=GoogleGenAILLM(model="gemma-4-12b-it", api_key=api_key),
        judge_llm=GoogleGenAILLM(model="gemini-2.5-flash", api_key=api_key),
        top_k=5,
    )
    evaluator.index(documents)
    results = evaluator.evaluate(queries, run_generation=True)
    print(f"\n=== {pipeline_name} ===")
    for k, v in results["summary"].items():
        print(f"  {k}: {v}")
    return results


if __name__ == "__main__":
    # Example skeleton - replace with your actual parsed pages / PDF paths / query truth table
    API_KEY = "YOUR_AI_STUDIO_KEY"

    pages_doc = {"pages": [{"text": "...vanilla-extracted text per page...", "page_number": 1}]}
    docling_doc_path = "your_file.pdf"

    queries = [
        {
            "Query": "example question",
            "Ground_Truths": [{"content": "exact fact from the pdf", "page_number": 1}],
            "Summary_of_ground_truths": "one-line reference answer",
        }
    ]

    from rag_chunking_evaluator import compare_pipelines

    all_results = {
        "Fixed-size": run_one_pipeline("Fixed-size", FixedSizeChunkerAdapter(), [pages_doc], queries, API_KEY),
        "Recursive": run_one_pipeline("Recursive", RecursiveChunkerAdapter(), [pages_doc], queries, API_KEY),
        "Docling": run_one_pipeline("Docling", DoclingChunkerAdapter(), [docling_doc_path], queries, API_KEY),
    }

    print("\n=== Comparison table ===")
    print(compare_pipelines(all_results))
