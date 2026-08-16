"""
Parser implementations. Add a new source format by subclassing Parser and
implementing .parse() -> Document - nothing downstream needs to change.

Parsers included (ordered roughly by sophistication):
─────────────────────────────────────────────────────
 1. PDFPlumberParser      – rule-based, no layout awareness
 2. PyMuPDF4LLMParser     – rule-based + layout module, Markdown-native
 3. UnstructuredParser     – hybrid rule-based + AI layout detection
 4. MarkerParser          – pipeline (Surya VLM backbone), GPU recommended
 5. MinerUParser          – pipeline + optional VLM, GPU recommended
 6. DoclingParser         – pipeline (RT-DETRv2 + TableFormer), GPU optional
 7. DoclingVLMParser      – single VLM end-to-end (SmolDocling / Granite)

Install only the parsers you need — each has its own dependency tree.
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import Dict, List

from .interfaces import Parser, Document, Page


# ═══════════════════════════════════════════════════════════════════════════
# TIER 1 — Rule-Based / Text-Layer Extraction
# ═══════════════════════════════════════════════════════════════════════════


class PDFPlumberParser(Parser):
    """Vanilla text extraction, page-by-page, no layout/structure awareness.
    This is the parser your FixedSizeChunker / RecursiveChunker are meant to
    expose the weaknesses of.

    pip install pdfplumber
    """

    def parse(self, source: str) -> Document:
        import pdfplumber

        pages = []
        with pdfplumber.open(source) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append(Page(page_no=i, content=text))
        return Document(source_id=source, pages=pages)


class PyMuPDF4LLMParser(Parser):
    """Layout-aware text extraction via PyMuPDF's layout module, outputting
    Markdown with header detection, multi-column reading order, and basic
    table support.  No GPU needed — runs on CPU only.

    Stronger than pdfplumber because it respects multi-column layouts and
    detects headers via font-size heuristics, but still purely rule-based
    (no ML models).

    pip install pymupdf4llm
    """

    def __init__(self, *, write_images: bool = False):
        self.write_images = write_images

    def parse(self, source: str) -> Document:
        import pymupdf4llm

        # page_chunks=True returns a list of dicts, one per page, each
        # containing 'text' (markdown), 'metadata', 'tables', 'images', etc.
        page_chunks = pymupdf4llm.to_markdown(
            source,
            page_chunks=True,
            write_images=self.write_images,
        )

        pages = []
        for chunk in page_chunks:
            page_no = chunk["metadata"]["page"] + 1   # 0-based -> 1-based
            pages.append(Page(page_no=page_no, content=chunk["text"]))

        return Document(
            source_id=source,
            pages=pages,
            metadata={"page_chunks": page_chunks},
        )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 2 — Hybrid (Rule-Based + AI Layout Detection)
# ═══════════════════════════════════════════════════════════════════════════


class UnstructuredParser(Parser):
    """Hybrid parsing via Unstructured — rule-based extraction augmented
    with AI-based layout detection (hi_res strategy uses detectron2 /
    YOLOX for element classification).  Returns semantically labelled
    elements (Title, NarrativeText, Table, ListItem, etc.).

    pip install "unstructured[pdf]"
    # For hi_res strategy, also: pip install "unstructured[all-docs]"
    """

    def __init__(self, *, strategy: str = "hi_res"):
        """
        strategy: "fast"    – text-layer only, no models (like pdfplumber)
                  "hi_res" – layout detection + OCR (best quality, slower)
                  "auto"   – picks fast for born-digital, hi_res for scans
        """
        self.strategy = strategy

    def parse(self, source: str) -> Document:
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            filename=source,
            strategy=self.strategy,
        )

        # Group elements by page number
        pages_dict: Dict[int, List[str]] = {}
        for el in elements:
            page_no = el.metadata.page_number or 1
            # Prefix titles/headers with markdown heading markers
            category = el.category  # Title, NarrativeText, Table, etc.
            text = str(el)
            if category == "Title":
                text = f"## {text}"
            pages_dict.setdefault(page_no, []).append(text)

        pages = [
            Page(page_no=pn, content="\n\n".join(texts))
            for pn, texts in sorted(pages_dict.items())
        ]

        return Document(
            source_id=source,
            pages=pages,
            metadata={
                "elements": elements,     # raw element objects
                "strategy": self.strategy,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 3 — Pipeline-Based (Layout + OCR + Table Models)
# ═══════════════════════════════════════════════════════════════════════════


class MarkerParser(Parser):
    """Pipeline-based parsing via Marker (Datalab).  Uses the Surya VLM
    suite under the hood: layout detection → OCR → table/math recognition
    → Markdown assembly.

    GPU strongly recommended (RTX 4050 works).  CPU is functional but
    very slow (~1-5 min/page).

    pip install marker-pdf
    # On Windows, also apply the torch dynamo fix:
    # os.environ["TORCHDYNAMO_DISABLE"] = "1"

    License: GPL-3.0 (code) + RAIL-M (model weights).
    """

    def __init__(self, *, use_llm: bool = False):
        """
        use_llm: if True, enables the optional LLM refinement pass that
                 improves tables, math, and forms at the cost of extra
                 latency.  Requires a model endpoint configured in Marker.
        """
        self.use_llm = use_llm

    def parse(self, source: str) -> Document:
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        models = create_model_dict()
        converter = PdfConverter(
            artifact_dict=models,
            config={"use_llm": self.use_llm},
        )
        result = converter(source)

        # result.markdown is the full document; result.pages has per-page
        # markdown if available.  Fall back to splitting on page markers.
        markdown = result.markdown

        # Marker inserts page-break markers: "\n\n---\n\n" between pages.
        raw_pages = re.split(r"\n{2,}---\n{2,}", markdown)

        pages = [
            Page(page_no=i, content=text.strip())
            for i, text in enumerate(raw_pages, start=1)
            if text.strip()
        ]

        return Document(
            source_id=source,
            pages=pages,
            metadata={
                "marker_metadata": getattr(result, "metadata", {}),
                "full_markdown": markdown,
            },
        )


class MinerUParser(Parser):
    """Pipeline-based parsing via MinerU (OpenDataLab).  Strong on tables,
    formulas, CJK documents.  MinerU v3+ uses a CLI/API architecture —
    this parser shells out to the `mineru` CLI and reads the output
    markdown, which is the most reliable cross-platform approach.

    GPU recommended for VLM/hybrid backends; pipeline backend runs on CPU.

    pip install "mineru[all]"
    # Then download models: mineru-models-download

    License: MinerU Open Source License (Apache-2.0 based, with conditions
    above 100M MAU or $20M monthly revenue).
    """

    def __init__(self, *, backend: str = "pipeline"):
        """
        backend: "pipeline"  – CPU-friendly, no hallucination, fast
                 "vlm"       – VLM-based, highest accuracy, needs GPU
                 "hybrid"    – native text extraction + VLM for hard parts
        """
        self.backend = backend

    def parse(self, source: str) -> Document:
        import subprocess
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                "mineru",
                source,
                tmpdir,
                "--backend", self.backend,
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # MinerU writes output to <tmpdir>/<pdf_stem>/<pdf_stem>.md
            pdf_stem = os.path.splitext(os.path.basename(source))[0]
            md_dir = os.path.join(tmpdir, pdf_stem)

            md_path = os.path.join(md_dir, f"{pdf_stem}.md")
            if not os.path.exists(md_path):
                # Fallback: find first .md file
                for f in os.listdir(md_dir):
                    if f.endswith(".md"):
                        md_path = os.path.join(md_dir, f)
                        break

            with open(md_path, "r", encoding="utf-8") as f:
                full_markdown = f.read()

            # Try loading the content_list.json for page-level info
            content_json = os.path.join(md_dir, f"{pdf_stem}_content_list.json")
            page_map: Dict[int, List[str]] = {}

            if os.path.exists(content_json):
                with open(content_json, "r", encoding="utf-8") as f:
                    content_list = json.load(f)
                for item in content_list:
                    pg = item.get("page_idx", 0) + 1   # 0-based -> 1-based
                    text = item.get("text", "")
                    if text:
                        page_map.setdefault(pg, []).append(text)
            else:
                # Fallback: treat entire markdown as page 1
                page_map[1] = [full_markdown]

            pages = [
                Page(page_no=pn, content="\n".join(texts))
                for pn, texts in sorted(page_map.items())
            ]

        return Document(
            source_id=source,
            pages=pages,
            metadata={
                "backend": self.backend,
                "full_markdown": full_markdown,
            },
        )


class DoclingParser(Parser):
    """Structure-aware parsing: layout, tables, reading order, heading
    hierarchy.  The rich DoclingDocument is stashed in
    document.metadata["docling_document"] for DoclingHybridChunker to
    consume directly — that's where the actual structure-awareness lives,
    not in the flattened `pages` text below (which exists mainly so this
    Document is still usable if you ever want to test Docling-parse +
    vanilla-chunk as a fourth combination).

    Needs the full `docling` package (DocumentConverter), not just
    docling-core.

    pip install docling

    License: MIT (code); model licenses tracked separately.
    """

    def __init__(self, *, use_gpu: bool = True):
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.accelerator_options import (
            AcceleratorOptions,
            AcceleratorDevice,
        )
        from docling.datamodel.base_models import InputFormat

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.CUDA if use_gpu else AcceleratorDevice.CPU,
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

    def parse(self, source: str) -> Document:
        dl_doc = self.converter.convert(source).document

        # Best-effort reconstruction of per-page plain text, grouping every
        # text item by its provenance page number.
        pages_dict: Dict[int, List[str]] = {}
        for item, _level in dl_doc.iterate_items():
            text = getattr(item, "text", None)
            if not text:
                continue
            for prov in getattr(item, "prov", []):
                pages_dict.setdefault(prov.page_no, []).append(text)

        pages = [
            Page(page_no=pn, content="\n".join(texts))
            for pn, texts in sorted(pages_dict.items())
        ]

        return Document(
            source_id=source,
            pages=pages,
            metadata={"docling_document": dl_doc},
        )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 4 — VLM-Native (Single Model, End-to-End)
# ═══════════════════════════════════════════════════════════════════════════


class DoclingVLMParser(Parser):
    """End-to-end VLM parsing via Docling's VlmPipeline.  A single vision-
    language model (default: SmolDocling / Granite-Docling-258M) looks at
    each page image and emits structured DocTags markup — no multi-stage
    pipeline, so no error cascading between layout → OCR → table stages.

    GPU required.  Your RTX 4050 handles this fine.

    pip install docling

    License: Apache-2.0 (SmolDocling model).
    """

    def __init__(self, *, model_spec: str = "SMOLDOCLING_TRANSFORMERS"):
        """
        model_spec: key from docling.datamodel.vlm_model_specs, e.g.
                    "SMOLDOCLING_TRANSFORMERS"  – local GPU inference
                    "SMOLDOCLING_MLX"           – Apple Silicon (MLX)
                    or configure a remote endpoint via vlm_model_specs.
        """
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        self.model_spec = model_spec

    def parse(self, source: str) -> Document:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.datamodel import vlm_model_specs
        from docling.datamodel.base_models import InputFormat

        spec = getattr(vlm_model_specs, self.model_spec)

        pipeline_options = VlmPipelineOptions(vlm_options=spec)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )

        dl_doc = converter.convert(source).document

        # Same page-grouping logic as DoclingParser
        pages_dict: Dict[int, List[str]] = {}
        for item, _level in dl_doc.iterate_items():
            text = getattr(item, "text", None)
            if not text:
                continue
            for prov in getattr(item, "prov", []):
                pages_dict.setdefault(prov.page_no, []).append(text)

        pages = [
            Page(page_no=pn, content="\n".join(texts))
            for pn, texts in sorted(pages_dict.items())
        ]

        return Document(
            source_id=source,
            pages=pages,
            metadata={
                "docling_document": dl_doc,
                "vlm_model": self.model_spec,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# Registry — convenience mapping for config-driven instantiation
# ═══════════════════════════════════════════════════════════════════════════

PARSER_REGISTRY: Dict[str, type] = {
    "pdfplumber":    PDFPlumberParser,
    "pymupdf4llm":   PyMuPDF4LLMParser,
    "unstructured":  UnstructuredParser,
    "marker":        MarkerParser,
    "mineru":        MinerUParser,
    "docling":       DoclingParser,
    "docling_vlm":   DoclingVLMParser,
}


def get_parser(name: str, **kwargs) -> Parser:
    """Instantiate a parser by name.

    Usage:
        parser = get_parser("marker", use_llm=True)
        doc = parser.parse("paper.pdf")
    """
    cls = PARSER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown parser '{name}'. "
            f"Available: {', '.join(PARSER_REGISTRY)}"
        )
    return cls(**kwargs)