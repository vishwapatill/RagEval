"""
Parser implementations. Add a new source format by subclassing Parser and
implementing .parse() -> Document - nothing downstream needs to change.
"""

from __future__ import annotations
from typing import Dict, List

from .interfaces import Parser, Document, Page


class PDFPlumberParser(Parser):
    """Vanilla text extraction, page-by-page, no layout/structure awareness.
    This is the parser your FixedSizeChunker / RecursiveChunker are meant to
    expose the weaknesses of."""

    def parse(self, source: str) -> Document:
        import pdfplumber
        pages = []
        with pdfplumber.open(source) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append(Page(page_no=i, content=text))
        return Document(source_id=source, pages=pages)


class DoclingParser(Parser):
    """Structure-aware parsing: layout, tables, reading order, heading
    hierarchy. The rich DoclingDocument is stashed in
    document.metadata["docling_document"] for DoclingHybridChunker to
    consume directly - that's where the actual structure-awareness lives,
    not in the flattened `pages` text below (which exists mainly so this
    Document is still usable if you ever want to test Docling-parse +
    vanilla-chunk as a fourth combination).

    Needs the full `docling` package (DocumentConverter), not just
    docling-core.
    """

    def __init__(self):
        from docling.document_converter import DocumentConverter
        self.converter = DocumentConverter()

    def parse(self, source: str) -> Document:
        dl_doc = self.converter.convert(source).document

        # Best-effort reconstruction of per-page plain text, grouping every
        # text item by its provenance page number. Verified against
        # docling-core's DoclingDocument.iterate_items() / TextItem.prov /
        # ProvenanceItem.page_no API.
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
