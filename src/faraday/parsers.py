from __future__ import annotations
from pathlib import Path
from faraday.models import Document

TEXT_EXTS = {".txt", ".md", ".markdown"}


def load_document(path: str | Path) -> Document:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif ext == ".pdf":
        text = _load_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return Document(source=path.name, text=text)


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(parts)
