from pathlib import Path
from faraday.parsers import load_document


def test_loads_text_file(tmp_path: Path):
    p = tmp_path / "note.txt"
    p.write_text("Hello Faraday", encoding="utf-8")
    doc = load_document(p)
    assert doc.source == "note.txt"
    assert doc.text == "Hello Faraday"


def test_loads_markdown_file(tmp_path: Path):
    p = tmp_path / "note.md"
    p.write_text("# Title\n\nBody", encoding="utf-8")
    doc = load_document(p)
    assert "Body" in doc.text


def test_rejects_unknown_extension(tmp_path: Path):
    p = tmp_path / "image.png"
    p.write_bytes(b"\x89PNG")
    try:
        load_document(p)
        assert False, "expected ValueError"
    except ValueError:
        pass
