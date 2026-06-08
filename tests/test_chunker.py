from faraday.models import Document
from faraday.chunker import chunk_document


def test_short_doc_is_single_chunk():
    doc = Document(source="a.txt", text="one two three")
    chunks = chunk_document(doc, size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].ord == 0
    assert chunks[0].source == "a.txt"
    assert chunks[0].text == "one two three"


def test_long_doc_splits_with_overlap():
    text = "x" * 250
    doc = Document(source="a.txt", text=text)
    chunks = chunk_document(doc, size=100, overlap=20)
    assert len(chunks) == 3            # 0-100, 80-180, 160-250
    assert [c.ord for c in chunks] == [0, 1, 2]
    # overlap: chunk 1 starts 20 chars before chunk 0 ended
    assert chunks[1].text[:20] == chunks[0].text[-20:]


def test_doc_ids_are_stable_for_same_source():
    doc = Document(source="a.txt", text="hello")
    a = chunk_document(doc, size=100, overlap=10)
    b = chunk_document(doc, size=100, overlap=10)
    assert a[0].doc_id == b[0].doc_id
