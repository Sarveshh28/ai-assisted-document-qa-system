from src.pipeline.chunking import chunk_document


def test_chunk_document_returns_chunks():
    text = (
        "Sentence one explains the topic. "
        "Sentence two expands the context. "
        "Sentence three adds another idea. "
        "Sentence four closes the section."
    )
    chunks = chunk_document(
        doc_id="doc",
        title="Doc",
        text=text,
        chunk_size=8,
        overlap=2,
    )
    assert len(chunks) >= 2
    assert chunks[0].doc_id == "doc"
    assert chunks[0].title == "Doc"
