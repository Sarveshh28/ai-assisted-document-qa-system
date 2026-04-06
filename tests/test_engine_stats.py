from src.pipeline.engine import ResearchAssistantEngine


class DummyRetriever:
    def __init__(self) -> None:
        self.calls = []

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        self.calls.append((question, top_k))
        return [
            {
                "doc_id": "rag_doc",
                "title": "RAG",
                "text": "Retrieval augmented generation improves factual grounding.",
                "hybrid_score": 0.9,
                "matched_terms": ["retrieval", "generation"],
            }
        ]


def test_stats_uses_loaded_payload_without_index_files():
    engine = ResearchAssistantEngine.__new__(ResearchAssistantEngine)
    engine.metadata = {"project": "demo"}
    engine.documents = [
        {"doc_id": "rag_doc", "title": "RAG", "word_count": 100},
        {"doc_id": "cv_doc", "title": "CV", "word_count": 200},
    ]
    engine.chunks = [{"chunk_id": "1"}, {"chunk_id": "2"}, {"chunk_id": "3"}]

    stats = engine.stats()

    assert stats["document_count"] == 2
    assert stats["chunk_count"] == 3
    assert stats["average_document_words"] == 150.0


def test_ask_returns_confidence_and_terms():
    engine = ResearchAssistantEngine.__new__(ResearchAssistantEngine)
    engine.retriever = DummyRetriever()

    result = engine.ask("How does retrieval generation work?", top_k=3)

    assert result["confidence"] > 0
    assert "retrieval" in result["matched_terms"]
