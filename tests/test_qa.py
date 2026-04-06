from src.pipeline.qa import generate_grounded_answer


def test_generate_grounded_answer_uses_passages():
    passages = [
        {
            "title": "RAG",
            "text": "Retrieval augmented generation improves factual grounding and reduces hallucinations.",
            "hybrid_score": 1.0,
        }
    ]
    result = generate_grounded_answer("How does RAG help?", passages)
    assert "grounding" in result["answer"].lower()
    assert result["evidence"]
