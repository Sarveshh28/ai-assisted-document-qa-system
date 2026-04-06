from __future__ import annotations

import joblib

from src.config import EVALUATION_REPORT_PATH, INDEX_PATH, VECTORIZER_PATH
from src.pipeline.evaluation import evaluate_dataset
from src.pipeline.preprocess import content_terms
from src.pipeline.qa import generate_grounded_answer
from src.pipeline.reranker import rerank_passages
from src.pipeline.retriever import HybridRetriever
from src.utils.io import write_json


class ResearchAssistantEngine:
    def __init__(self) -> None:
        if not INDEX_PATH.exists() or not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                "Index not found. Run `python scripts/ingest.py --input-dir data/sample_docs` first."
            )

        payload = joblib.load(INDEX_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        self.payload = payload
        self.metadata = payload.get("metadata", {})
        self.documents = payload.get("documents", [])
        self.chunks = payload.get("chunks", [])
        self.retriever = HybridRetriever(
            chunks=self.chunks,
            vectorizer=vectorizer,
            matrix=payload["matrix"],
        )

    def ask(self, question: str, top_k: int = 5) -> dict:
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        reranked = rerank_passages(question, retrieved)
        answer = generate_grounded_answer(
            question,
            reranked,
            max_sentences=min(4, top_k),
            max_passages=top_k,
        )
        query_terms = content_terms(question)
        matched_terms = sorted(
            {
                term
                for passage in reranked
                for term in passage.get("matched_terms", [])
                if term in query_terms
            }
        )
        return {
            "question": question,
            "answer": answer["answer"],
            "evidence": answer["evidence"],
            "confidence": answer["confidence"],
            "coverage": answer["coverage"],
            "matched_terms": matched_terms,
            "top_k_used": top_k,
            "retrieved_passages": reranked,
        }

    def stats(self) -> dict:
        doc_word_counts = [doc.get("word_count", 0) for doc in self.documents]
        return {
            "metadata": self.metadata,
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "average_document_words": round(sum(doc_word_counts) / max(len(doc_word_counts), 1), 2),
            "documents": self.documents,
        }

    def list_documents(self) -> list[dict]:
        return self.documents

    def evaluate(self, dataset: list[dict], top_k: int = 5, save_report: bool = True) -> dict:
        report = evaluate_dataset(self, dataset, top_k=top_k)
        if save_report:
            write_json(EVALUATION_REPORT_PATH, report)
        return report
