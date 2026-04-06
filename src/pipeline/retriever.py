from __future__ import annotations

import math
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.pipeline.preprocess import content_terms, tokenize


QUERY_EXPANSIONS = {
    "rag": ["retrieval", "augmented", "generation"],
    "llm": ["large", "language", "model"],
    "nlp": ["natural", "language", "processing"],
    "cv": ["computer", "vision"],
    "qa": ["question", "answering"],
}


class HybridRetriever:
    def __init__(self, chunks: list[dict], vectorizer, matrix) -> None:
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.doc_term_freqs = [Counter(content_terms(chunk["text"])) for chunk in chunks]
        self.doc_lengths = np.array([max(len(content_terms(chunk["text"])), 1) for chunk in chunks], dtype=float)
        self.avg_doc_length = float(self.doc_lengths.mean()) if len(self.doc_lengths) else 0.0
        self.doc_freqs = self._compute_document_frequencies()
        self.total_docs = len(chunks)

    def _compute_document_frequencies(self) -> dict[str, int]:
        doc_freqs: dict[str, int] = {}
        for term_freqs in self.doc_term_freqs:
            for term in term_freqs:
                doc_freqs[term] = doc_freqs.get(term, 0) + 1
        return doc_freqs

    def _expand_query_terms(self, question: str) -> list[str]:
        expanded: list[str] = []
        for token in content_terms(question):
            expanded.append(token)
            expanded.extend(QUERY_EXPANSIONS.get(token, []))
        return expanded or tokenize(question)

    def _bm25_score(
        self,
        query_tokens: list[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> np.ndarray:
        if not self.chunks:
            return np.array([])

        scores = np.zeros(len(self.chunks), dtype=float)
        for term in query_tokens:
            doc_freq = self.doc_freqs.get(term, 0)
            if doc_freq == 0:
                continue
            idf = math.log(1 + (self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            for idx, term_freqs in enumerate(self.doc_term_freqs):
                tf = term_freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + k1 * (1 - b + b * (self.doc_lengths[idx] / (self.avg_doc_length + 1e-9)))
                scores[idx] += idf * (tf * (k1 + 1)) / (denom + 1e-9)
        return scores

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        if not self.chunks:
            return []

        query_vec = self.vectorizer.transform([question])
        semantic = cosine_similarity(query_vec, self.matrix).flatten()
        expanded_terms = self._expand_query_terms(question)
        bm25 = self._bm25_score(expanded_terms)

        sem_norm = semantic / (semantic.max() + 1e-9) if semantic.max() > 0 else semantic
        bm25_norm = bm25 / (bm25.max() + 1e-9) if bm25.max() > 0 else bm25

        combined = 0.65 * sem_norm + 0.35 * bm25_norm
        ranked_idx = combined.argsort()[::-1][:top_k]

        results: list[dict] = []
        for idx in ranked_idx:
            item = dict(self.chunks[idx])
            matched_terms = sorted(set(expanded_terms).intersection(set(content_terms(item["text"]))))
            item["semantic_score"] = float(semantic[idx])
            item["bm25_score"] = float(bm25[idx])
            item["hybrid_score"] = float(combined[idx])
            item["matched_terms"] = matched_terms
            results.append(item)
        return results
