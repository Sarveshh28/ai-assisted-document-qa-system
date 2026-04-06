from collections import Counter

from src.pipeline.preprocess import content_terms


def _overlap_score(question: str, text: str) -> float:
    q_terms = Counter(content_terms(question))
    t_terms = Counter(content_terms(text))
    overlap = sum(min(q_terms[token], t_terms[token]) for token in q_terms)
    return overlap / max(len(q_terms), 1)


def rerank_passages(question: str, passages: list[dict]) -> list[dict]:
    rescored: list[dict] = []
    for passage in passages:
        lexical_overlap = _overlap_score(question, passage["text"])
        title_overlap = _overlap_score(question, passage["title"])
        rerank_score = (
            0.6 * passage["hybrid_score"]
            + 0.25 * lexical_overlap
            + 0.15 * title_overlap
        )
        item = dict(passage)
        item["rerank_score"] = float(rerank_score)
        item["lexical_overlap"] = float(lexical_overlap)
        item["title_overlap"] = float(title_overlap)
        rescored.append(item)
    return sorted(rescored, key=lambda item: item["rerank_score"], reverse=True)
