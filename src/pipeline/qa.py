from src.pipeline.preprocess import content_terms, split_sentences


def generate_grounded_answer(
    question: str,
    passages: list[dict],
    max_sentences: int = 3,
    max_passages: int = 3,
) -> dict:
    if not passages:
        return {
            "answer": "I could not find enough supporting evidence in the indexed documents.",
            "evidence": [],
            "confidence": 0.0,
            "coverage": 0.0,
        }

    query_terms = set(content_terms(question))
    scored_sentences: list[tuple[float, str, str, list[str]]] = []

    for passage in passages[:max_passages]:
        for sentence in split_sentences(passage["text"]):
            sentence_terms = set(content_terms(sentence))
            overlap_terms = sorted(query_terms.intersection(sentence_terms))
            score = len(overlap_terms) + passage.get("rerank_score", 0.0)
            if score > 0:
                scored_sentences.append((score, sentence, passage["title"], overlap_terms))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    selected = scored_sentences[:max_sentences]

    if not selected:
        selected = [(1.0, passages[0]["text"], passages[0]["title"], passages[0].get("matched_terms", []))]

    answer = " ".join(sentence for _, sentence, _, _ in selected)
    evidence = []
    matched_terms: set[str] = set()
    for score, sentence, title, overlap_terms in selected:
        matched_terms.update(overlap_terms)
        evidence.append(
            {
                "title": title,
                "sentence": sentence,
                "score": round(float(score), 3),
                "matched_terms": overlap_terms,
            }
        )

    coverage = len(matched_terms) / max(len(query_terms), 1)
    confidence = min(0.99, 0.45 + 0.25 * coverage + 0.15 * len(evidence))

    return {
        "answer": answer,
        "evidence": evidence,
        "confidence": round(confidence, 3),
        "coverage": round(coverage, 3),
    }
