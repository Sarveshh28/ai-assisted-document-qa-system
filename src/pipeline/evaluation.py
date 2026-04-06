from __future__ import annotations


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_id: str, k: int) -> float:
    return 1.0 if relevant_doc_id in retrieved_doc_ids[:k] else 0.0


def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_id: str) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == relevant_doc_id:
            return 1.0 / rank
    return 0.0


def evaluate_dataset(engine, examples: list[dict], top_k: int = 5) -> dict:
    recall_1 = 0.0
    recall_3 = 0.0
    recall_5 = 0.0
    mrr = 0.0
    details: list[dict] = []

    for item in examples:
        result = engine.ask(item["question"], top_k=top_k)
        ranked_doc_ids = [passage["doc_id"] for passage in result["retrieved_passages"]]
        target = item["relevant_doc_id"]
        recall_1 += recall_at_k(ranked_doc_ids, target, 1)
        recall_3 += recall_at_k(ranked_doc_ids, target, 3)
        recall_5 += recall_at_k(ranked_doc_ids, target, 5)
        mrr += reciprocal_rank(ranked_doc_ids, target)
        details.append(
            {
                "question": item["question"],
                "target_doc_id": target,
                "top_prediction": ranked_doc_ids[0] if ranked_doc_ids else None,
                "hit@1": bool(recall_at_k(ranked_doc_ids, target, 1)),
                "hit@3": bool(recall_at_k(ranked_doc_ids, target, 3)),
                "reciprocal_rank": round(reciprocal_rank(ranked_doc_ids, target), 4),
            }
        )

    total = max(len(examples), 1)
    return {
        "examples": len(examples),
        "recall@1": round(recall_1 / total, 4),
        "recall@3": round(recall_3 / total, 4),
        "recall@5": round(recall_5 / total, 4),
        "mrr": round(mrr / total, 4),
        "details": details,
    }
