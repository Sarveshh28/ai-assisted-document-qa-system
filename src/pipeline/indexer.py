from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import joblib

from src.config import (
    CHUNKS_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    INDEX_METADATA_PATH,
    INDEX_PATH,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    VECTORIZER_PATH,
)
from src.pipeline.chunking import chunk_document
from src.pipeline.embedder import TfidfEmbedder
from src.pipeline.preprocess import normalize_text, split_sentences, top_keywords
from src.utils.io import ensure_dir, read_text_file, write_json, write_jsonl


def load_documents(input_dir: Path) -> list[dict]:
    documents: list[dict] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            continue
        if "question" in path.stem.lower():
            continue
        text = normalize_text(read_text_file(path))
        documents.append(
            {
                "doc_id": path.stem.lower().replace(" ", "_"),
                "title": path.stem.replace("_", " ").title(),
                "text": text,
                "source_path": str(path),
                "word_count": len(text.split()),
                "sentence_count": len(split_sentences(text)),
                "keywords": top_keywords(text),
            }
        )
    return documents


def build_index(
    input_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    ensure_dir(PROCESSED_DIR)
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORTS_DIR)

    documents = load_documents(input_dir)
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(
            doc_id=doc["doc_id"],
            title=doc["title"],
            text=doc["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(asdict(chunk) for chunk in chunks)

    texts = [chunk["text"] for chunk in all_chunks]
    embedder = TfidfEmbedder()
    matrix = embedder.fit_transform(texts) if texts else None
    avg_chunk_tokens = round(
        sum(chunk["token_count"] for chunk in all_chunks) / max(len(all_chunks), 1),
        2,
    )
    metadata = {
        "project": "Research Paper Intelligence Assistant",
        "source_dir": str(input_dir),
        "document_count": len(documents),
        "chunk_count": len(all_chunks),
        "avg_chunk_tokens": avg_chunk_tokens,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "documents": [
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "word_count": doc["word_count"],
                "sentence_count": doc["sentence_count"],
                "keywords": doc["keywords"],
                "source_path": doc["source_path"],
            }
            for doc in documents
        ],
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

    write_jsonl(CHUNKS_PATH, all_chunks)
    write_json(INDEX_METADATA_PATH, metadata)
    joblib.dump(embedder.vectorizer, VECTORIZER_PATH)
    joblib.dump(
        {
            "documents": documents,
            "chunks": all_chunks,
            "matrix": matrix,
            "metadata": metadata,
        },
        INDEX_PATH,
    )

    return metadata | {"index_path": str(INDEX_PATH)}
