from dataclasses import dataclass

from src.pipeline.preprocess import split_sentences


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    token_count: int


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_document(
    *,
    doc_id: str,
    title: str,
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_words = _word_count(sentence)
        if current and current_words + sentence_words > chunk_size:
            chunk_text = " ".join(current).strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_index}",
                    doc_id=doc_id,
                    title=title,
                    text=chunk_text,
                    token_count=_word_count(chunk_text),
                )
            )
            chunk_index += 1

            overlap_words: list[str] = []
            if overlap > 0:
                tail_words = chunk_text.split()[-overlap:]
                overlap_words = tail_words
            current = [" ".join(overlap_words)] if overlap_words else []
            current_words = len(overlap_words)

        current.append(sentence)
        current_words += sentence_words

    if current:
        chunk_text = " ".join(part for part in current if part).strip()
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                doc_id=doc_id,
                title=title,
                text=chunk_text,
                token_count=_word_count(chunk_text),
            )
        )

    return chunks
