# Research Paper Intelligence Assistant

An advanced artificial intelligence project that builds a complete research-document question answering system with hybrid retrieval, reranking, grounded answer synthesis, evaluation, API endpoints, and a direct Streamlit UI.

## Why This Fits AAI

This project combines several strong AAI concepts in one end-to-end system:

- Text preprocessing and sentence-aware chunking
- Vector search with TF-IDF semantic retrieval
- BM25-style lexical retrieval
- Hybrid score fusion
- Lightweight reranking
- Grounded answer generation from evidence
- Retrieval evaluation with Recall@K and MRR
- FastAPI backend and Streamlit frontend

## Project Idea

The assistant ingests technical notes, lecture material, or research documents in `.txt` and `.md` format. It builds an index over those documents, retrieves the most relevant passages for a question, reranks them, and produces an answer supported by cited evidence.

## Main Features

- Build an index from bundled sample papers or your own uploaded documents
- Ask natural language questions through the UI, CLI, or API
- View confidence, coverage, evidence sentences, and retrieved passages
- Inspect indexed document metadata and keywords
- Run a benchmark dataset and generate retrieval metrics
- Save processed outputs such as chunk files, index metadata, and evaluation reports

## Folder Structure

```text
.
|-- data/
|   |-- processed/
|   |-- raw/
|   |-- reports/
|   `-- sample_docs/
|-- models/
|-- scripts/
|-- src/
|   |-- api/
|   |-- pipeline/
|   |-- ui/
|   `-- utils/
`-- tests/
```

## Quick Start

### 1. Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Build the sample corpus

```powershell
python scripts/ingest.py --input-dir data/sample_docs
```

### 3. Ask a question from the CLI

```powershell
python scripts/query.py --question "What are the major benefits of retrieval augmented generation?"
```

### 4. Run the evaluation pipeline

```powershell
python scripts/evaluate.py --dataset data/sample_docs/eval_questions.json
```

### 5. Launch the API

```powershell
uvicorn src.api.main:app --reload
```

API endpoints:

- `GET /health`
- `GET /stats`
- `GET /documents`
- `POST /query`
- `POST /ingest`
- `POST /evaluate`

### 6. Launch the direct UI

```powershell
streamlit run src/ui/app.py
```

The UI includes:

- `Ask AI`: grounded question answering
- `Ingest Documents`: build the corpus from sample or custom files
- `Evaluate`: run benchmark metrics and inspect results
- `Document Library`: explore indexed document stats and previews

## Pipeline

1. Load raw `.txt` or `.md` documents.
2. Normalize and clean text.
3. Split documents into overlapping chunks.
4. Fit a TF-IDF vectorizer across chunks.
5. Compute lexical statistics for BM25-style retrieval.
6. Fuse semantic and lexical scores into a hybrid rank.
7. Rerank passages using overlap and title relevance.
8. Generate a grounded answer from top evidence sentences.
9. Evaluate retrieval quality with Recall@K and MRR.

## Example Outputs

After ingestion, the system writes:

- `data/processed/chunks.jsonl`
- `data/processed/index_metadata.json`
- `models/research_index.joblib`
- `models/vectorizer.joblib`
- `data/reports/evaluation_report.json`

## Suggested Submission Title

**Research Paper Intelligence Assistant using Hybrid Retrieval and Grounded Question Answering**
