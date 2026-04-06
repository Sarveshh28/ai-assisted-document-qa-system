from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import RAW_DIR, SAMPLE_DOCS_DIR
from src.pipeline.engine import ResearchAssistantEngine
from src.pipeline.indexer import build_index
from src.utils.io import read_json, slugify_filename, write_text_file


app = FastAPI(title="Research Paper Intelligence Assistant", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5)
    top_k: int = Field(default=5, ge=1, le=10)


class IngestDocument(BaseModel):
    title: str = Field(..., min_length=2)
    text: str = Field(..., min_length=20)


class IngestRequest(BaseModel):
    documents: list[IngestDocument] = Field(default_factory=list)
    source: str = Field(default="custom", pattern="^(custom|sample)$")
    chunk_size: int = Field(default=120, ge=40, le=300)
    overlap: int = Field(default=30, ge=0, le=100)


class EvaluateRequest(BaseModel):
    dataset_path: str = Field(default="data/sample_docs/eval_questions.json")
    top_k: int = Field(default=5, ge=1, le=10)


def _load_engine() -> ResearchAssistantEngine:
    try:
        return ResearchAssistantEngine()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "project": "Research Paper Intelligence Assistant"}


@app.get("/stats")
def stats() -> dict:
    return _load_engine().stats()


@app.get("/documents")
def documents() -> list[dict]:
    return _load_engine().list_documents()


@app.post("/query")
def query(request: QueryRequest) -> dict:
    return _load_engine().ask(request.question, top_k=request.top_k)


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    if request.source == "sample":
        return build_index(SAMPLE_DOCS_DIR, chunk_size=request.chunk_size, overlap=request.overlap)

    if not request.documents:
        raise HTTPException(status_code=400, detail="Provide at least one document for custom ingestion.")

    for document in request.documents:
        filename = slugify_filename(document.title)
        write_text_file(RAW_DIR / filename, document.text)

    return build_index(RAW_DIR, chunk_size=request.chunk_size, overlap=request.overlap)


@app.post("/evaluate")
def evaluate(request: EvaluateRequest) -> dict:
    engine = _load_engine()
    dataset_path = Path(request.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")
    dataset = read_json(dataset_path)
    return engine.evaluate(dataset, top_k=request.top_k, save_report=True)
