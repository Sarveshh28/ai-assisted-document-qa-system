from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
INDEX_PATH = MODELS_DIR / "research_index.joblib"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
INDEX_METADATA_PATH = PROCESSED_DIR / "index_metadata.json"
EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.json"

DEFAULT_CHUNK_SIZE = 120
DEFAULT_CHUNK_OVERLAP = 30
SUPPORTED_DOCUMENT_EXTENSIONS = (".txt", ".md")
