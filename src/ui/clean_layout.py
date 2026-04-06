import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import INDEX_PATH, RAW_DIR, SAMPLE_DOCS_DIR
from src.pipeline.engine import ResearchAssistantEngine
from src.pipeline.indexer import build_index
from src.utils.io import read_json, slugify_filename, write_text_file


def apply_theme() -> None:
    st.set_page_config(
        page_title="AI-Assisted Document Retrieval and Evidence-Based Question Answering System",
        page_icon="📄",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root {
            --bg: #0c1117;
            --panel: #121923;
            --panel-2: #17212d;
            --ink: #edf3fb;
            --muted: #96a6b8;
            --line: #263445;
            --navy: #17324d;
            --navy-2: #214766;
            --accent: #f3b45d;
            --accent-soft: #3c2a14;
            --success-bg: #173826;
            --success-text: #e8f7eb;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(243, 180, 93, 0.09), transparent 18%),
                radial-gradient(circle at top right, rgba(72, 132, 188, 0.10), transparent 20%),
                linear-gradient(180deg, #0c1117 0%, #0f1620 48%, #111a25 100%);
            color: var(--ink);
            font-family: "Trebuchet MS", "Gill Sans", "Segoe UI", sans-serif;
        }
        .main .block-container {
            max-width: 1240px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        [data-testid="stHeader"] {
            background: rgba(12, 17, 23, 0.88);
        }
        [data-testid="stSidebar"] {
            background: #111821;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * {
            color: var(--ink) !important;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: var(--ink);
        }
        .hero-shell {
            padding: 0.25rem 0 0.8rem 0;
            border-radius: 0;
            background: transparent;
            box-shadow: none;
            margin-bottom: 1rem;
        }
        .hero-shell h1 {
            color: var(--ink) !important;
            font-size: 2.35rem;
            line-height: 1.08;
            margin: 0 0 0.45rem 0;
            font-family: Georgia, "Times New Roman", serif;
        }
        .hero-shell p {
            color: var(--muted) !important;
            font-size: 1rem;
            margin: 0;
            max-width: 820px;
        }
        .panel {
            background: rgba(18, 25, 35, 0.96);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.28);
        }
        .metric-card {
            background: rgba(18, 25, 35, 0.98);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.95rem 1rem;
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.22);
        }
        .metric-label {
            font-size: 0.84rem;
            color: var(--muted) !important;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 2.15rem;
            line-height: 1;
            color: var(--ink) !important;
        }
        .subtle {
            color: var(--muted) !important;
        }
        .answer-box {
            background: linear-gradient(180deg, #131b25 0%, #162230 100%);
            border: 1px solid var(--line);
            border-left: 6px solid var(--accent);
            border-radius: 20px;
            padding: 1.15rem 1.2rem;
        }
        .answer-box p, .answer-box strong {
            color: var(--ink) !important;
        }
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .chip {
            display: inline-block;
            background: var(--accent-soft);
            color: #ffd89c !important;
            border-radius: 999px;
            padding: 0.3rem 0.65rem;
            font-size: 0.82rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
            margin-top: 2.35rem;
            margin-bottom: 2.25rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: #141d28;
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--ink) !important;
            font-weight: 600;
            padding: 0.62rem 1rem;
        }
        .stTabs [aria-selected="true"] {
            background: var(--accent) !important;
            color: #0f1720 !important;
            border-color: var(--accent) !important;
        }
        .stTabs [aria-selected="true"] * {
            color: #0f1720 !important;
            fill: #0f1720 !important;
        }
        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox [data-baseweb="select"] > div {
            background: var(--panel) !important;
            color: var(--ink) !important;
            border: 1px solid var(--line) !important;
            border-radius: 14px !important;
        }
        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: var(--muted) !important;
            opacity: 1 !important;
        }
        .stButton button,
        .stForm button {
            background: var(--accent) !important;
            color: #0f1720 !important;
            border: 1px solid var(--accent) !important;
            border-radius: 14px !important;
            font-weight: 600 !important;
            padding: 0.55rem 1rem !important;
            min-height: 3rem !important;
        }
        .stButton button:hover,
        .stForm button:hover {
            background: #ffc872 !important;
            color: #0f1720 !important;
        }
        .stButton button *,
        .stForm button *,
        [data-testid="stSidebar"] .stButton button,
        [data-testid="stSidebar"] .stButton button * {
            color: #0f1720 !important;
            fill: #0f1720 !important;
        }
        [data-testid="stSidebar"] .stSuccess,
        [data-testid="stSidebar"] .stSuccess * {
            color: var(--success-text) !important;
        }
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div {
            color: var(--ink) !important;
        }
        [data-testid="stSidebar"] .stButton button {
            background: var(--accent) !important;
            color: #0f1720 !important;
            border-color: var(--accent) !important;
        }
        [data-testid="stSidebar"] .stButton button p,
        [data-testid="stSidebar"] .stButton button span,
        [data-testid="stSidebar"] .stButton button div {
            color: #0f1720 !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: var(--ink) !important;
        }
        [data-testid="stSidebar"] .stAlert,
        [data-testid="stSidebar"] .stSuccess {
            background: var(--success-bg) !important;
            color: var(--success-text) !important;
        }
        [data-testid="stSidebar"] .stSuccess p,
        [data-testid="stSidebar"] .stSuccess span,
        [data-testid="stSidebar"] .stSuccess div {
            color: var(--success-text) !important;
        }
        .stSlider [data-baseweb="slider"] > div > div,
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
        }
        .stSlider [data-baseweb="slider"] span,
        .stSlider [data-baseweb="slider"] div {
            color: var(--ink) !important;
        }
        .stSlider [data-baseweb="slider"] {
            padding-top: 0.35rem;
        }
        .stDataFrame, [data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid var(--line);
        }
        [data-testid="stSidebarCollapseButton"] button,
        button[kind="headerNoPadding"],
        [data-testid="collapsedControl"] {
            color: var(--accent) !important;
        }
        [data-testid="stSidebarCollapseButton"] button *,
        button[kind="headerNoPadding"] *,
        [data-testid="collapsedControl"] * {
            color: var(--accent) !important;
            fill: var(--accent) !important;
        }
        [data-testid="stMarkdownContainer"] p,
        .stCaption,
        .stRadio label,
        .stSlider label,
        .stTextInput label,
        .stTextArea label,
        .stFileUploader label {
            color: var(--ink) !important;
        }
        table, th, td {
            color: var(--ink) !important;
        }
        @media (max-width: 900px) {
            .hero-shell h1 {
                font-size: 1.9rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def current_index_signature() -> float:
    return INDEX_PATH.stat().st_mtime if INDEX_PATH.exists() else 0.0


@st.cache_resource(show_spinner=False)
def load_engine(signature: float) -> ResearchAssistantEngine:
    return ResearchAssistantEngine()


def clear_engine_cache() -> None:
    load_engine.clear()


def try_get_engine() -> ResearchAssistantEngine | None:
    signature = current_index_signature()
    if not signature:
        return None
    try:
        return load_engine(signature)
    except FileNotFoundError:
        return None


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_passage_table(passages: list[dict]) -> None:
    if not passages:
        st.info("No passages retrieved yet.")
        return
    frame = pd.DataFrame(
        [
            {
                "Title": passage["title"],
                "Score": round(passage["rerank_score"], 3),
                "Semantic": round(passage["semantic_score"], 3),
                "BM25": round(passage["bm25_score"], 3),
                "Matched Terms": ", ".join(passage.get("matched_terms", [])),
                "Preview": passage["text"][:220] + ("..." if len(passage["text"]) > 220 else ""),
            }
            for passage in passages
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)


def persist_uploaded_documents(files: list, title: str, text: str) -> int:
    saved = 0
    for file in files:
        suffix = Path(file.name).suffix or ".txt"
        filename = slugify_filename(Path(file.name).stem, suffix=suffix)
        write_text_file(RAW_DIR / filename, file.getvalue().decode("utf-8"))
        saved += 1
    if title.strip() and text.strip():
        filename = slugify_filename(title, suffix=".txt")
        write_text_file(RAW_DIR / filename, text.strip())
        saved += 1
    return saved


def sample_question_list() -> list[str]:
    dataset = read_json(SAMPLE_DOCS_DIR / "eval_questions.json")
    return [item["question"] for item in dataset]


def render_chip_list(items: list[str]) -> None:
    if not items:
        st.markdown('<span class="subtle">No matched terms</span>', unsafe_allow_html=True)
        return
    chips = "".join(f'<span class="chip">{item}</span>' for item in items)
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <h1>AI-Assisted Document Retrieval and Evidence-Based Question Answering System</h1>
            <p>Upload notes or use the sample dataset, ask grounded questions, and review the exact evidence used by the system.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(stats: dict | None) -> None:
    with st.sidebar:
        st.markdown("## Project Status")
        st.caption("This panel shows status and quick actions.")
        if stats:
            metadata = stats["metadata"]
            st.success("Knowledge base is ready")
            st.write(f"Documents: **{stats['document_count']}**")
            st.write(f"Chunks: **{stats['chunk_count']}**")
            st.write(f"Chunk size: **{metadata.get('chunk_size', '-')}**")
            st.write(f"Overlap: **{metadata.get('overlap', '-')}**")
        else:
            st.info("No index loaded yet. Build the sample corpus first.")

        if st.button("Build Sample Corpus", use_container_width=True):
            with st.spinner("Building sample corpus..."):
                summary = build_index(SAMPLE_DOCS_DIR)
            clear_engine_cache()
            st.session_state["last_ingest_summary"] = summary
            st.rerun()


def render_stats(stats: dict | None) -> None:
    if not stats:
        st.info("Start with `Use Sample Data` or `Upload Documents`, then come back to `Ask Questions`.")
        return

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Documents", str(stats["document_count"]))
    with metric_cols[1]:
        render_metric_card("Chunks", str(stats["chunk_count"]))
    with metric_cols[2]:
        render_metric_card("Avg Doc Words", str(stats["average_document_words"]))
    with metric_cols[3]:
        render_metric_card("Avg Chunk Tokens", str(stats["metadata"].get("avg_chunk_tokens", 0)))
    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)


def render_overview(questions: list[str]) -> None:
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("### What this project does")
        st.markdown(
            """
            <div class="panel">
                <p>This app is a document question-answering system. It answers from the documents you load, not from the full internet.</p>
                <p><strong>Input:</strong> `.txt` or `.md` technical documents.</p>
                <p><strong>Output:</strong> a grounded answer, matched terms, evidence sentences, and retrieved passages.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("### Quick flow")
        st.markdown(
            """
            <div class="panel">
                <p><strong>Step 1:</strong> Open <em>Use Sample Data</em> and build the bundled corpus.</p>
                <p><strong>Step 2:</strong> Open <em>Ask Questions</em> and ask something like "What is retrieval augmented generation?"</p>
                <p><strong>Step 3:</strong> Review the answer and evidence.</p>
                <p><strong>Step 4:</strong> Use <em>Evaluate</em> to run benchmark questions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Example questions")
    question_lines = "".join(f"<p>{idx}. {question}</p>" for idx, question in enumerate(questions, start=1))
    st.markdown(f'<div class="panel">{question_lines}</div>', unsafe_allow_html=True)


def render_ask_tab(engine: ResearchAssistantEngine | None) -> None:
    st.markdown("### Ask Questions")
    with st.form("qa_form", clear_on_submit=False):
        default_question = st.session_state.get(
            "selected_question",
            "What are the major benefits of retrieval augmented generation?",
        )
        question = st.text_input(
            "Question",
            value=default_question,
            placeholder="Ask a question about the indexed documents",
        )
        control_left, control_right = st.columns([1.25, 0.55], gap="large")
        with control_left:
            top_k = st.slider("Search Depth (Top K)", min_value=1, max_value=10, value=5)
            st.caption("Top K is the number of passages searched before the answer is written.")
        with control_right:
            st.markdown("<div style='height: 1.85rem;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Run Research Assistant", use_container_width=True)

    if submitted:
        if not engine:
            st.error("No index available yet. Build or ingest documents first.")
        else:
            st.session_state["qa_result"] = engine.ask(question, top_k=top_k)

    result = st.session_state.get("qa_result")
    if result:
        answer_col, signal_col = st.columns([1.6, 0.9], gap="large")
        with answer_col:
            st.markdown(
                f"""
                <div class="answer-box">
                    <strong>Grounded answer</strong>
                    <p>{result["answer"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with signal_col:
            st.markdown(
                f"""
                <div class="panel">
                    <p><strong>Top K used:</strong> {result["top_k_used"]}</p>
                    <p><strong>Confidence:</strong> {result["confidence"]}</p>
                    <p><strong>Coverage:</strong> {result["coverage"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("Matched terms")
            render_chip_list(result["matched_terms"])

        st.markdown("### Evidence")
        for evidence in result["evidence"]:
            st.markdown(
                f"""
                <div class="panel">
                    <p><strong>{evidence["title"]}</strong></p>
                    <p>{evidence["sentence"]}</p>
                    <p class="subtle">Score: {evidence["score"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Retrieved passages")
        render_passage_table(result["retrieved_passages"])


def render_sample_tab() -> None:
    st.markdown("### Use Sample Data")
    st.caption("Best for demo and testing. These documents are already included in the project.")
    sample_docs = [
        path.name
        for path in sorted(SAMPLE_DOCS_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in {".txt", ".md"} and "question" not in path.stem.lower()
    ]
    sample_left, sample_right = st.columns([1.1, 1], gap="large")
    with sample_left:
        st.markdown(
            """
            <div class="panel">
                <p><strong>Included topics:</strong> RAG, Transformers, Computer Vision, AI Agents, LLM Safety, and Multimodal AI.</p>
                <p>Press the button below to rebuild the sample corpus and refresh the retrieval index.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Build Sample Dataset Now"):
            with st.spinner("Building sample dataset..."):
                summary = build_index(SAMPLE_DOCS_DIR)
            clear_engine_cache()
            st.session_state["last_ingest_summary"] = summary
            st.rerun()
    with sample_right:
        st.markdown("#### Included files")
        st.dataframe(pd.DataFrame({"Sample files": sample_docs}), use_container_width=True, hide_index=True)


def render_upload_tab() -> None:
    st.markdown("### Upload Documents")
    st.caption("Use this when you want the assistant to answer from your own notes or project files.")
    last_summary = st.session_state.get("last_ingest_summary")
    if last_summary:
        st.success(
            f"Last build: {last_summary['document_count']} documents, {last_summary['chunk_count']} chunks."
        )

    with st.form("upload_form"):
        upload_col, paste_col = st.columns(2, gap="large")
        with upload_col:
            uploads = st.file_uploader(
                "Upload `.txt` or `.md` files",
                type=["txt", "md"],
                accept_multiple_files=True,
            )
        with paste_col:
            doc_title = st.text_input("Optional pasted document title")
            doc_text = st.text_area("Optional pasted document text", height=220)

        settings_col, overlap_col = st.columns(2)
        with settings_col:
            chunk_size = st.slider("Chunk size", min_value=40, max_value=300, value=120, step=10)
        with overlap_col:
            overlap = st.slider("Overlap", min_value=0, max_value=100, value=30, step=5)

        build_custom = st.form_submit_button("Build Index From My Documents")

    if build_custom:
        with st.spinner("Indexing your documents..."):
            saved = persist_uploaded_documents(uploads or [], doc_title, doc_text)
            if saved == 0:
                st.error("Upload at least one file or paste one document before building the index.")
                st.stop()
            summary = build_index(RAW_DIR, chunk_size=chunk_size, overlap=overlap)
        clear_engine_cache()
        st.session_state["last_ingest_summary"] = summary
        st.rerun()


def render_evaluate_tab(engine: ResearchAssistantEngine | None) -> None:
    st.markdown("### Evaluate")
    st.caption("Runs the benchmark question set against the current sample corpus.")
    eval_top_k = st.slider("Evaluation top K", min_value=1, max_value=10, value=5)
    if st.button("Run Benchmark Evaluation"):
        if not engine:
            st.error("No index available yet. Build the sample corpus first.")
        else:
            report = engine.evaluate(read_json(SAMPLE_DOCS_DIR / "eval_questions.json"), top_k=eval_top_k, save_report=True)
            metric_cols = st.columns(4)
            with metric_cols[0]:
                render_metric_card("Recall@1", str(report["recall@1"]))
            with metric_cols[1]:
                render_metric_card("Recall@3", str(report["recall@3"]))
            with metric_cols[2]:
                render_metric_card("Recall@5", str(report["recall@5"]))
            with metric_cols[3]:
                render_metric_card("MRR", str(report["mrr"]))
            chart_frame = pd.DataFrame(
                {
                    "metric": ["recall@1", "recall@3", "recall@5", "mrr"],
                    "score": [report["recall@1"], report["recall@3"], report["recall@5"], report["mrr"]],
                }
            ).set_index("metric")
            st.bar_chart(chart_frame)
            st.dataframe(pd.DataFrame(report["details"]), use_container_width=True, hide_index=True)


def render_library_tab(engine: ResearchAssistantEngine | None) -> None:
    st.markdown("### Document Library")
    if not engine:
        st.info("No indexed documents yet.")
        return

    documents = engine.list_documents()
    st.dataframe(
        pd.DataFrame(documents)[["title", "word_count", "sentence_count", "keywords"]],
        use_container_width=True,
        hide_index=True,
    )
    selected_title = st.selectbox("Preview a document", options=[doc["title"] for doc in documents])
    selected_doc = next(doc for doc in documents if doc["title"] == selected_title)
    preview_chunk = next(
        (chunk for chunk in engine.chunks if chunk["doc_id"] == selected_doc["doc_id"]),
        None,
    )
    st.markdown(
        f"""
        <div class="panel">
            <p><strong>Title:</strong> {selected_doc["title"]}</p>
            <p><strong>Keywords:</strong> {", ".join(selected_doc.get("keywords", []))}</p>
            <p><strong>Source:</strong> {selected_doc.get("source_path", "-")}</p>
            <p><strong>Preview:</strong> {(preview_chunk["text"][:650] + "...") if preview_chunk else "No preview available."}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_app() -> None:
    apply_theme()
    render_header()
    engine = try_get_engine()
    stats = engine.stats() if engine else None
    questions = sample_question_list()
    render_sidebar(stats)
    render_stats(stats)

    tabs = st.tabs(
        [
            "Overview",
            "Ask Questions",
            "Use Sample Data",
            "Upload Documents",
            "Evaluate",
            "Document Library",
        ]
    )

    with tabs[0]:
        render_overview(questions)
    with tabs[1]:
        render_ask_tab(engine)
    with tabs[2]:
        render_sample_tab()
    with tabs[3]:
        render_upload_tab()
    with tabs[4]:
        render_evaluate_tab(engine)
    with tabs[5]:
        render_library_tab(engine)
