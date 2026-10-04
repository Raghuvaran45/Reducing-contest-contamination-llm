import time
import pandas as pd
import streamlit as st

from retrieval import retrieval_pipeline as rag


st.set_page_config(
    page_title="PDF RAG Intelligence System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


CASE_STUDIES = [
    "General PDF QA",
    "Medical Report Analysis",
    "Legal Court Case Analysis",
]


def init_state():
    defaults = {
        "case_study": "General PDF QA",
        "uploaded_signature": None,
        "chat_history": [],
        "last_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def pct(value):
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def signed_pct(value):
    if value is None:
        return "N/A"
    return f"{float(value) * 100:+.2f}%"


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def load_pdf_if_needed(uploaded_file, case_study):
    if uploaded_file is None:
        return

    signature = (
        uploaded_file.name,
        uploaded_file.size,
        case_study,
    )

    if st.session_state.uploaded_signature == signature:
        return

    with st.spinner("Processing PDF: extracting text, chunking, embedding and building FAISS..."):
        info = rag.load_uploaded_pdf_bytes(
            uploaded_file.getvalue(),
            uploaded_file.name,
            case_study=case_study,
        )

    st.session_state.uploaded_signature = signature
    st.session_state.chat_history = []
    st.session_state.last_result = None
    st.success(
        f"Loaded **{info['pdf_name']}** — "
        f"{info['chunks']} chunks / {info['vectors']} FAISS vectors."
    )


def metric_table(result):
    evaluation = result.get("evaluation", {})
    current = evaluation.get("current_question_metrics", {})

    retrieval = evaluation.get("retrieval", {})
    rb = retrieval.get("baseline", {})
    ra = retrieval.get("proposed", {})

    gb = current.get("text_generation", {}).get("baseline", {})
    ga = current.get("text_generation", {}).get("proposed", {})

    b = current.get("baseline", {})
    a = current.get("proposed", {})

    def diff(before, after, lower=False):
        if before is None or after is None:
            return None
        return (float(before) - float(after)) if lower else (float(after) - float(before))

    rows = []

    # Retrieval metrics.
    for key, label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-score"),
    ]:
        before = rb.get(key)
        after = ra.get(key)
        rows.append({
            "Metric": label,
            "Retrieval — Before Filtering": pct(before),
            "Retrieval — After Filtering": pct(after),
            "Retrieval Improvement": signed_pct(diff(before, after)),
            "Generation — Before Filtering": pct(b.get(key)),
            "Generation — After Filtering": pct(a.get(key)),
            "Generation Improvement": signed_pct(diff(b.get(key), a.get(key))),
        })

    generation_rows = [
        ("bleu", "BLEU-4"),
        ("rouge1", "ROUGE-1 F1"),
        ("rouge2", "ROUGE-2 F1"),
        ("rougeL", "ROUGE-L F1"),
    ]

    for key, label in generation_rows:
        before = gb.get(key)
        after = ga.get(key)
        rows.append({
            "Metric": label,
            "Retrieval — Before Filtering": "—",
            "Retrieval — After Filtering": "—",
            "Retrieval Improvement": "—",
            "Generation — Before Filtering": pct(before),
            "Generation — After Filtering": pct(after),
            "Generation Improvement": signed_pct(diff(before, after)),
        })

    bp = gb.get("perplexity")
    ap = ga.get("perplexity")

    rows.append({
        "Metric": "Perplexity",
        "Retrieval — Before Filtering": "—",
        "Retrieval — After Filtering": "—",
        "Retrieval Improvement": "—",
        "Generation — Before Filtering": f"{bp:.2f}" if bp is not None else "N/A",
        "Generation — After Filtering": f"{ap:.2f}" if ap is not None else "N/A",
        "Generation Improvement": (
            f"{((bp - ap) / bp) * 100:+.2f}%"
            if bp is not None and ap is not None and bp != 0
            else "N/A"
        ),
    })

    return pd.DataFrame(rows)


def render_sidebar():
    st.sidebar.title("📚 PDF RAG Intelligence")
    st.sidebar.caption("One shared RAG pipeline for General, Medical and Legal analysis.")

    case_study = st.sidebar.selectbox(
        "Case Study",
        CASE_STUDIES,
        index=CASE_STUDIES.index(st.session_state.case_study),
    )

    if case_study != st.session_state.case_study:
        st.session_state.case_study = case_study
        st.session_state.uploaded_signature = None
        st.session_state.chat_history = []
        st.session_state.last_result = None

    st.sidebar.divider()

    uploaded_file = st.sidebar.file_uploader(
        "Import PDF",
        type=["pdf"],
        help="Upload the PDF you want this case study to analyze.",
    )

    if uploaded_file:
        try:
            load_pdf_if_needed(
                uploaded_file,
                case_study,
            )
        except Exception as exc:
            st.sidebar.error(f"PDF processing failed: {exc}")

    st.sidebar.divider()

    try:
        info = rag.get_current_pdf_info()
        st.sidebar.metric("FAISS vectors", info["vectors"])
        st.sidebar.metric("PDF chunks", info["chunks"])
        st.sidebar.caption(f"Active PDF: {info['pdf_name']}")
        st.sidebar.caption(f"Case study: {info.get('case_study', case_study)}")
    except Exception:
        st.sidebar.info("Upload a PDF to start.")

    return case_study


def render_chat(case_study):
    st.title("💬 PDF Question Answering")

    try:
        info = rag.get_current_pdf_info()
    except Exception:
        info = {"pdf_name": "No PDF loaded", "chunks": 0, "vectors": 0}

    if info["pdf_name"] == "No document selected":
        st.info("Select a case study and upload a PDF from the sidebar.")
        return

    st.caption(
        f"Case study: **{case_study}**  ·  "
        f"Document: **{info['pdf_name']}**"
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the imported PDF...")

    if not question:
        return

    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence, filtering context and generating answer..."):
            start = time.time()
            result = rag.answer_question(
                question,
                return_metadata=True,
            )
            elapsed = time.time() - start

        answer = result.get("proposed_answer", result.get("answer", ""))

        if case_study == "Medical Report Analysis":
            # The central RAG answer remains the source. The optional
            # medical layer adds domain formatting without creating a second RAG.
            from medical.medical_analysis import generate_medical_analysis
            answer = generate_medical_analysis(question, result)

        elif case_study == "Legal Court Case Analysis":
            from legal.legal_analysis import generate_legal_analysis
            answer = generate_legal_analysis(question, result)

        st.markdown(answer)

        cols = st.columns(4)
        cols[0].metric("Retrieved", result.get("retrieved_count", 0))
        cols[1].metric("Kept", result.get("clean_count", 0))
        cols[2].metric("Removed", result.get("removed_count", 0))
        cols[3].metric("Time", f"{elapsed:.2f}s")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
        })
        st.session_state.last_result = result


def render_overview(case_study):
    st.title("📄 Document Overview")

    try:
        info = rag.get_document_info()
    except Exception:
        st.info("Upload a PDF first.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Chunks", info.get("chunks", 0))
    c2.metric("FAISS vectors", info.get("vectors", 0))
    c3.metric("References", info.get("references") or "N/A")

    st.subheader("Document")
    st.write(f"**PDF:** {info.get('pdf_name', '')}")
    st.write(f"**Case study:** {case_study}")

    if info.get("title"):
        st.subheader("Title")
        st.write(info["title"])

    if info.get("authors"):
        st.subheader("Authors")
        st.write(info["authors"])


def render_retrieval_analysis():
    st.title("🔎 Retrieval Analysis")

    if not st.session_state.last_result:
        st.info("Ask a question first.")
        return

    result = st.session_state.last_result

    st.subheader("Current Question")
    st.write(result.get("question", ""))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retrieved", result.get("retrieved_count", 0))
    c2.metric("After filtering", result.get("clean_count", 0))
    c3.metric("Removed", result.get("removed_count", 0))
    c4.metric(
        "Contamination reduction",
        f"{result.get('contamination_rate', 0):.2f}%"
    )

    st.subheader("Combined Metrics")
    st.dataframe(
        metric_table(result),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Before Filtering Answer")
    st.write(result.get("baseline_answer", ""))

    st.subheader("After Filtering Answer")
    st.write(result.get("proposed_answer", ""))

    st.subheader("Reference Answer")
    st.write(result.get("reference_answer", ""))

    retrieval = result.get("retrieval_metrics", {})
    if retrieval.get("available"):
        st.subheader("Retrieval Evidence Metrics")
        st.json(retrieval)


def render_contamination():
    st.title("🧹 Contamination Analysis")

    result = st.session_state.last_result

    if not result:
        st.info("Ask a question first.")
        return

    evidence = result.get("evidence", [])
    clean_evidence = result.get("clean_evidence", [])

    c1, c2, c3 = st.columns(3)
    c1.metric("Before filtering", len(evidence))
    c2.metric("After filtering", len(clean_evidence))
    c3.metric("Removed", max(0, len(evidence) - len(clean_evidence)))

    st.subheader("Removed-context behavior")
    st.write(
        "Filtering is adaptive. It does not keep a fixed number of chunks. "
        "Evidence is protected when it supports the question/baseline answer, "
        "and only redundant or clearly weak context is removed."
    )

    with st.expander("Retrieved evidence"):
        for i, chunk in enumerate(evidence, 1):
            st.markdown(f"**Evidence {i}**")
            st.write(rag.get_chunk_text(chunk))

    with st.expander("Kept evidence"):
        for i, chunk in enumerate(clean_evidence, 1):
            st.markdown(f"**Evidence {i}**")
            st.write(rag.get_chunk_text(chunk))


def render_analytics():
    st.title("📊 Analytics")

    records = rag.load_saved_evaluation_records()

    if not records:
        st.info("No saved evaluations yet.")
        return

    rows = []
    for record in records:
        rows.append({
            "Question": record.get("question", ""),
            "Baseline Score": safe_float(record.get("baseline_score")),
            "Proposed Score": safe_float(record.get("proposed_score")),
            "Retrieved": record.get("retrieved_chunks", ""),
            "Kept": record.get("kept_chunks", ""),
            "Removed": record.get("removed_chunks", ""),
            "Verified": record.get("verified", ""),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    metrics = rag.calculate_cumulative_metrics()

    st.subheader("Cumulative Answer Quality")
    b = metrics["baseline"]
    p = metrics["proposed"]

    table = pd.DataFrame([
        {
            "Metric": "Accuracy",
            "Before": pct(b["accuracy"]),
            "After": pct(p["accuracy"]),
            "Improvement": signed_pct(p["accuracy"] - b["accuracy"]),
        },
        {
            "Metric": "Precision",
            "Before": pct(b["precision"]),
            "After": pct(p["precision"]),
            "Improvement": signed_pct(p["precision"] - b["precision"]),
        },
        {
            "Metric": "Recall",
            "Before": pct(b["recall"]),
            "After": pct(p["recall"]),
            "Improvement": signed_pct(p["recall"] - b["recall"]),
        },
        {
            "Metric": "F1-score",
            "Before": pct(b["f1_score"]),
            "After": pct(p["f1_score"]),
            "Improvement": signed_pct(p["f1_score"] - b["f1_score"]),
        },
    ])

    st.dataframe(table, use_container_width=True, hide_index=True)


def render_architecture():
    st.title("🏗️ System Architecture")

    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            PDF -> Chunking -> BGE -> FAISS -> Retrieval -> Filtering;
            Filtering -> "Before / After";
            "Before / After" -> Generation;
            Generation -> Evaluation;
            Evaluation -> Metrics;
            "Case Study" -> PDF;
            "Case Study" -> "Domain Output";
            Generation -> "Domain Output";
        }
        """
    )

    st.markdown(
        """
        ### Shared pipeline

        **Case study → imported PDF → chunking → BGE embeddings → FAISS →
        retrieval → adaptive evidence filtering → generation → evaluation**

        General, Medical and Legal use the same retrieval infrastructure.
        Medical and Legal only add document-grounded output formatting after
        the shared RAG result.
        """
    )


def main():
    case_study = render_sidebar()

    pages = [
        "Chat",
        "Document Overview",
        "Retrieval Analysis",
        "Contamination Analysis",
        "Analytics",
        "System Architecture",
    ]

    page = st.sidebar.radio("Navigation", pages)

    if page == "Chat":
        render_chat(case_study)
    elif page == "Document Overview":
        render_overview(case_study)
    elif page == "Retrieval Analysis":
        render_retrieval_analysis()
    elif page == "Contamination Analysis":
        render_contamination()
    elif page == "Analytics":
        render_analytics()
    else:
        render_architecture()


if __name__ == "__main__":
    main()
