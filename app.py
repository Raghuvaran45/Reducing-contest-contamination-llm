import streamlit as st
import pandas as pd


# ============================================================
# IMPORT RETRIEVAL PIPELINE
# ============================================================

from retrieval.retrieval_pipeline import (
    answer_question,
    get_document_info,
    get_document_statistics,
    inspect_retrieval,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PDF RAG Intelligence Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main-title {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 16px;
    color: #777;
    margin-bottom: 25px;
}

.metric-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.2);
    text-align: center;
}

.metric-number {
    font-size: 28px;
    font-weight: 700;
}

.metric-label {
    font-size: 14px;
    color: #777;
}

.chat-user {
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 8px;
    border: 1px solid rgba(128,128,128,0.2);
}

.chat-assistant {
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
    border: 1px solid rgba(128,128,128,0.2);
}

.small-text {
    font-size: 13px;
    color: #777;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "analytics" not in st.session_state:
    st.session_state.analytics = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# LOAD DOCUMENT INFORMATION
# ============================================================

@st.cache_resource
def load_document_info():

    return get_document_info()


@st.cache_resource
def load_document_statistics():

    return get_document_statistics()


document_info = load_document_info()
document_stats = load_document_statistics()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📚 RAG Dashboard")

    st.caption(
        "PDF Question Answering and Context Contamination Analysis"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "💬 Chat",
            "📄 Document Overview",
            "🔎 Retrieval Analysis",
            "🧠 Contamination Analysis",
            "📊 Analytics",
            "🏗️ System Architecture"
        ]
    )

    st.divider()

    st.subheader("System")

    st.write(
        f"FAISS vectors: {document_info['vectors']}"
    )

    st.write(
        f"Chunks: {document_info['chunks']}"
    )

    st.write(
        "Embedding: BGE-small-en-v1.5"
    )

    st.write(
        "Generator: Gemini"
    )

    st.divider()

    if st.button(
        "Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.analytics = []
        st.session_state.last_result = None

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">PDF RAG Intelligence Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Retrieval-Augmented Generation with Context Contamination Detection'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CHAT PAGE
# ============================================================

if page == "💬 Chat":

    st.header("💬 Ask Questions About the PDF")

    st.info(
        "Ask any question about the uploaded PDF."
    )

    # --------------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.messages:

        if message["role"] == "user":

            with st.chat_message("user"):

                st.write(
                    message["content"]
                )

        else:

            with st.chat_message("assistant"):

                st.write(
                    message["content"]
                )

                if "metadata" in message:

                    metadata = message["metadata"]

                    cols = st.columns(5)

                    with cols[0]:

                        st.caption(
                            f"Type: {metadata['question_type']}"
                        )

                    with cols[1]:

                        st.caption(
                            f"Retrieved: "
                            f"{metadata['retrieved_count']}"
                        )

                    with cols[2]:

                        st.caption(
                            f"Filtered: "
                            f"{metadata['clean_count']}"
                        )

                    with cols[3]:

                        st.caption(
                            f"Contamination: "
                            f"{metadata['contamination_rate']:.1f}%"
                        )

                    with cols[4]:

                        st.caption(
                            f"Time: "
                            f"{metadata['response_time']:.2f}s"
                        )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about the PDF..."
    )

    if question:

        question = question.strip()

        if question:

            # ------------------------------------------------
            # SAVE USER MESSAGE
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):

                st.write(
                    question
                )

            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            with st.chat_message("assistant"):

                with st.spinner(
                    "Retrieving and analyzing the PDF..."
                ):

                    result = answer_question(
                        question,
                        return_metadata=True
                    )

                answer = result["answer"]

                st.write(
                    answer
                )

                # --------------------------------------------
                # METRICS
                # --------------------------------------------

                cols = st.columns(5)

                with cols[0]:

                    st.caption(
                        f"Type: "
                        f"{result['question_type']}"
                    )

                with cols[1]:

                    st.caption(
                        f"Retrieved: "
                        f"{result['retrieved_count']}"
                    )

                with cols[2]:

                    st.caption(
                        f"Filtered: "
                        f"{result['clean_count']}"
                    )

                with cols[3]:

                    st.caption(
                        f"Contamination: "
                        f"{result['contamination_rate']:.1f}%"
                    )

                with cols[4]:

                    st.caption(
                        f"Time: "
                        f"{result['response_time']:.2f}s"
                    )

                # --------------------------------------------
                # VERIFICATION STATUS
                # --------------------------------------------

                if result["verified"]:

                    st.success(
                        "Answer verified against PDF evidence."
                    )

                else:

                    st.warning(
                        "Answer could not be fully verified."
                    )

            # ------------------------------------------------
            # SAVE ASSISTANT MESSAGE
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "metadata": result
                }
            )

            # ------------------------------------------------
            # SAVE ANALYTICS
            # ------------------------------------------------

            st.session_state.analytics.append(
                {
                    "question": question,
                    "question_type":
                        result["question_type"],
                    "retrieved":
                        result["retrieved_count"],
                    "filtered":
                        result["clean_count"],
                    "contamination":
                        result["contamination_rate"],
                    "response_time":
                        result["response_time"],
                    "verified":
                        result["verified"]
                }
            )

            st.session_state.last_result = result

            st.rerun()


# ============================================================
# DOCUMENT OVERVIEW
# ============================================================

elif page == "📄 Document Overview":

    st.header("📄 Document Overview")

    st.subheader(
        document_info["title"]
    )

    st.write(
        f"Authors: {document_info['authors']}"
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Chunks",
            document_info["chunks"]
        )

    with col2:

        st.metric(
            "FAISS Vectors",
            document_info["vectors"]
        )

    with col3:

        reference_text = document_info["references"]

        if reference_text:

            reference_number = (
                reference_text
                .replace(
                    "The PDF contains ",
                    ""
                )
                .replace(
                    " references.",
                    ""
                )
            )

        else:

            reference_number = "N/A"

        st.metric(
            "References",
            reference_number
        )

    with col4:

        st.metric(
            "Avg Chunk Length",
            f"{document_stats['average_chunk_length']:.0f}"
        )

    st.divider()

    st.subheader(
        "Document Statistics"
    )

    stats_df = pd.DataFrame(
        {
            "Metric": [
                "Total chunks",
                "Average chunk length",
                "Minimum chunk length",
                "Maximum chunk length"
            ],

            "Value": [
                document_stats[
                    "total_chunks"
                ],

                round(
                    document_stats[
                        "average_chunk_length"
                    ],
                    2
                ),

                document_stats[
                    "min_chunk_length"
                ],

                document_stats[
                    "max_chunk_length"
                ]
            ]
        }
    )

    st.dataframe(
        stats_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Document Components"
    )

    components = pd.DataFrame(
        {
            "Component": [
                "PDF document",
                "Text chunks",
                "Embedding model",
                "FAISS vector index",
                "Question classifier",
                "Semantic retrieval",
                "Context contamination filter",
                "Gemini answer generation",
                "Answer verification"
            ],

            "Status": [
                "Loaded",
                "Loaded",
                "Loaded",
                "Loaded",
                "Active",
                "Active",
                "Active",
                "Active",
                "Active"
            ]
        }
    )

    st.dataframe(
        components,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RETRIEVAL ANALYSIS
# ============================================================

elif page == "🔎 Retrieval Analysis":

    st.header(
        "🔎 Retrieval Analysis"
    )

    retrieval_question = st.text_input(
        "Enter a question to inspect retrieval:",
        placeholder="What is RAG?"
    )

    if retrieval_question:

        with st.spinner(
            "Searching FAISS..."
        ):

            retrieval_results = inspect_retrieval(
                retrieval_question
            )

        if retrieval_results:

            st.subheader(
                "Top Retrieved Chunks"
            )

            rows = []

            for item in retrieval_results:

                rows.append(
                    {
                        "Rank":
                            item["rank"],

                        "Similarity Score":
                            round(
                                item["score"],
                                4
                            ),

                        "Chunk Index":
                            item["chunk_index"],

                        "Text":
                            item["text"][:500]
                    }
                )

            retrieval_df = pd.DataFrame(
                rows
            )

            st.dataframe(
                retrieval_df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader(
                "Similarity Score Comparison"
            )

            chart_df = retrieval_df[
                [
                    "Rank",
                    "Similarity Score"
                ]
            ].copy()

            chart_df = chart_df.set_index(
                "Rank"
            )

            st.bar_chart(
                chart_df
            )

            st.divider()

            st.subheader(
                "Retrieved Evidence"
            )

            for item in retrieval_results:

                with st.expander(
                    f"Rank {item['rank']} | "
                    f"Score {item['score']:.4f}"
                ):

                    st.write(
                        item["text"]
                    )

        else:

            st.warning(
                "No retrieval results found."
            )


# ============================================================
# CONTAMINATION ANALYSIS
# ============================================================

elif page == "🧠 Contamination Analysis":

    st.header(
        "🧠 Context Contamination Analysis"
    )

    st.write(
        "This page compares retrieved context before "
        "and after contamination filtering."
    )

    if not st.session_state.analytics:

        st.info(
            "Ask at least one question in the Chat page "
            "to generate contamination analysis."
        )

    else:

        analytics_df = pd.DataFrame(
            st.session_state.analytics
        )

        latest = analytics_df.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Retrieved Chunks",
                int(
                    latest["retrieved"]
                )
            )

        with col2:

            st.metric(
                "Relevant Chunks",
                int(
                    latest["filtered"]
                )
            )

        with col3:

            st.metric(
                "Contamination",
                f"{latest['contamination']:.1f}%"
            )

        with col4:

            st.metric(
                "Verified",
                "YES"
                if latest["verified"]
                else "NO"
            )

        st.divider()

        st.subheader(
            "Before vs After Filtering"
        )

        comparison_df = pd.DataFrame(
            {
                "Stage": [
                    "Retrieved",
                    "After filtering"
                ],

                "Chunks": [
                    int(
                        latest["retrieved"]
                    ),

                    int(
                        latest["filtered"]
                    )
                ]
            }
        )

        st.bar_chart(
            comparison_df.set_index(
                "Stage"
            )
        )

        st.divider()

        st.subheader(
            "Contamination Rate"
        )

        contamination_value = float(
            latest["contamination"]
        )

        st.progress(
            min(
                max(
                    contamination_value / 100,
                    0
                ),
                1
            )
        )

        st.write(
            f"Estimated contamination rate: "
            f"{contamination_value:.2f}%"
        )

        st.divider()

        if st.session_state.last_result:

            result = (
                st.session_state.last_result
            )

            st.subheader(
                "Retrieved Context"
            )

            for i, chunk in enumerate(
                result["evidence"]
            ):

                with st.expander(
                    f"Retrieved Chunk {i + 1}"
                ):

                    if isinstance(
                        chunk,
                        dict
                    ):

                        text = chunk.get(
                            "text",
                            ""
                        )

                    else:

                        text = str(chunk)

                    st.write(
                        text
                    )

            st.subheader(
                "Filtered Context"
            )

            for i, chunk in enumerate(
                result["clean_evidence"]
            ):

                with st.expander(
                    f"Relevant Chunk {i + 1}"
                ):

                    if isinstance(
                        chunk,
                        dict
                    ):

                        text = chunk.get(
                            "text",
                            ""
                        )

                    else:

                        text = str(chunk)

                    st.write(
                        text
                    )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.header(
        "📊 System Analytics"
    )

    if not st.session_state.analytics:

        st.info(
            "Ask questions in the Chat page "
            "to generate analytics."
        )

    else:

        analytics_df = pd.DataFrame(
            st.session_state.analytics
        )

        total_questions = len(
            analytics_df
        )

        average_time = analytics_df[
            "response_time"
        ].mean()

        average_contamination = analytics_df[
            "contamination"
        ].mean()

        verification_rate = (
            analytics_df[
                "verified"
            ].mean()
            * 100
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Questions",
                total_questions
            )

        with col2:

            st.metric(
                "Average Response Time",
                f"{average_time:.2f}s"
            )

        with col3:

            st.metric(
                "Average Contamination",
                f"{average_contamination:.1f}%"
            )

        with col4:

            st.metric(
                "Verification Rate",
                f"{verification_rate:.1f}%"
            )

        st.divider()

        # ----------------------------------------------------
        # QUESTION TYPES
        # ----------------------------------------------------

        st.subheader(
            "Question Type Distribution"
        )

        type_counts = (
            analytics_df[
                "question_type"
            ]
            .value_counts()
            .rename("Count")
        )

        st.bar_chart(
            type_counts
        )

        st.divider()

        # ----------------------------------------------------
        # RESPONSE TIME
        # ----------------------------------------------------

        st.subheader(
            "Response Time by Question"
        )

        response_df = analytics_df[
            [
                "response_time"
            ]
        ].copy()

        response_df[
            "Question"
        ] = range(
            1,
            len(response_df) + 1
        )

        response_df = response_df.set_index(
            "Question"
        )

        st.line_chart(
            response_df[
                "response_time"
            ]
        )

        st.divider()

        # ----------------------------------------------------
        # CONTAMINATION
        # ----------------------------------------------------

        st.subheader(
            "Contamination Rate by Question"
        )

        contamination_df = analytics_df[
            [
                "contamination"
            ]
        ].copy()

        contamination_df[
            "Question"
        ] = range(
            1,
            len(contamination_df) + 1
        )

        contamination_df = (
            contamination_df
            .set_index(
                "Question"
            )
        )

        st.line_chart(
            contamination_df[
                "contamination"
            ]
        )

        st.divider()

        # ----------------------------------------------------
        # RETRIEVED VS FILTERED
        # ----------------------------------------------------

        st.subheader(
            "Retrieved vs Filtered Context"
        )

        comparison = analytics_df[
            [
                "retrieved",
                "filtered"
            ]
        ].copy()

        comparison[
            "Question"
        ] = range(
            1,
            len(comparison) + 1
        )

        comparison = comparison.set_index(
            "Question"
        )

        st.bar_chart(
            comparison
        )

        st.divider()

        st.subheader(
            "Complete Query Analytics"
        )

        st.dataframe(
            analytics_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

elif page == "🏗️ System Architecture":

    st.header(
        "🏗️ RAG System Architecture"
    )

    st.write(
        "The complete processing pipeline of the capstone system."
    )

    architecture = [

        "1. User Question",

        "↓",

        "2. Streamlit Chat Interface",

        "↓",

        "3. Question Type Detection",

        "↓",

        "4. Query Embedding",

        "↓",

        "5. FAISS Semantic Retrieval",

        "↓",

        "6. Section / Multi-Query Retrieval",

        "↓",

        "7. Context Contamination Detection",

        "↓",

        "8. Relevant Context Selection",

        "↓",

        "9. Gemini Answer Generation",

        "↓",

        "10. Answer Verification",

        "↓",

        "11. Final Answer"
    ]

    for item in architecture:

        if item == "↓":

            st.markdown(
                """
                <div style="
                    text-align:center;
                    font-size:25px;
                    padding:5px;
                ">
                    ↓
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div style="
                    padding:15px;
                    margin:5px;
                    border-radius:10px;
                    border:1px solid rgba(128,128,128,0.3);
                    text-align:center;
                    font-weight:600;
                ">
                    {item}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    st.subheader(
        "Technology Stack"
    )

    technology_df = pd.DataFrame(
        {
            "Layer": [
                "Interface",
                "Programming",
                "Embeddings",
                "Vector Database",
                "Retrieval",
                "LLM",
                "Context Filtering",
                "Verification",
                "Visualization"
            ],

            "Technology": [
                "Streamlit",
                "Python",
                "BAAI/bge-small-en-v1.5",
                "FAISS",
                "Semantic + Multi-query Retrieval",
                "Google Gemini",
                "Gemini-based Context Filtering",
                "Gemini-based Verification",
                "Streamlit Charts"
            ]
        }
    )

    st.dataframe(
        technology_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Core Capstone Innovation"
    )

    st.write(
        """
        The system does not directly send every retrieved chunk
        to the language model. It first evaluates the retrieved
        context and removes evidence that is unrelated to the
        user's question. This reduces context contamination and
        allows the final answer to be generated from more relevant
        evidence.
        """
    )