
import os

import requests
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

REQUEST_TIMEOUT = 600

st.set_page_config(
    page_title="MyManah AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------


st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f8fc;
        color: #1f2937;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Make all main content text readable */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp p,
    .stApp label,
    .stApp span,
    .stApp li,
    .stApp [data-testid="stMarkdownContainer"] {
        color: #1f2937;
    }

    .hero {
        padding: 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #312e81, #6366f1);
        color: #ffffff;
        margin-bottom: 1.5rem;
    }

    .hero h1,
    .hero p {
        color: #ffffff !important;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1f2937 !important;
    }

    /* Inputs */
    .stTextArea textarea,
    .stTextInput input {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border: 1px solid #cbd5e1 !important;
    }

    /* Improve button readability */
    .stButton button,
    .stFormSubmitButton button {
        border-radius: 10px;
        font-weight: 600;
    }

    .disclaimer {
        color: #64748b !important;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# API helpers
# --------------------------------------------------

def api_get(path):
    return requests.get(
        f"{API_BASE_URL}{path}",
        timeout=10,
    )


def api_post_json(path, payload):
    return requests.post(
        f"{API_BASE_URL}{path}",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )


def display_api_error(response):
    try:
        detail = response.json().get(
            "detail",
            response.text,
        )
    except ValueError:
        detail = response.text

    st.error(
        f"API error ({response.status_code}): {detail}"
    )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("🧠 MyManah AI")
    st.caption("Local AI-powered journal and document assistant")

    page = st.radio(
        "Navigate",
        [
            "Journal Analysis",
            "Document Q&A",
        ],
    )

    st.divider()

    st.markdown("**Backend connection**")

    try:
        health_response = api_get("/health")

        if health_response.ok:
            st.success("FastAPI connected")
        else:
            st.warning("Backend responded with an error")

    except requests.RequestException:
        st.error("Backend is not reachable")
        st.caption(
            "Start FastAPI on port 8000 to use the app locally."
        )

    st.divider()

    st.caption(
        "Prototype only. AI predictions can be incorrect."
    )


# --------------------------------------------------
# Hero
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>MyManah AI</h1>
        <p>
            Understand journal entries and ask questions
            about your documents using local AI models.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# PAGE 1: JOURNAL ANALYSIS
# ==================================================

if page == "Journal Analysis":

    st.header("📔 Journal Analysis")

    st.write(
        "Write about your day to generate AI-powered insights."
    )

    with st.form("journal_form"):
        journal_text = st.text_area(
            "Your journal entry",
            height=200,
            placeholder=(
                "Today was a difficult day at work. "
                "I felt stressed about my deadlines, "
                "but I managed to finish my important tasks..."
            ),
        )

        analyze_button = st.form_submit_button(
            "✨ Analyze journal",
            type="primary",
            use_container_width=True,
        )

    if analyze_button:
        if not journal_text.strip():
            st.warning("Please enter a journal entry first.")

        else:
            try:
                with st.spinner(
                    "Analyzing your journal with local AI models..."
                ):
                    response = api_post_json(
                        "/analyze-journal",
                        {"text": journal_text.strip()},
                    )

                if response.ok:
                    result = response.json()

                    st.subheader("Your analysis")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Sentiment",
                            str(result.get("sentiment", "N/A")).title(),
                        )

                    with col2:
                        st.metric(
                            "Emotion",
                            str(result.get("emotion", "N/A")).title(),
                        )

                    with col3:
                        mood = result.get("moodScore", "N/A")
                        st.metric("Mood score", f"{mood}/10")

                    st.subheader("📝 Summary")
                    st.write(
                        result.get(
                            "summary",
                            "No summary returned.",
                        )
                    )

                    st.subheader("🛡️ Crisis-risk screening")

                    risk = str(
                        result.get("crisisRisk", "N/A")
                    ).upper()

                    if risk == "HIGH":
                        st.error(f"Screening result: {risk}")
                    elif risk == "MEDIUM":
                        st.warning(f"Screening result: {risk}")
                    else:
                        st.info(f"Screening result: {risk}")

                    confidence = result.get("confidence")

                    if confidence is not None:
                        st.caption(
                            f"Model confidence: {float(confidence):.2%}"
                        )

                    st.caption(
                        "These are model-generated estimates, "
                        "not a clinical diagnosis. Crisis screening "
                        "may produce false positives or false negatives."
                    )

                    with st.expander("View raw API response"):
                        st.json(result)

                else:
                    display_api_error(response)

            except requests.RequestException as exc:
                st.error(
                    "Could not reach the FastAPI backend. "
                    "Check that the backend is running."
                )
                st.caption(str(exc))


# ==================================================
# PAGE 2: DOCUMENT Q&A
# ==================================================

elif page == "Document Q&A":

    st.header("📚 Document Q&A")

    st.write(
        "Upload a PDF, index its content, and ask questions "
        "using document-grounded retrieval."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        help="Upload a text-based PDF for indexing.",
    )

    if uploaded_file is not None:

        st.caption(
            f"Selected file: {uploaded_file.name} "
            f"({uploaded_file.size / 1024:.1f} KB)"
        )

        if st.button(
            "📥 Upload and index PDF",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Extracting PDF text and creating the vector index..."
                ):
                    response = requests.post(
                        f"{API_BASE_URL}/rag/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        },
                        timeout=REQUEST_TIMEOUT,
                    )

                if response.ok:
                    result = response.json()

                    st.session_state["rag_indexed"] = True
                    st.session_state["rag_filename"] = uploaded_file.name

                    st.success(
                        result.get(
                            "message",
                            "PDF indexed successfully.",
                        )
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Pages",
                            result.get("pages", "N/A"),
                        )

                    with col2:
                        st.metric(
                            "Chunks",
                            result.get("chunks", "N/A"),
                        )

                else:
                    display_api_error(response)

            except requests.RequestException as exc:
                st.error("PDF upload failed.")
                st.caption(str(exc))

    if st.session_state.get("rag_indexed", False):
        st.success(
            "Indexed document: "
            + st.session_state.get("rag_filename", "PDF")
        )

    st.divider()

    st.subheader("💬 Ask your document")

    with st.form("rag_query_form"):
        question = st.text_input(
            "Your question",
            placeholder="What is deep learning?",
        )

        ask_button = st.form_submit_button(
            "Ask question",
            type="primary",
            use_container_width=True,
        )

    if ask_button:
        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner(
                    "Retrieving relevant passages and generating an answer..."
                ):
                    response = api_post_json(
                        "/rag/query",
                        {"question": question.strip()},
                    )

                if response.ok:
                    result = response.json()

                    st.subheader("Answer")
                    st.write(
                        result.get(
                            "answer",
                            "No answer returned.",
                        )
                    )

                    sources = result.get("sources", [])

                    if sources:
                        st.subheader("📖 Retrieved sources")

                        for index, source in enumerate(sources, 1):
                            page_number = source.get("page", "N/A")
                            chunk_number = source.get("chunk", "N/A")
                            similarity = source.get("similarity")

                            details = (
                                f"Source {index} — Page {page_number}, "
                                f"Chunk {chunk_number}"
                            )

                            if similarity is not None:
                                details += (
                                    f", Similarity: {float(similarity):.3f}"
                                )

                            st.write(f"- {details}")

                    with st.expander("View raw API response"):
                        st.json(result)

                else:
                    display_api_error(response)

            except requests.RequestException as exc:
                st.error("Could not retrieve an answer.")
                st.caption(str(exc))


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <p class="disclaimer">
        MyManah AI is a prototype for journal analysis and document Q&A.
        It can make mistakes and should not be used as a substitute
        for professional mental-health assessment.
    </p>
    """,
    unsafe_allow_html=True,
)
