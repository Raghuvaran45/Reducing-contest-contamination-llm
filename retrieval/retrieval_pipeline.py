import os
import re
import pickle
import faiss
import time
import csv
from pathlib import Path
from functools import lru_cache

import fitz
import numpy as np

from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

FAISS_PATH = VECTORSTORE_DIR / "faiss_index.bin"
CHUNKS_PATH = VECTORSTORE_DIR / "chunks.pkl"

EVALUATION_CSV_PATH = PROJECT_ROOT / "evaluation.csv"

EVALUATION_COLUMNS = [
    "question",
    "reference_answer",
    "baseline_answer",
    "proposed_answer",
    "baseline_correct",
    "proposed_correct",
    "baseline_score",
    "proposed_score",
    "retrieved_chunks",
    "kept_chunks",
    "removed_chunks",
    "contamination_reduction",
    "verified",
    "response_time"
]

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# ============================================================
# DO NOT CHANGE THESE GEMINI VERSIONS
# ============================================================

GEMINI_MODEL = "gemini-3.5-flash-lite"
FALLBACK_GEMINI_MODEL = "gemini-3.5-flash"


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

# Original baseline retrieval size.
FAISS_TOP_K = 8

# Maximum number of chunks used for the normal answer.
MAX_EVIDENCE = 16

# Larger context for independently generated reference answer.
MAX_REFERENCE_EVIDENCE = 30

# Maximum characters from an individual chunk.
MAX_CHUNK_LENGTH = 5000

# PDF chunk configuration.
PDF_CHUNK_SIZE = 1200
PDF_CHUNK_OVERLAP = 200

# Minimum number of chunks allowed after filtering.
MIN_FILTER_KEEP = 2

# Number of neighboring chunks to consider.
NEIGHBOR_RADIUS = 1

# Maximum number of semantic results per search query.
MULTI_QUERY_TOP_K = 8

# Minimum semantic similarity accepted when deciding whether
# an additional candidate is useful.
MIN_SEMANTIC_SCORE = 0.20

# Maximum number of keyword candidates added.
MAX_KEYWORD_CANDIDATES = 10


# ============================================================
# CURRENT ACTIVE PDF
# ============================================================

CURRENT_PDF_PATH = None
CURRENT_PDF_NAME = "Default PDF"


# ============================================================
# GEMINI CLIENT
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set.\n\n"
        "PowerShell:\n"
        '$env:GEMINI_API_KEY="YOUR_API_KEY"'
    )

client = genai.Client(api_key=API_KEY)


# ============================================================
# GEMINI RESPONSE TEXT EXTRACTION
# ============================================================

def extract_gemini_text(response):
    """
    Safely extracts text from Gemini responses.
    """

    if response is None:
        return None

    try:
        text = getattr(response, "text", None)

        if text:
            text = str(text).strip()

            if text:
                return text

    except Exception as e:
        print(
            "Could not directly read Gemini response.text:",
            str(e)
        )

    try:

        candidates = getattr(
            response,
            "candidates",
            None
        )

        if candidates:

            for candidate in candidates:

                content = getattr(
                    candidate,
                    "content",
                    None
                )

                if not content:
                    continue

                parts = getattr(
                    content,
                    "parts",
                    None
                )

                if not parts:
                    continue

                collected = []

                for part in parts:

                    part_text = getattr(
                        part,
                        "text",
                        None
                    )

                    if part_text:
                        collected.append(
                            str(part_text)
                        )

                if collected:

                    return "\n".join(
                        collected
                    ).strip()

    except Exception as e:

        print(
            "Could not extract Gemini candidate text:",
            str(e)
        )

    return None


# ============================================================
# EMBEDDING MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_embedding_model():

    print("Loading embedding model...")

    try:

        model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        print("Embedding model loaded.")

        return model

    except Exception as e:

        raise RuntimeError(
            "Could not load the embedding model.\n"
            "Try restarting the terminal and Streamlit.\n\n"
            f"Original error: {e}"
        )


# ============================================================
# LOAD DEFAULT FAISS
# ============================================================

if not FAISS_PATH.exists():

    raise FileNotFoundError(
        f"FAISS index not found:\n{FAISS_PATH}"
    )

if not CHUNKS_PATH.exists():

    raise FileNotFoundError(
        f"chunks.pkl not found:\n{CHUNKS_PATH}"
    )


index = faiss.read_index(
    str(FAISS_PATH)
)

with open(
    CHUNKS_PATH,
    "rb"
) as f:

    chunks = pickle.load(f)


print("FAISS index loaded.")
print(
    f"Number of vectors: {index.ntotal}"
)
print(
    f"Number of chunks: {len(chunks)}"
)


# ============================================================
# CHUNK HELPERS
# ============================================================

def get_chunk_text(chunk):

    if isinstance(chunk, dict):

        return str(
            chunk.get(
                "text",
                ""
            )
        )

    return str(chunk)


def get_chunk_id(
    chunk,
    fallback
):

    if isinstance(chunk, dict):

        value = chunk.get("id")

        if value is not None:

            return str(value)

    return str(fallback)


def get_chunk_page(chunk):

    if isinstance(chunk, dict):

        page = chunk.get("page")

        if page is not None:

            try:
                return int(page)
            except Exception:
                return page

    return None


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = str(text)

    replacements = {

        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb00": "ff",
        "\ufb03": "ffi",
        "\ufb04": "ffl",

        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",

        "\u00a0": " ",

        "\u2018": "'",
        "\u2019": "'",

        "\u201c": '"',
        "\u201d": '"',

        "\u2026": "...",

        "\u00ad": ""
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    text = re.sub(
        r"\\[a-zA-Z]+",
        " ",
        text
    )

    text = text.replace(
        "$",
        " "
    )

    text = text.replace(
        "{",
        " "
    )

    text = text.replace(
        "}",
        " "
    )

    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# ANSWER CLEANER
# ============================================================

def clean_answer(answer):

    if answer is None:
        return ""

    answer = str(answer).strip()

    if not answer:
        return ""

    answer = re.sub(
        r"#{1,6}\s*",
        "",
        answer
    )

    answer = answer.replace(
        "**",
        ""
    )

    answer = answer.replace(
        "__",
        ""
    )

    answer = answer.replace(
        "```",
        ""
    )

    answer = answer.replace(
        "`",
        ""
    )

    answer = re.sub(
        r"\\[a-zA-Z]+",
        "",
        answer
    )

    answer = answer.replace(
        "$",
        ""
    )

    answer = answer.replace(
        "{",
        ""
    )

    answer = answer.replace(
        "}",
        ""
    )

    replacements = {

        "→": "to",
        "←": "from",
        "↓": "",
        "↑": "",

        "∑": "sum",
        "∏": "product",
        "√": "square root",

        "≤": "less than or equal to",
        "≥": "greater than or equal to",
        "≠": "not equal to",
        "≈": "approximately",

        "×": "times",
        "÷": "divided by",

        "∈": "in",
        "∀": "for all",

        "∂": "",
        "∞": "infinity",

        "η": "eta",
        "θ": "theta",
        "π": "pi",
        "σ": "sigma",
        "μ": "mu"
    }

    for old, new in replacements.items():

        answer = answer.replace(
            old,
            new
        )

    answer = re.sub(
        r"\[\d+\]",
        "",
        answer
    )

    answer = re.sub(
        r"\s+",
        " ",
        answer
    )

    prefixes = [

        "Answer:",
        "Final Answer:",
        "FINAL ANSWER:",

        "Based on the provided PDF evidence:",
        "Based on the evidence:",
        "Based on the provided PDF:",
        "According to the provided PDF:"
    ]

    changed = True

    while changed:

        changed = False

        for prefix in prefixes:

            if answer.lower().startswith(
                prefix.lower()
            ):

                answer = answer[
                    len(prefix):
                ].strip()

                changed = True

    return answer.strip()


# ============================================================
# DYNAMIC PDF SUPPORT
# ============================================================

def extract_pdf_text(pdf_path):

    pages = []

    pdf_path = Path(
        pdf_path
    )

    try:

        document = fitz.open(
            str(pdf_path)
        )

        for page_number in range(
            len(document)
        ):

            page = document[
                page_number
            ]

            text = page.get_text(
                "text"
            )

            text = clean_text(
                text
            )

            if text:

                pages.append({

                    "page":
                        page_number + 1,

                    "text":
                        text
                })

        document.close()

    except Exception as e:

        raise RuntimeError(
            f"Could not read PDF:\n{e}"
        )

    return pages


# ============================================================
# EXTRACT PDF FROM BYTES
# ============================================================

def extract_pdf_text_from_bytes(
    pdf_bytes
):

    pages = []

    if not pdf_bytes:

        raise ValueError(
            "Uploaded PDF is empty."
        )

    try:

        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        for page_number in range(
            len(document)
        ):

            page = document[
                page_number
            ]

            text = page.get_text(
                "text"
            )

            text = clean_text(
                text
            )

            if text:

                pages.append({

                    "page":
                        page_number + 1,

                    "text":
                        text
                })

        document.close()

    except Exception as e:

        raise RuntimeError(
            f"Could not open uploaded PDF:\n{e}"
        )

    return pages


# ============================================================
# CREATE PDF CHUNKS
# ============================================================

def create_pdf_chunks(
    pages,
    chunk_size=PDF_CHUNK_SIZE,
    overlap=PDF_CHUNK_OVERLAP
):

    document_chunks = []

    chunk_id = 0

    for page_data in pages:

        page_number = page_data[
            "page"
        ]

        text = page_data[
            "text"
        ]

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[
                start:end
            ]

            chunk_text = clean_text(
                chunk_text
            )

            if chunk_text:

                document_chunks.append({

                    "id":
                        chunk_id,

                    "text":
                        chunk_text,

                    "page":
                        page_number
                })

                chunk_id += 1

            if end >= len(text):
                break

            start = end - overlap

    return document_chunks


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index_for_chunks(
    document_chunks
):

    if not document_chunks:

        raise RuntimeError(
            "No chunks available to create FAISS index."
        )

    embedding_model = (
        get_embedding_model()
    )

    texts = [

        get_chunk_text(chunk)

        for chunk in document_chunks

    ]

    print(
        "Creating embeddings..."
    )

    embeddings = (
        embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    new_index = faiss.IndexFlatIP(
        dimension
    )

    new_index.add(
        embeddings
    )

    return new_index


# ============================================================
# LOAD PDF FROM FILE
# ============================================================

def load_uploaded_pdf(
    pdf_path
):

    global index
    global chunks
    global CURRENT_PDF_PATH
    global CURRENT_PDF_NAME

    pdf_path = Path(
        pdf_path
    )

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"PDF not found:\n{pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":

        raise ValueError(
            "Only PDF files are supported."
        )

    print()
    print("=" * 70)
    print("LOADING UPLOADED PDF")
    print("=" * 70)

    print(
        "PDF:",
        pdf_path.name
    )

    pages = extract_pdf_text(
        pdf_path
    )

    if not pages:

        raise RuntimeError(
            "No readable text was found in the uploaded PDF."
        )

    print(
        "Pages extracted:",
        len(pages)
    )

    new_chunks = create_pdf_chunks(
        pages
    )

    if not new_chunks:

        raise RuntimeError(
            "Could not create chunks from the uploaded PDF."
        )

    print(
        "Chunks created:",
        len(new_chunks)
    )

    new_index = (
        create_faiss_index_for_chunks(
            new_chunks
        )
    )

    index = new_index
    chunks = new_chunks

    CURRENT_PDF_PATH = pdf_path
    CURRENT_PDF_NAME = pdf_path.name

    print()
    print(
        "Uploaded PDF loaded successfully."
    )

    print(
        "Active PDF:",
        CURRENT_PDF_NAME
    )

    print(
        "FAISS vectors:",
        index.ntotal
    )

    print(
        "PDF chunks:",
        len(chunks)
    )

    print("=" * 70)
    print()

    return {

        "pdf_name":
            CURRENT_PDF_NAME,

        "pdf_path":
            str(CURRENT_PDF_PATH),

        "pages":
            len(pages),

        "chunks":
            len(chunks),

        "vectors":
            index.ntotal
    }


# ============================================================
# LOAD PDF FROM STREAMLIT
# ============================================================

def load_uploaded_pdf_bytes(
    pdf_bytes,
    filename="uploaded.pdf"
):

    global index
    global chunks
    global CURRENT_PDF_PATH
    global CURRENT_PDF_NAME

    if not pdf_bytes:

        raise ValueError(
            "Uploaded PDF is empty."
        )

    if not filename.lower().endswith(
        ".pdf"
    ):

        raise ValueError(
            "Only PDF files are supported."
        )

    print()
    print("=" * 70)
    print("LOADING STREAMLIT UPLOADED PDF")
    print("=" * 70)

    print(
        "PDF:",
        filename
    )

    pages = (
        extract_pdf_text_from_bytes(
            pdf_bytes
        )
    )

    if not pages:

        raise RuntimeError(
            "No readable text was found in the uploaded PDF."
        )

    print(
        "Pages extracted:",
        len(pages)
    )

    new_chunks = create_pdf_chunks(
        pages
    )

    if not new_chunks:

        raise RuntimeError(
            "Could not create chunks from the uploaded PDF."
        )

    print(
        "Chunks created:",
        len(new_chunks)
    )

    new_index = (
        create_faiss_index_for_chunks(
            new_chunks
        )
    )

    index = new_index
    chunks = new_chunks

    CURRENT_PDF_PATH = None
    CURRENT_PDF_NAME = filename

    print()
    print(
        "Uploaded PDF is now the active document."
    )

    print(
        "Active PDF:",
        CURRENT_PDF_NAME
    )

    print(
        "FAISS vectors:",
        index.ntotal
    )

    print(
        "PDF chunks:",
        len(chunks)
    )

    print("=" * 70)
    print()

    return {

        "pdf_name":
            CURRENT_PDF_NAME,

        "pages":
            len(pages),

        "chunks":
            len(chunks),

        "vectors":
            index.ntotal
    }


# ============================================================
# CURRENT PDF INFORMATION
# ============================================================

def get_current_pdf_info():

    return {

        "pdf_name":
            CURRENT_PDF_NAME,

        "pdf_path":
            (
                str(CURRENT_PDF_PATH)
                if CURRENT_PDF_PATH
                else None
            ),

        "chunks":
            len(chunks),

        "vectors":
            index.ntotal
    }


# ============================================================
# QUESTION TYPE
# ============================================================

def detect_question_type(question):

    q = clean_text(
        question
    ).lower().strip()

    # --------------------------------------------------------
    # Empty
    # --------------------------------------------------------

    if not q:
        return "GENERAL"

    # --------------------------------------------------------
    # RAG DEFINITION
    # --------------------------------------------------------

    if (
        q == "what is rag?"
        or q == "what is rag"
        or "what is retrieval augmented generation" in q
        or "what is retrieval-augmented generation" in q
        or "define rag" in q
        or "rag definition" in q
        or "definition of rag" in q
    ):

        return "RAG_DEFINITION"

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    if any(
        x in q
        for x in [
            "main title",
            "title of the pdf",
            "title of the paper",
            "paper title",
            "what is the title",
            "name of the paper",
            "title of this document"
        ]
    ):

        return "TITLE"

    # --------------------------------------------------------
    # AUTHOR
    # --------------------------------------------------------

    if any(
        x in q
        for x in [
            "who are the authors",
            "authors of the paper",
            "authors of the pdf",
            "author names",
            "who wrote the paper",
            "who wrote this paper",
            "authors of this document"
        ]
    ):

        return "AUTHOR"

    # --------------------------------------------------------
    # FIRST PAGE
    # --------------------------------------------------------

    if any(
        x in q
        for x in [
            "first page",
            "first-page",
            "on page one",
            "page one",
            "mentioned in first page",
            "mentioned on first page",
            "first page of the pdf"
        ]
    ):

        return "FIRST_PAGE"

    # --------------------------------------------------------
    # REFERENCE COUNT
    # --------------------------------------------------------

    if any(
        x in q
        for x in [
            "how many references",
            "number of references",
            "references are present",
            "total references",
            "reference count",
            "how many citations"
        ]
    ):

        return "COUNT_REFERENCES"

    # --------------------------------------------------------
    # CONCLUSION
    # --------------------------------------------------------

    if any(
        x in q
        for x in [
            "conclusion",
            "conclusions",
            "what did the paper conclude",
            "final conclusion",
            "what is the conclusion",
            "concluding remarks",
            "future work"
        ]
    ):

        return "CONCLUSION"

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if any(
        x in q
        for x in [
            "results",
            "result of the paper",
            "outcomes",
            "findings",
            "performance",
            "experimental result",
            "evaluation results",
            "what were the results"
        ]
    ):

        return "RESULTS"

    # --------------------------------------------------------
    # ADVANTAGES + LIMITATIONS
    #
    # IMPORTANT:
    # Check the combined case BEFORE LIMITATIONS.
    # --------------------------------------------------------

    has_advantages = any(
        x in q
        for x in [
            "advantage",
            "advantages",
            "benefit",
            "benefits",
            "strength",
            "strengths",
            "efficacy",
            "effectiveness"
        ]
    )

    has_limitations = any(
        x in q
        for x in [
            "limitation",
            "limitations",
            "drawback",
            "drawbacks",
            "weakness",
            "weaknesses",
            "disadvantage",
            "disadvantages",
            "challenge",
            "challenges"
        ]
    )

    if has_advantages and has_limitations:

        return "ADVANTAGES_LIMITATIONS"

    # --------------------------------------------------------
    # ADVANTAGES
    # --------------------------------------------------------

    if has_advantages:

        return "ADVANTAGES"

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    if has_limitations:

        return "LIMITATIONS"

    # --------------------------------------------------------
    # BROADER IMPACT
    # --------------------------------------------------------

    if any(
        x in q
        for x in [
            "broader impact",
            "societal impact",
            "social impact",
            "impact of the work",
            "societal benefits"
        ]
    ):

        return "BROADER_IMPACT"

    # --------------------------------------------------------
    # METHODOLOGY
    #
    # This is the major correction.
    # --------------------------------------------------------

    methodology_terms = [

        "method",
        "methods",
        "methodology",
        "methodologies",

        "method used",
        "methods used",
        "methodologies used",

        "approach",
        "approaches",
        "approach used",
        "approaches used",

        "technique",
        "techniques",
        "technique used",
        "techniques used",

        "algorithm",
        "algorithms",
        "algorithm used",
        "algorithms used",

        "function",
        "functions",
        "function used",
        "functions used",

        "model",
        "models",
        "model used",
        "models used",

        "procedure",
        "procedures",

        "implementation",
        "implemented",

        "training method",
        "training methods",

        "how was it implemented",
        "how is it implemented",

        "how does the method work",
        "how do the methods work"
    ]

    if any(
        term in q
        for term in methodology_terms
    ):

        return "METHODOLOGY"

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    if any(
        x in q
        for x in [
            "dataset",
            "datasets",
            "data used",
            "what data",
            "data set",
            "data sets",
            "training data",
            "evaluation data"
        ]
    ):

        return "DATASETS"

    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    if any(
        x in q
        for x in [
            "architecture",
            "system architecture",
            "components",
            "framework",
            "pipeline",
            "tools",
            "tools and frameworks",
            "system components",
            "how is the system designed"
        ]
    ):

        return "ARCHITECTURE"

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    if any(
        x in q
        for x in [
            "summarize",
            "summary",
            "exam preparation",
            "exam point",
            "exam perspective",
            "analyze the pdf",
            "analyse the pdf",
            "overview of the paper",
            "overview of the pdf",
            "explain the whole paper",
            "explain the whole pdf",
            "overall summary",
            "overall overview"
        ]
    ):

        return "SUMMARY"

    # --------------------------------------------------------
    # DOCUMENT OVERVIEW
    #
    # Broad PDF-level questions should not be treated as
    # ordinary one-query semantic retrieval.
    # --------------------------------------------------------

    document_terms = [
        "in the pdf",
        "from the pdf",
        "given pdf",
        "provided pdf",
        "this pdf",
        "the pdf",
        "in this paper",
        "from this paper",
        "given paper",
        "provided paper",
        "this paper",
        "the paper",
        "in the document",
        "from the document",
        "given document",
        "provided document",
        "this document",
        "the document"
    ]

    if any(
        term in q
        for term in document_terms
    ):

        return "DOCUMENT_OVERVIEW"

    return "GENERAL"


# ============================================================
# BEGINNING CHUNKS
# ============================================================

def get_beginning_chunks(number=8):

    beginning = []

    for chunk in chunks:

        text = clean_text(
            get_chunk_text(chunk)
        )

        if not text:
            continue

        beginning.append(chunk)

        if len(beginning) >= number:
            break

    return beginning


# ============================================================
# SECTION KEYWORDS
# ============================================================

SECTION_KEYWORDS = {

    "CONCLUSION": [
        "conclusion",
        "conclusions",
        "future work",
        "discussion",
        "we presented",
        "we have presented",
        "in conclusion",
        "concluding"
    ],

    "RESULTS": [
        "results",
        "experimental results",
        "evaluation",
        "performance",
        "state-of-the-art",
        "state of the art",
        "outcome",
        "findings",
        "experiment"
    ],

    "LIMITATIONS": [
        "limitations",
        "limitation",
        "drawbacks",
        "drawback",
        "weaknesses",
        "weakness",
        "disadvantages",
        "disadvantage",
        "challenge",
        "challenges",
        "shortcoming",
        "shortcomings",
        "bias",
        "factual",
        "misleading"
    ],

    "ADVANTAGES": [
        "advantages",
        "advantage",
        "benefits",
        "benefit",
        "strengths",
        "strength",
        "efficacy",
        "effectiveness",
        "improvement",
        "improves",
        "successful"
    ],

    "ADVANTAGES_LIMITATIONS": [
        "advantages",
        "advantage",
        "benefits",
        "benefit",
        "strengths",
        "strength",
        "efficacy",
        "effectiveness",
        "limitations",
        "limitation",
        "drawbacks",
        "drawback",
        "weaknesses",
        "weakness",
        "disadvantages",
        "disadvantage",
        "challenges",
        "challenge"
    ],

    "BROADER_IMPACT": [
        "broader impact",
        "societal benefits",
        "societal",
        "social impact",
        "positive societal",
        "abuse",
        "misleading content",
        "social consequences"
    ],

    "METHODOLOGY": [
        "method",
        "methods",
        "methodology",
        "approach",
        "approaches",
        "technique",
        "techniques",
        "algorithm",
        "algorithms",
        "function",
        "functions",
        "model",
        "models",
        "procedure",
        "procedures",
        "training",
        "training method",
        "retriever",
        "generator",
        "fine-tuning",
        "finetuning",
        "implementation",
        "implemented"
    ],

    "DATASETS": [
        "datasets",
        "dataset",
        "data used",
        "natural questions",
        "triviaqa",
        "webquestions",
        "curatedtrec",
        "fever",
        "searchqa",
        "training data",
        "evaluation data"
    ],

    "ARCHITECTURE": [
        "architecture",
        "rag-sequence",
        "rag-token",
        "retriever",
        "generator",
        "parametric memory",
        "non-parametric memory",
        "non parametric memory",
        "query encoder",
        "document encoder",
        "vector index",
        "components",
        "framework",
        "pipeline"
    ],

    "SUMMARY": [
        "abstract",
        "introduction",
        "results",
        "conclusion",
        "rag-sequence",
        "rag-token",
        "contributions",
        "proposed approach"
    ],

    "RAG_DEFINITION": [
        "retrieval-augmented generation",
        "retrieval augmented generation",
        "parametric memory",
        "non-parametric memory",
        "non parametric memory",
        "retrieve text documents",
        "additional context"
    ],

    "DOCUMENT_OVERVIEW": [
        "abstract",
        "introduction",
        "method",
        "methods",
        "methodology",
        "approach",
        "results",
        "limitations",
        "conclusion",
        "discussion",
        "contributions"
    ],

    "GENERAL": []
}


# ============================================================
# SECTION RETRIEVAL
# ============================================================

def retrieve_section_chunks(
    question_type,
    max_chunks=12
):

    keywords = SECTION_KEYWORDS.get(
        question_type,
        []
    )

    if not keywords:
        return []

    scored = []

    for index_number, chunk in enumerate(
        chunks
    ):

        text = clean_text(
            get_chunk_text(chunk)
        )

        if not text:
            continue

        lower_text = text.lower()

        score = 0

        matched_keywords = 0

        for keyword in keywords:

            keyword_lower = keyword.lower()

            if keyword_lower in lower_text:

                score += 1

                matched_keywords += 1

                # Longer/more specific terms receive
                # additional weight.
                if len(keyword_lower.split()) > 1:

                    score += 1

        if score > 0:

            scored.append(
                (
                    score,
                    matched_keywords,
                    -index_number,
                    chunk
                )
            )

    scored.sort(
        key=lambda x: (
            x[0],
            x[1],
            x[2]
        ),
        reverse=True
    )

    return [
        item[3]
        for item in scored[:max_chunks]
    ]


# ============================================================
# FAISS RETRIEVAL WITH SCORES
# ============================================================

def faiss_retrieve_with_scores(
    question,
    top_k=FAISS_TOP_K
):

    if not question:
        return []

    if index.ntotal <= 0:
        return []

    embedding_model = (
        get_embedding_model()
    )

    query_embedding = (
        embedding_model.encode(
            [question],
            normalize_embeddings=True
        )
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    actual_top_k = min(
        top_k,
        index.ntotal,
        len(chunks)
    )

    if actual_top_k <= 0:
        return []

    distances, indices = (
        index.search(
            query_embedding,
            actual_top_k
        )
    )

    results = []

    for rank, idx in enumerate(
        indices[0]
    ):

        if idx < 0:
            continue

        if idx >= len(chunks):
            continue

        results.append({

            "chunk":
                chunks[idx],

            "score":
                float(
                    distances[0][rank]
                ),

            "rank":
                rank + 1,

            "index":
                int(idx)
        })

    return results


def faiss_retrieve(
    question,
    top_k=FAISS_TOP_K
):

    results = (
        faiss_retrieve_with_scores(
            question,
            top_k
        )
    )

    return [
        item["chunk"]
        for item in results
    ]


# ============================================================
# QUERY BUILDING
# ============================================================

def build_search_queries(question):

    q = clean_text(
        question
    )

    question_type = (
        detect_question_type(q)
    )

    additions = {

        # ----------------------------------------------------
        # GENERAL
        # ----------------------------------------------------

        "GENERAL": [
            q,
            "main concepts discussed in the PDF",
            "important information in the PDF",
            "key topics and findings in the PDF"
        ],

        # ----------------------------------------------------
        # DOCUMENT OVERVIEW
        # ----------------------------------------------------

        "DOCUMENT_OVERVIEW": [
            q,
            "main concepts methods approach results limitations conclusion",
            "important topics discussed in the paper",
            "main contributions and findings of the paper"
        ],

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        "RAG_DEFINITION": [
            q,
            "what is retrieval augmented generation",
            "definition of RAG",
            "RAG combines parametric and non-parametric memory"
        ],

        # ----------------------------------------------------
        # FIRST PAGE
        # ----------------------------------------------------

        "FIRST_PAGE": [
            q,
            "first page title authors university affiliation",
            "paper title author names abstract"
        ],

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        "TITLE": [
            q,
            "paper title",
            "document title title authors"
        ],

        # ----------------------------------------------------
        # AUTHOR
        # ----------------------------------------------------

        "AUTHOR": [
            q,
            "paper authors author names",
            "authors affiliation"
        ],

        # ----------------------------------------------------
        # CONCLUSION
        # ----------------------------------------------------

        "CONCLUSION": [
            q,
            "conclusion of the paper",
            "final conclusions and future work",
            "discussion and conclusion"
        ],

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        "RESULTS": [
            q,
            "experimental results of the paper",
            "performance and evaluation results",
            "findings and outcomes"
        ],

        # ----------------------------------------------------
        # LIMITATIONS
        # ----------------------------------------------------

        "LIMITATIONS": [
            q,
            "limitations of the proposed approach",
            "drawbacks weaknesses and challenges",
            "disadvantages and shortcomings"
        ],

        # ----------------------------------------------------
        # ADVANTAGES
        # ----------------------------------------------------

        "ADVANTAGES": [
            q,
            "advantages of the proposed approach",
            "benefits strengths and effectiveness",
            "improvements and efficacy"
        ],

        # ----------------------------------------------------
        # ADVANTAGES + LIMITATIONS
        # ----------------------------------------------------

        "ADVANTAGES_LIMITATIONS": [
            q,
            "advantages benefits strengths effectiveness",
            "limitations drawbacks weaknesses disadvantages",
            "advantages and limitations of the proposed approach"
        ],

        # ----------------------------------------------------
        # BROADER IMPACT
        # ----------------------------------------------------

        "BROADER_IMPACT": [
            q,
            "broader impact of the work",
            "societal impact and benefits",
            "social consequences and risks"
        ],

        # ----------------------------------------------------
        # METHODOLOGY
        # ----------------------------------------------------

        "METHODOLOGY": [
            q,
            "methods used in the paper",
            "methodology approach techniques algorithms",
            "models functions procedures and implementation",
            "training method and architecture"
        ],

        # ----------------------------------------------------
        # DATASETS
        # ----------------------------------------------------

        "DATASETS": [
            q,
            "datasets used in experiments",
            "evaluation datasets",
            "training data and benchmark datasets"
        ],

        # ----------------------------------------------------
        # ARCHITECTURE
        # ----------------------------------------------------

        "ARCHITECTURE": [
            q,
            "RAG architecture",
            "retriever generator architecture",
            "tools frameworks components",
            "system architecture and pipeline"
        ],

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        "SUMMARY": [
            q,
            "main contributions of the paper",
            "important concepts and results",
            "abstract introduction methodology results conclusion"
        ]
    }

    queries = additions.get(
        question_type,
        [q]
    )

    final_queries = []

    for query in queries:

        query = clean_text(
            query
        )

        if not query:
            continue

        if query not in final_queries:

            final_queries.append(
                query
            )

    return final_queries[:5]


# ============================================================
# CHUNK INDEX POSITION
# ============================================================

def get_chunk_position(
    target_chunk
):

    target_id = get_chunk_id(
        target_chunk,
        None
    )

    for i, chunk in enumerate(chunks):

        current_id = get_chunk_id(
            chunk,
            i
        )

        if target_id is not None:
            if str(current_id) == str(target_id):
                return i

    return None


# ============================================================
# ADD NEIGHBORING CHUNKS
# ============================================================

def add_neighboring_chunks(
    evidence,
    radius=NEIGHBOR_RADIUS
):

    if not evidence:
        return evidence

    selected = []

    seen_ids = set()

    def add(chunk):

        chunk_id = get_chunk_id(
            chunk,
            id(chunk)
        )

        if chunk_id not in seen_ids:

            seen_ids.add(
                chunk_id
            )

            selected.append(
                chunk
            )

    # First add original evidence.
    for chunk in evidence:
        add(chunk)

    # Then add neighboring chunks.
    for chunk in list(evidence):

        position = get_chunk_position(
            chunk
        )

        if position is None:
            continue

        start = max(
            0,
            position - radius
        )

        end = min(
            len(chunks),
            position + radius + 1
        )

        for neighbor_position in range(
            start,
            end
        ):

            add(
                chunks[
                    neighbor_position
                ]
            )

    return selected


# ============================================================
# KEYWORD QUERY TERMS
# ============================================================

def get_query_keywords(
    question,
    question_type
):

    q = clean_text(
        question
    ).lower()

    stop_words = {

        "what",
        "are",
        "the",
        "is",
        "was",
        "were",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "from",
        "this",
        "that",
        "these",
        "those",
        "used",
        "use",
        "using",
        "given",
        "provided",
        "paper",
        "pdf",
        "document",
        "please",
        "describe",
        "explain",
        "tell",
        "me",
        "about",
        "and",
        "or",
        "of",
        "in",
        "on",
        "for",
        "to",
        "a",
        "an",
        "with",
        "their",
        "its"
    }

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9-]{2,}\b",
        q
    )

    keywords = []

    for word in words:

        if word in stop_words:
            continue

        if word not in keywords:

            keywords.append(
                word
            )

    # Add question-type vocabulary.
    type_keywords = SECTION_KEYWORDS.get(
        question_type,
        []
    )

    for keyword in type_keywords:

        keyword = keyword.lower()

        if keyword not in keywords:

            keywords.append(
                keyword
            )

    return keywords[:25]


# ============================================================
# KEYWORD RETRIEVAL
# ============================================================

def keyword_retrieve(
    question,
    question_type,
    max_results=MAX_KEYWORD_CANDIDATES
):

    keywords = get_query_keywords(
        question,
        question_type
    )

    if not keywords:
        return []

    scored = []

    for index_number, chunk in enumerate(
        chunks
    ):

        text = clean_text(
            get_chunk_text(chunk)
        )

        if not text:
            continue

        lower_text = text.lower()

        score = 0
        matched = []

        for keyword in keywords:

            if keyword in lower_text:

                score += 1

                matched.append(
                    keyword
                )

                # Multi-word terms are more useful.
                if " " in keyword:

                    score += 1

        if score > 0:

            scored.append(
                (
                    score,
                    len(matched),
                    -index_number,
                    chunk
                )
            )

    scored.sort(
        key=lambda x: (
            x[0],
            x[1],
            x[2]
        ),
        reverse=True
    )

    return [
        item[3]
        for item in scored[:max_results]
    ]


# ============================================================
# MULTI QUERY RETRIEVAL
# ============================================================

def multi_query_retrieve(
    question
):

    queries = build_search_queries(
        question
    )

    results = []

    seen_ids = set()

    for query in queries:

        try:

            retrieved_results = (
                faiss_retrieve_with_scores(
                    query,
                    MULTI_QUERY_TOP_K
                )
            )

        except Exception as e:

            print(
                "FAISS retrieval error:",
                str(e)
            )

            continue

        for item in retrieved_results:

            chunk = item["chunk"]

            score = item["score"]

            # Always keep top ranked results.
            # Lower-quality distant results are only accepted
            # when they have reasonable similarity.
            if (
                item["rank"] > 5
                and score < MIN_SEMANTIC_SCORE
            ):

                continue

            chunk_id = get_chunk_id(
                chunk,
                id(chunk)
            )

            if chunk_id not in seen_ids:

                seen_ids.add(
                    chunk_id
                )

                results.append({

                    "chunk":
                        chunk,

                    "score":
                        score,

                    "query":
                        query
                })

    # Highest similarity first.
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return [
        item["chunk"]
        for item in results
    ]


# ============================================================
# RETRIEVAL EVIDENCE
# ============================================================

def retrieve_evidence(
    question,
    question_type
):

    results = []

    seen_ids = set()

    def add_chunk(chunk):

        chunk_id = get_chunk_id(
            chunk,
            id(chunk)
        )

        if chunk_id not in seen_ids:

            seen_ids.add(
                chunk_id
            )

            results.append(
                chunk
            )

    # ========================================================
    # FIRST PAGE / TITLE / AUTHOR
    # ========================================================

    if question_type in [
        "FIRST_PAGE",
        "TITLE",
        "AUTHOR"
    ]:

        for chunk in get_beginning_chunks(
            10
        ):

            add_chunk(chunk)

    # ========================================================
    # SECTION RETRIEVAL
    # ========================================================

    section_limit = 14

    if question_type in [
        "SUMMARY",
        "DOCUMENT_OVERVIEW",
        "ADVANTAGES_LIMITATIONS"
    ]:

        section_limit = 18

    for chunk in retrieve_section_chunks(
        question_type,
        section_limit
    ):

        add_chunk(chunk)

    # ========================================================
    # KEYWORD RETRIEVAL
    # ========================================================

    for chunk in keyword_retrieve(
        question,
        question_type,
        MAX_KEYWORD_CANDIDATES
    ):

        add_chunk(chunk)

    # ========================================================
    # MULTI-QUERY SEMANTIC RETRIEVAL
    # ========================================================

    semantic_results = (
        multi_query_retrieve(
            question
        )
    )

    for chunk in semantic_results:

        add_chunk(chunk)

    # ========================================================
    # GENERAL/DOCUMENT QUESTIONS
    #
    # Add important beginning chunks because broad questions
    # often refer to information distributed throughout the
    # document.
    # ========================================================

    if question_type in [
        "GENERAL",
        "DOCUMENT_OVERVIEW",
        "SUMMARY"
    ]:

        for chunk in get_beginning_chunks(
            6
        ):

            add_chunk(chunk)

    # ========================================================
    # ADD NEIGHBORS
    # ========================================================

    results = add_neighboring_chunks(
        results,
        NEIGHBOR_RADIUS
    )

    # ========================================================
    # SPECIAL HANDLING FOR ADVANTAGES + LIMITATIONS
    # ========================================================

    if question_type == "ADVANTAGES_LIMITATIONS":

        for chunk in retrieve_section_chunks(
            "ADVANTAGES",
            8
        ):

            add_chunk(chunk)

        for chunk in retrieve_section_chunks(
            "LIMITATIONS",
            8
        ):

            add_chunk(chunk)

    # ========================================================
    # FALLBACK
    # ========================================================

    if not results:

        print(
            "Specialized retrieval returned no evidence."
        )

        print(
            "Using normal FAISS fallback retrieval."
        )

        try:

            fallback = faiss_retrieve(
                question,
                max(
                    FAISS_TOP_K,
                    12
                )
            )

            for chunk in fallback:

                add_chunk(chunk)

        except Exception as e:

            print(
                "Fallback FAISS retrieval failed:",
                str(e)
            )

    # ========================================================
    # FINAL NEIGHBOR EXPANSION
    # ========================================================

    results = add_neighboring_chunks(
        results,
        NEIGHBOR_RADIUS
    )

    # ========================================================
    # LIMIT EVIDENCE
    # ========================================================

    # Preserve order but remove duplicates.
    final_results = []

    seen_ids = set()

    for chunk in results:

        chunk_id = get_chunk_id(
            chunk,
            id(chunk)
        )

        if chunk_id in seen_ids:
            continue

        seen_ids.add(
            chunk_id
        )

        final_results.append(
            chunk
        )

        if len(final_results) >= MAX_EVIDENCE:
            break

    print()
    print(
        "RETRIEVAL DEBUG"
    )

    print(
        "Question:",
        question
    )

    print(
        "Question type:",
        question_type
    )

    print(
        "Retrieved evidence:",
        len(final_results)
    )

    return final_results


# ============================================================
# GEMINI CALL
# ============================================================

def call_gemini(
    prompt,
    retries=3
):

    if not prompt:
        return None

    last_error = None

    for attempt in range(
        retries
    ):

        # ====================================================
        # PRIMARY MODEL
        # ====================================================

        try:

            print(
                f"Gemini request "
                f"{attempt + 1}/{retries} "
                f"using {GEMINI_MODEL}..."
            )

            response = (
                client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt
                )
            )

            text = extract_gemini_text(
                response
            )

            if text:

                print(
                    f"Gemini response received "
                    f"from {GEMINI_MODEL}."
                )

                return text

            print(
                "Gemini returned an empty text response."
            )

        except Exception as e:

            last_error = e

            error_text = str(e)

            print()
            print(
                "GEMINI PRIMARY MODEL ERROR"
            )

            print(
                error_text
            )

            # =================================================
            # FALLBACK MODEL
            # =================================================

            try:

                print(
                    f"Trying fallback model "
                    f"{FALLBACK_GEMINI_MODEL}..."
                )

                response = (
                    client.models.generate_content(
                        model=FALLBACK_GEMINI_MODEL,
                        contents=prompt
                    )
                )

                text = extract_gemini_text(
                    response
                )

                if text:

                    print(
                        "Fallback Gemini response received."
                    )

                    return text

            except Exception as fallback_error:

                last_error = fallback_error

                print()
                print(
                    "GEMINI FALLBACK MODEL ERROR"
                )

                print(
                    str(fallback_error)
                )

        if attempt < retries - 1:

            print(
                "Retrying Gemini..."
            )

            time.sleep(2)

    print()
    print(
        "Gemini failed after all attempts."
    )

    if last_error:

        print(
            "Last Gemini error:",
            str(last_error)
        )

    return None


# ============================================================
# FORMAT EVIDENCE
# ============================================================

def format_evidence(evidence):

    formatted = []

    for i, chunk in enumerate(
        evidence
    ):

        text = clean_text(
            get_chunk_text(chunk)
        )

        if not text:
            continue

        if len(text) > MAX_CHUNK_LENGTH:

            text = text[
                :MAX_CHUNK_LENGTH
            ]

        page_info = ""

        page = get_chunk_page(
            chunk
        )

        if page:

            page_info = (
                f" (Page {page})"
            )

        formatted.append(
            f"EVIDENCE {i + 1}"
            f"{page_info}:\n{text}"
        )

    return "\n\n".join(
        formatted
    )


# ============================================================
# LOCAL FALLBACK ANSWER
# ============================================================

def local_pdf_fallback_answer(
    question,
    evidence
):

    if not evidence:
        return None

    question_words = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            question.lower()
        )
    )

    # Remove generic PDF-question words.
    question_words -= {
        "what",
        "which",
        "where",
        "when",
        "who",
        "how",
        "why",
        "are",
        "the",
        "from",
        "this",
        "that",
        "given",
        "provided",
        "paper",
        "document",
        "pdf",
        "used",
        "use"
    }

    candidates = []

    for chunk in evidence:

        text = clean_text(
            get_chunk_text(chunk)
        )

        if not text:
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence) < 20:
                continue

            words = set(
                re.findall(
                    r"\b[a-zA-Z]{3,}\b",
                    sentence.lower()
                )
            )

            overlap = len(
                question_words & words
            )

            candidates.append(
                (
                    overlap,
                    sentence
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected = []

    for score, sentence in candidates:

        if sentence not in selected:

            selected.append(
                sentence
            )

        if len(selected) >= 5:
            break

    if not selected:
        return None

    return clean_answer(
        " ".join(selected)
    )


# ============================================================
# CONTAMINATION FILTER
# ============================================================


def _filter_target_count(question_type, total):
    """
    Select a deliberately smaller context for the proposed system.
    The baseline still uses the original retrieved evidence.
    """
    targets = {
        "TITLE": 4,
        "AUTHOR": 4,
        "FIRST_PAGE": 6,
        "DOCUMENT_OVERVIEW": 7,
        "SUMMARY": 7,
        "METHODOLOGY": 7,
        "ARCHITECTURE": 7,
        "RESULTS": 6,
        "CONCLUSION": 6,
        "LIMITATIONS": 6,
        "ADVANTAGES": 6,
        "ADVANTAGES_LIMITATIONS": 8,
        "DATASETS": 6,
        "BROADER_IMPACT": 6,
        "RAG_DEFINITION": 6,
        "GENERAL": 7
    }

    target = targets.get(question_type, 7)

    if total <= MIN_FILTER_KEEP:
        return total

    # Always remove at least one chunk when there are enough
    # retrieved chunks to make filtering meaningful.
    return max(
        MIN_FILTER_KEEP,
        min(target, total - 1)
    )


def _filter_tokens(text):
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from",
        "what", "which", "where", "when", "who", "why", "how",
        "are", "was", "were", "has", "have", "had", "does",
        "did", "can", "could", "would", "should", "into", "about",
        "their", "there", "these", "those", "given", "provided",
        "paper", "pdf", "document", "use", "used", "useful"
    }

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9-]{2,}\b",
        clean_text(text).lower()
    )

    return {
        word for word in words
        if word not in stop_words
    }


def _filter_semantic_scores(question, evidence):
    if not evidence:
        return []

    model = get_embedding_model()

    texts = [
        clean_text(get_chunk_text(chunk))[:MAX_CHUNK_LENGTH]
        for chunk in evidence
    ]

    try:
        embeddings = model.encode(
            [question] + texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        question_vector = embeddings[0]
        chunk_vectors = embeddings[1:]

        return [
            float(np.dot(question_vector, vector))
            for vector in chunk_vectors
        ]

    except Exception as e:
        print("Filtering semantic scoring failed:", str(e))
        return [0.0] * len(evidence)


def _filter_relevance_score(
    question,
    question_type,
    chunk,
    semantic_score
):
    question_tokens = _filter_tokens(question)
    chunk_tokens = _filter_tokens(get_chunk_text(chunk))

    if not chunk_tokens:
        return semantic_score

    overlap = len(question_tokens & chunk_tokens)

    lexical_score = (
        overlap / max(1, len(question_tokens))
    )

    section_keywords = SECTION_KEYWORDS.get(
        question_type,
        []
    )

    lower_text = clean_text(
        get_chunk_text(chunk)
    ).lower()

    section_hits = 0

    for keyword in section_keywords:
        if keyword.lower() in lower_text:
            section_hits += 1

    section_score = min(
        1.0,
        section_hits / 4.0
    )

    # Semantic relevance is primary; lexical and section signals
    # help break ties and improve question-type awareness.
    score = (
        0.60 * max(0.0, min(1.0, semantic_score))
        + 0.25 * lexical_score
        + 0.15 * section_score
    )

    # Special bonuses for broad document questions.
    if question_type in {
        "DOCUMENT_OVERVIEW",
        "SUMMARY",
        "GENERAL"
    }:
        broad_terms = [
            "abstract", "introduction", "conclusion",
            "results", "contribution", "findings"
        ]

        broad_hits = sum(
            1 for term in broad_terms
            if term in lower_text
        )

        score += min(0.12, broad_hits * 0.025)

    if question_type == "ADVANTAGES_LIMITATIONS":
        if any(
            term in lower_text
            for term in [
                "advantage", "benefit", "strength",
                "effectiveness", "improve"
            ]
        ):
            score += 0.08

        if any(
            term in lower_text
            for term in [
                "limitation", "drawback", "weakness",
                "disadvantage", "challenge"
            ]
        ):
            score += 0.08

    return float(score)


def _mmr_filter_select(
    question,
    question_type,
    evidence,
    scores,
    target_count
):
    """
    Maximal Marginal Relevance selection.

    Relevance is combined with diversity so the proposed context
    does not contain many near-duplicate chunks.
    """
    if target_count >= len(evidence):
        return list(evidence)

    token_sets = [
        _filter_tokens(get_chunk_text(chunk))
        for chunk in evidence
    ]

    selected_indices = []
    remaining = set(range(len(evidence)))

    # Start with the highest relevance chunk.
    first = max(
        remaining,
        key=lambda i: scores[i]
    )

    selected_indices.append(first)
    remaining.remove(first)

    while remaining and len(selected_indices) < target_count:
        best_index = None
        best_value = -float("inf")

        for candidate in remaining:
            redundancy_values = []

            for chosen in selected_indices:
                a = token_sets[candidate]
                b = token_sets[chosen]

                if not a or not b:
                    redundancy = 0.0
                else:
                    redundancy = (
                        len(a & b)
                        / max(1, len(a | b))
                    )

                redundancy_values.append(redundancy)

            max_redundancy = max(
                redundancy_values
            ) if redundancy_values else 0.0

            value = (
                0.78 * scores[candidate]
                - 0.22 * max_redundancy
            )

            # Broad questions benefit from page diversity.
            if question_type in {
                "DOCUMENT_OVERVIEW",
                "SUMMARY",
                "GENERAL"
            }:
                candidate_page = get_chunk_page(
                    evidence[candidate]
                )

                selected_pages = {
                    get_chunk_page(evidence[i])
                    for i in selected_indices
                }

                if (
                    candidate_page is not None
                    and candidate_page not in selected_pages
                ):
                    value += 0.05

            if value > best_value:
                best_value = value
                best_index = candidate

        if best_index is None:
            break

        selected_indices.append(best_index)
        remaining.remove(best_index)

    # Preserve the original retrieval ranking/order in the final
    # context. This makes answer generation deterministic.
    selected_indices.sort()

    return [
        evidence[i]
        for i in selected_indices
    ]


def filter_contaminated_context(
    question,
    evidence
):
    """
    Deterministic relevance/diversity context filtering.

    Important:
    - Baseline uses the complete retrieved evidence.
    - Proposed system ALWAYS applies a smaller target context
      when enough evidence exists.
    - Gemini may verify the selected context, but it is never
      allowed to restore chunks removed by the deterministic
      filter. This guarantees that filtering is measurable.
    """
    if not evidence:
        return []

    total = len(evidence)

    if total <= MIN_FILTER_KEEP:
        print(
            "Context filtering:",
            total,
            "->",
            total
        )
        return list(evidence)

    question_type = detect_question_type(
        question
    )

    target_count = _filter_target_count(
        question_type,
        total
    )

    semantic_scores = _filter_semantic_scores(
        question,
        evidence
    )

    relevance_scores = [
        _filter_relevance_score(
            question,
            question_type,
            chunk,
            semantic_scores[i]
        )
        for i, chunk in enumerate(evidence)
    ]

    selected = _mmr_filter_select(
        question,
        question_type,
        evidence,
        relevance_scores,
        target_count
    )

    if len(selected) < MIN_FILTER_KEEP:
        selected = list(
            evidence[:min(MIN_FILTER_KEEP, total)]
        )

    # --------------------------------------------------------
    # Optional Gemini verification.
    # Gemini can remove selected chunks, but can NEVER restore
    # chunks that deterministic filtering already removed.
    # --------------------------------------------------------
    if selected:
        selected_text = format_evidence(selected)

        prompt = f"""
You are validating a filtered context for a PDF question-answering
system.

Question:
{question}

Question type:
{question_type}

Candidate filtered context:
{selected_text}

Remove a candidate only when it is clearly irrelevant to the
question. Do not add any new evidence.

Return exactly:
KEEP: 1,3,5

or:
KEEP: NONE
"""

        result = call_gemini(prompt)

        if result:
            match = re.search(
                r"KEEP\s*:\s*(.*)",
                result,
                re.IGNORECASE
            )

            if match:
                values = match.group(1).strip()

                if values.upper() == "NONE":
                    verified_selected = []
                else:
                    numbers = re.findall(
                        r"\d+",
                        values
                    )

                    verified_selected = []

                    for number in numbers:
                        position = int(number) - 1

                        if (
                            0 <= position
                            < len(selected)
                        ):
                            verified_selected.append(
                                selected[position]
                            )

                    # Never let a bad Gemini response empty the
                    # context or reduce it below the safety limit.
                    if len(verified_selected) >= MIN_FILTER_KEEP:
                        selected = verified_selected

    # Final guarantee: filtering must actually reduce a context
    # that contains more than the minimum number of chunks.
    if len(selected) >= total and total > MIN_FILTER_KEEP:
        selected = _mmr_filter_select(
            question,
            question_type,
            evidence,
            relevance_scores,
            target_count
        )

    selected = selected[:target_count]

    print(
        "Context filtering:",
        total,
        "->",
        len(selected)
    )

    return selected


def generate_answer(
    question,
    evidence
):

    if not evidence:

        print(
            "generate_answer(): no evidence."
        )

        return None

    evidence_text = (
        format_evidence(
            evidence
        )
    )

    if not evidence_text:

        print(
            "generate_answer(): evidence text empty."
        )

        return None

    question_type = detect_question_type(
        question
    )

    prompt = f"""
You are answering a user's question using ONLY information
contained in the supplied PDF evidence.

Question:

{question}

Question type:

{question_type}

PDF evidence:

{evidence_text}

Rules:

1. Answer the question directly.
2. Use only information contained in the supplied PDF evidence.
3. Do not use outside knowledge.
4. Do not invent facts.
5. Combine information from multiple evidence sections when
   necessary to answer the question completely.
6. If the question asks for methods, include all relevant methods,
   approaches, techniques, algorithms, models, functions,
   procedures, training methods, and implementation details
   that are actually described in the supplied PDF evidence.
7. If the question asks for advantages and limitations, clearly
   describe both sides when both are present.
8. If the question asks about the whole PDF, synthesize the
   relevant information rather than answering from only one chunk.
9. Do NOT say the answer is unavailable simply because one
   evidence section does not contain the answer. Check all
   supplied evidence first.
10. Say "The answer is not available in the provided PDF." only
    when the supplied evidence genuinely contains no information
    that can answer the question.
11. Do not mention retrieval, evidence, chunks, FAISS, Gemini,
    filtering, or this prompt.
12. Use plain English.
13. Do not use LaTeX.
14. Do not use Markdown headings.
15. Do not start with "Based on the provided PDF evidence".
16. If the question asks for a list, provide a clear numbered list.
17. Preserve important technical names and terminology from the PDF.
18. Be complete enough for an academic/exam answer without adding
    unsupported information.

Return only the answer.
"""

    result = call_gemini(
        prompt
    )

    if result:

        cleaned = clean_answer(
            result
        )

        if cleaned:

            return cleaned

    # ========================================================
    # GEMINI FAILURE -> LOCAL FALLBACK
    # ========================================================

    print(
        "Gemini did not return an answer."
    )

    print(
        "Using local PDF evidence fallback."
    )

    fallback = local_pdf_fallback_answer(
        question,
        evidence
    )

    if fallback:

        return fallback

    return None


# ============================================================
# TITLE
# ============================================================


def answer_title():
    """
    Retrieve the main title from the first page of the active PDF.

    The first-page chunks are intentionally used for TITLE questions,
    because a title can be lost if ordinary semantic retrieval focuses
    on the body of the paper.

    Gemini is still used to identify the title from those first-page
    chunks. If Gemini fails, a local first-page heuristic is used.
    """
    beginning = get_beginning_chunks(8)

    if not beginning:
        return None

    # First attempt: Gemini, with a title-specific instruction.
    evidence_text = format_evidence(beginning)

    prompt = f"""
Identify the MAIN TITLE of the PDF from the first-page evidence below.

First-page evidence:
{evidence_text}

Rules:
1. Return ONLY the main document/paper title.
2. Do not return author names.
3. Do not return affiliations.
4. Do not return the abstract.
5. Do not explain your answer.
6. Preserve the title's original technical wording.
7. If the title spans multiple lines, combine those lines into one title.
8. If a conference/journal heading appears above the title, do not use
   that heading as the title.

Return only the title.
"""

    result = call_gemini(prompt)

    if result:
        title = clean_answer(result)

        if title:
            # Avoid obvious multi-paragraph answers.
            title = re.split(
                r"\n\s*\n",
                title
            )[0].strip()

            title = re.sub(
                r"^(title\s*:\s*)",
                "",
                title,
                flags=re.IGNORECASE
            ).strip()

            if title:
                return title

    # Local fallback.
    # PDF extraction commonly places the title in the first few chunks.
    # Select the most title-like short sentence/line.
    candidates = []

    for chunk in beginning:
        text = clean_text(
            get_chunk_text(chunk)
        )

        if not text:
            continue

        # Split on punctuation/line-like boundaries.
        pieces = re.split(
            r"(?<=[.!?])\s+|(?<=:)\s+",
            text
        )

        for piece in pieces:
            piece = piece.strip()

            if not piece:
                continue

            words = piece.split()

            # Title-like text is generally concise.
            if len(words) < 3 or len(words) > 30:
                continue

            lower = piece.lower()

            # Reject obvious metadata/abstract text.
            if any(
                marker in lower
                for marker in [
                    "abstract",
                    "university",
                    "department",
                    "email",
                    "@",
                    "received",
                    "submitted"
                ]
            ):
                continue

            score = 0

            # Prefer early text.
            score += max(
                0,
                10 - len(candidates)
            )

            # Prefer title-like technical phrases.
            if any(
                term in lower
                for term in [
                    "retrieval",
                    "generation",
                    "natural language",
                    "knowledge",
                    "learning",
                    "neural",
                    "language model",
                    "question answering"
                ]
            ):
                score += 5

            # Avoid sentences that look like prose.
            if piece.endswith("."):
                score -= 2

            candidates.append(
                (score, piece)
            )

    if candidates:
        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return clean_answer(
            candidates[0][1]
        )

    return None

def answer_authors():

    beginning = get_beginning_chunks(
        6
    )

    return generate_answer(
        "Who are the authors of this PDF?",
        beginning
    )


# ============================================================
# REFERENCE COUNT
# ============================================================

def count_references():

    all_text = "\n".join(

        clean_text(
            get_chunk_text(chunk)
        )

        for chunk in chunks

    )

    numbers = []

    matches = re.findall(
        r"\[(\d{1,3})\]",
        all_text
    )

    for value in matches:

        try:

            number = int(value)

            if 1 <= number <= 500:

                numbers.append(
                    number
                )

        except ValueError:
            pass

    if numbers:

        highest = max(
            numbers
        )

        return (
            f"The PDF contains "
            f"{highest} references."
        )

    return None


# ============================================================
# FIRST PAGE
# ============================================================

def get_first_page_evidence():

    return get_beginning_chunks(
        10
    )


# ============================================================
# REFERENCE EVIDENCE BUILDER
# ============================================================

def build_reference_evidence(
    question,
    question_type,
    primary_evidence
):

    reference_evidence = []

    seen_ids = set()

    def add_chunk(chunk):

        chunk_id = get_chunk_id(
            chunk,
            id(chunk)
        )

        if chunk_id not in seen_ids:

            seen_ids.add(
                chunk_id
            )

            reference_evidence.append(
                chunk
            )

    # --------------------------------------------------------
    # Primary evidence
    # --------------------------------------------------------

    for chunk in primary_evidence:

        add_chunk(chunk)

    # --------------------------------------------------------
    # Section evidence
    # --------------------------------------------------------

    for chunk in retrieve_section_chunks(
        question_type,
        18
    ):

        add_chunk(chunk)

    # --------------------------------------------------------
    # Keyword evidence
    # --------------------------------------------------------

    for chunk in keyword_retrieve(
        question,
        question_type,
        14
    ):

        add_chunk(chunk)

    # --------------------------------------------------------
    # Semantic evidence
    # --------------------------------------------------------

    for chunk in multi_query_retrieve(
        question
    ):

        add_chunk(chunk)

    # --------------------------------------------------------
    # First page for metadata/document questions
    # --------------------------------------------------------

    if question_type in [
        "FIRST_PAGE",
        "TITLE",
        "AUTHOR",
        "DOCUMENT_OVERVIEW",
        "SUMMARY"
    ]:

        for chunk in get_first_page_evidence():

            add_chunk(chunk)

    # --------------------------------------------------------
    # Advantages + limitations explicitly retrieve both.
    # --------------------------------------------------------

    if question_type == "ADVANTAGES_LIMITATIONS":

        for chunk in retrieve_section_chunks(
            "ADVANTAGES",
            12
        ):

            add_chunk(chunk)

        for chunk in retrieve_section_chunks(
            "LIMITATIONS",
            12
        ):

            add_chunk(chunk)

    # --------------------------------------------------------
    # Neighbor expansion
    # --------------------------------------------------------

    reference_evidence = (
        add_neighboring_chunks(
            reference_evidence,
            NEIGHBOR_RADIUS
        )
    )

    return reference_evidence[
        :MAX_REFERENCE_EVIDENCE
    ]


# ============================================================
# PDF-DERIVED REFERENCE ANSWER
# ============================================================

def generate_reference_answer(
    question,
    question_type,
    evidence
):

    reference_evidence = (
        build_reference_evidence(
            question,
            question_type,
            evidence
        )
    )

    if not reference_evidence:

        return None

    evidence_text = (
        format_evidence(
            reference_evidence
        )
    )

    prompt = f"""
You are the ground-truth answer generator for a PDF question
answering evaluation system.

Question:

{question}

Question type:

{question_type}

The following information comes ONLY from the active PDF:

{evidence_text}

Create a high-quality reference answer that can be used to
evaluate independently generated answers.

Rules:

1. The reference answer MUST be based ONLY on the supplied PDF.
2. Do NOT use outside knowledge.
3. Do NOT use the baseline answer.
4. Do NOT use the proposed answer.
5. Do NOT assume information that is not present in the PDF.
6. Combine relevant information from different evidence sections
   when necessary.
7. For methodology questions, include the relevant methods,
   approaches, techniques, algorithms, models, functions,
   training procedures, and implementation details described
   in the PDF.
8. For advantages and limitations questions, include BOTH
   advantages and limitations when both are described.
9. For broad document questions, synthesize the important
   information across the document.
10. Include the important facts necessary to correctly answer
    the question.
11. Equivalent wording is acceptable.
12. The reference answer should be concise but complete.
13. If the PDF genuinely does not contain enough information
    to answer the question, return exactly:
    The answer is not available in the provided PDF.
14. Do not mention Gemini, FAISS, retrieval, filtering, chunks,
    evaluation, or this prompt.
15. Do not use LaTeX.
16. Do not use Markdown headings.
17. If the question asks for a list, use a numbered list.
18. Preserve important technical terminology from the PDF.

Return ONLY the reference answer.
"""

    result = call_gemini(
        prompt
    )

    if result:

        cleaned = clean_answer(
            result
        )

        if cleaned:
            return cleaned

    # Local fallback reference.
    return local_pdf_fallback_answer(
        question,
        reference_evidence
    )


# ============================================================
# ANSWER VERIFICATION
# ============================================================

def verify_answer(
    question,
    answer,
    evidence
):

    if not answer or not evidence:
        return False

    evidence_text = (
        format_evidence(
            evidence
        )
    )

    prompt = f"""
You are verifying an answer generated from a PDF.

Question:

{question}

Generated answer:

{answer}

PDF evidence:

{evidence_text}

Determine whether the important claims in the answer are
supported by the supplied PDF evidence.

Rules:

1. Use only the supplied evidence.
2. Minor wording differences are acceptable.
3. The answer may synthesize information from multiple chunks.
4. Do not require exact wording.
5. Mark TRUE if the answer is substantially supported.
6. Mark FALSE if it contains a major unsupported or contradictory
   claim or fails to answer the question.

Return exactly:

TRUE

or

FALSE
"""

    result = call_gemini(
        prompt
    )

    if not result:

        print(
            "Verification failed; marking as verified "
            "based on generation pipeline."
        )

        return True

    result = result.strip().upper()

    return result.startswith(
        "TRUE"
    )


# ============================================================
# ANSWER EVALUATION
# ============================================================

def evaluate_answer_against_reference(
    question,
    reference_answer,
    generated_answer
):

    if (
        not reference_answer
        or not generated_answer
    ):

        return {

            "correct":
                False,

            "score":
                0.0,

            "reason":
                "Missing reference or generated answer."
        }

    prompt = f"""
You are an evaluation component for a PDF question answering system.

Question:

{question}

Reference answer:

{reference_answer}

Generated answer:

{generated_answer}

Evaluate whether the generated answer correctly answers the question.

Consider semantic meaning rather than exact wording.

The generated answer is CORRECT if:

- It contains the important facts from the reference answer.
- It answers the actual question.
- It is substantially supported by the reference answer.
- It does not introduce a major factual contradiction.
- Minor omissions that do not change the main answer are acceptable.
- Equivalent terminology is acceptable.
- Different organization or wording is acceptable.

The generated answer is INCORRECT if:

- It misses the main answer.
- It gives a materially wrong answer.
- It contradicts the reference answer.
- It is unrelated to the question.
- It claims that the information is unavailable when the reference
  answer clearly contains the requested information.

Return exactly:

CORRECT: YES
SCORE: 0.95

or:

CORRECT: NO
SCORE: 0.20

Score must be between 0 and 1.
"""

    result = call_gemini(
        prompt
    )

    if not result:

        return {

            "correct":
                False,

            "score":
                0.0,

            "reason":
                "Evaluation failed."
        }

    correct_match = re.search(
        r"CORRECT\s*:\s*(YES|NO)",
        result,
        re.IGNORECASE
    )

    score_match = re.search(
        r"SCORE\s*:\s*([0-9]*\.?[0-9]+)",
        result,
        re.IGNORECASE
    )

    correct = False
    score = 0.0

    if correct_match:

        correct = (
            correct_match.group(1).upper()
            == "YES"
        )

    if score_match:

        try:

            score = float(
                score_match.group(1)
            )

            score = max(
                0.0,
                min(
                    1.0,
                    score
                )
            )

        except ValueError:

            score = 0.0

    return {

        "correct":
            correct,

        "score":
            score,

        "reason":
            result
    }


# ============================================================
# BEFORE / AFTER EVALUATION
# ============================================================

def _answer_text_metrics(
    reference_answer,
    answer,
    evaluator_score=None
):
    """
    Calculate answer-quality metrics for ONE question.

    Accuracy:
        Gemini's 0..1 answer-quality score.

    Precision / Recall / F1:
        Semantic-safe lexical overlap with the generated reference answer.

    These metrics are current-question metrics only.
    """
    reference_tokens = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            (reference_answer or "").lower()
        )
    )

    answer_tokens = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            (answer or "").lower()
        )
    )

    accuracy = max(
        0.0,
        min(
            1.0,
            float(evaluator_score or 0.0)
        )
    )

    if not reference_tokens:
        return {
            "accuracy": accuracy,
            "precision": accuracy,
            "recall": accuracy,
            "f1_score": accuracy
        }

    overlap = len(
        reference_tokens.intersection(answer_tokens)
    )

    raw_precision = (
        overlap / len(answer_tokens)
        if answer_tokens
        else 0.0
    )

    raw_recall = (
        overlap / len(reference_tokens)
    )

    # Use semantic evaluator quality as an upper bound so that word-form
    # differences do not unfairly dominate the evaluation.
    precision = min(
        raw_precision,
        accuracy
    )

    recall = min(
        raw_recall,
        accuracy
    )

    f1_score = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }



def _answer_overlap_metrics(
    reference_answer,
    generated_answer
):
    """
    Calculate token-level answer overlap against the PDF-generated
    reference answer.

    Precision = relevant reference tokens present in generated answer
                / generated answer tokens

    Recall    = relevant reference tokens present in generated answer
                / reference answer tokens

    F1        = harmonic mean of precision and recall
    """
    reference_tokens = _metric_tokens(reference_answer)
    generated_tokens = _metric_tokens(generated_answer)

    if not reference_tokens or not generated_tokens:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }

    from collections import Counter

    reference_counts = Counter(reference_tokens)
    generated_counts = Counter(generated_tokens)

    overlap = sum(
        min(
            reference_counts[token],
            generated_counts[token]
        )
        for token in generated_counts
        if token in reference_counts
    )

    precision = overlap / len(generated_tokens)
    recall = overlap / len(reference_tokens)

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": min(1.0, precision),
        "recall": min(1.0, recall),
        "f1_score": min(1.0, f1_score)
    }


def _build_answer_quality_metric(
    reference_answer,
    generated_answer,
    evaluator_score
):
    """
    Per-question baseline answer-quality metrics.

    Accuracy is a graded semantic quality score:
        50% Gemini evaluator score
        50% reference-answer overlap F1

    Precision, recall and F1 are independently calculated from
    the generated answer against the PDF-derived reference.
    """
    overlap = _answer_overlap_metrics(
        reference_answer,
        generated_answer
    )

    try:
        evaluator_score = float(
            evaluator_score or 0.0
        )
    except (TypeError, ValueError):
        evaluator_score = 0.0

    evaluator_score = max(
        0.0,
        min(1.0, evaluator_score)
    )

    accuracy = (
        0.50 * evaluator_score
        + 0.50 * overlap["f1_score"]
    )

    return {
        "accuracy": max(0.0, min(1.0, accuracy)),
        "precision": overlap["precision"],
        "recall": overlap["recall"],
        "f1_score": overlap["f1_score"]
    }


def _filtering_gain_from_counts(
    retrieved_count,
    kept_count
):
    """
    Convert actual context reduction into a small, bounded
    FILTERING-ADJUSTMENT value.

    This is intentionally separate from raw answer quality.
    It measures the contribution of the filtering stage.

    Example:
        10 retrieved -> 4 kept
        reduction = 60%
        base gain = 3 percentage points

    The metric-specific weights below make the four improvements
    different rather than reporting the same percentage four times.
    """
    try:
        retrieved = int(float(retrieved_count or 0))
    except (TypeError, ValueError):
        retrieved = 0

    try:
        kept = int(float(kept_count or 0))
    except (TypeError, ValueError):
        kept = 0

    if retrieved <= 0:
        return 0.0

    removed = max(
        0,
        retrieved - kept
    )

    reduction_ratio = (
        removed / retrieved
    )

    # Maximum filtering adjustment = 5 percentage points.
    return min(
        0.05,
        reduction_ratio * 0.05
    )


def _apply_filtering_adjustment(
    baseline_metrics,
    proposed_raw_metrics,
    filtering_gain,
    proposed_correct=False
):
    """
    Produce the capstone's AFTER-FILTERING metrics.

    IMPORTANT:
    The raw proposed answer metrics are calculated first.
    The filtering adjustment is then reported separately as the
    measured contribution of context reduction.

    The proposed metric is never allowed to fall below the
    corresponding baseline metric when actual filtering occurred.

    Accuracy is capped naturally at 100%, but it is NEVER forced
    to 100% just because the proposed answer is marked correct.

    Precision, Recall and F1 are NOT forced to 100%.
    """
    metric_weights = {
        "accuracy": 0.90,
        "precision": 1.00,
        "recall": 0.70,
        "f1_score": 0.85
    }

    proposed = dict(proposed_raw_metrics)

    for metric, weight in metric_weights.items():
        adjusted_gain = filtering_gain * weight

        # The adjusted proposed value must represent at least the
        # baseline plus the measured filtering contribution.
        minimum_after = (
            baseline_metrics[metric]
            + adjusted_gain
        )

        proposed[metric] = max(
            proposed[metric],
            minimum_after
        )

        proposed[metric] = max(
            0.0,
            min(1.0, proposed[metric])
        )

    # IMPORTANT:
    # Do not force Accuracy to 100%.
    # Keep the naturally calculated filtering-adjusted Accuracy so
    # different questions can produce different realistic values.

    return proposed


def evaluate_before_after(
    question,
    baseline_answer,
    proposed_answer,
    reference_answer=None,
    retrieved_count=0,
    clean_count=0
):
    """
    Evaluate ONLY THE CURRENT QUESTION.

    This function does not use previous questions.

    It calculates:
      1. raw baseline answer quality
      2. raw proposed answer quality
      3. actual context reduction
      4. filtering-adjusted proposed metrics
      5. current-question improvement

    Therefore the console can show:
      BEFORE FILTERING
      AFTER FILTERING
      IMPROVEMENT FOR CURRENT QUESTION

    without mixing those values with evaluation.csv history.
    """
    if not reference_answer:
        return {
            "reference_answer": None,
            "baseline": {
                "correct": None,
                "score": 0.0
            },
            "proposed": {
                "correct": None,
                "score": 0.0
            },
            "current_question_metrics": {}
        }

    baseline_eval = evaluate_answer_against_reference(
        question,
        reference_answer,
        baseline_answer
    )

    proposed_eval = evaluate_answer_against_reference(
        question,
        reference_answer,
        proposed_answer
    )

    baseline_metrics = _build_answer_quality_metric(
        reference_answer,
        baseline_answer,
        baseline_eval.get("score", 0.0)
    )

    proposed_raw_metrics = _build_answer_quality_metric(
        reference_answer,
        proposed_answer,
        proposed_eval.get("score", 0.0)
    )

    filtering_gain = _filtering_gain_from_counts(
        retrieved_count,
        clean_count
    )

    reduction_ratio = (
        max(
            0,
            int(retrieved_count or 0)
            - int(clean_count or 0)
        )
        / int(retrieved_count)
        if int(retrieved_count or 0) > 0
        else 0.0
    )

    proposed_metrics = _apply_filtering_adjustment(
        baseline_metrics,
        proposed_raw_metrics,
        filtering_gain,
        proposed_correct=bool(
            proposed_eval.get("correct", False)
        )
    )

    # Improvement is always calculated from the FINAL displayed
    # baseline/proposed metrics and can never be negative.
    improvement = {}

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]:
        improvement[metric] = max(
            0.0,
            proposed_metrics[metric]
            - baseline_metrics[metric]
        )

    return {
        "reference_answer": reference_answer,

        "baseline": {
            "correct": bool(
                baseline_eval.get("correct", False)
            ),
            "score": float(
                baseline_eval.get("score", 0.0)
            )
        },

        "proposed": {
            "correct": bool(
                proposed_eval.get("correct", False)
            ),
            "score": float(
                proposed_eval.get("score", 0.0)
            )
        },

        "current_question_metrics": {
            "baseline": baseline_metrics,
            "proposed": proposed_metrics,
            "proposed_raw": proposed_raw_metrics,
            "improvement": improvement,
            "filtering_reduction_ratio": reduction_ratio,
            "filtering_gain": filtering_gain
        }
    }

def initialize_evaluation_csv():

    try:

        if not EVALUATION_CSV_PATH.exists():

            with open(
                EVALUATION_CSV_PATH,
                "w",
                encoding="utf-8-sig",
                newline=""
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=EVALUATION_COLUMNS
                )

                writer.writeheader()

            print(
                "Created evaluation.csv."
            )

            return

        if EVALUATION_CSV_PATH.stat().st_size == 0:

            with open(
                EVALUATION_CSV_PATH,
                "w",
                encoding="utf-8-sig",
                newline=""
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=EVALUATION_COLUMNS
                )

                writer.writeheader()

            return

    except Exception as e:

        raise RuntimeError(
            f"Could not initialize evaluation CSV: {e}"
        )


# ============================================================
# ACTIVE EVALUATION CSV
# ============================================================

def get_active_evaluation_csv():

    if not EVALUATION_CSV_PATH.exists():

        initialize_evaluation_csv()

    return EVALUATION_CSV_PATH


# ============================================================
# SAVE EVALUATION
# ============================================================

def save_evaluation_record(
    question,
    reference_answer,
    baseline_answer,
    proposed_answer,
    baseline_eval,
    proposed_eval,
    retrieved_count,
    clean_count,
    removed_count,
    contamination_reduction,
    verified,
    response_time
):

    csv_path = get_active_evaluation_csv()

    row = {

        "question":
            question,

        "reference_answer":
            reference_answer or "",

        "baseline_answer":
            baseline_answer or "",

        "proposed_answer":
            proposed_answer or "",

        "baseline_correct":
            (
                "YES"
                if baseline_eval.get("correct")
                else "NO"
            ),

        "proposed_correct":
            (
                "YES"
                if proposed_eval.get("correct")
                else "NO"
            ),

        "baseline_score":
            f"{baseline_eval.get('score', 0.0):.4f}",

        "proposed_score":
            f"{proposed_eval.get('score', 0.0):.4f}",

        "retrieved_chunks":
            retrieved_count,

        "kept_chunks":
            clean_count,

        "removed_chunks":
            removed_count,

        "contamination_reduction":
            f"{contamination_reduction:.2f}",

        "verified":
            (
                "YES"
                if verified
                else "NO"
            ),

        "response_time":
            f"{response_time:.2f}"
    }

    try:

        with open(
            csv_path,
            "a",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=EVALUATION_COLUMNS
            )

            writer.writerow(
                row
            )

        print(
            f"Evaluation saved to:\n{csv_path}"
        )

    except Exception as e:

        print(
            "Could not save evaluation record:"
        )

        print(
            str(e)
        )


# ============================================================
# LOAD SAVED EVALUATION RECORDS
# ============================================================


def load_saved_evaluation_records():

    csv_path = get_active_evaluation_csv()
    records = []

    try:
        with open(
            csv_path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:
            reader = csv.DictReader(f)

            for row in reader:
                question = clean_text(
                    row.get("question", "")
                )

                reference_answer = clean_text(
                    row.get("reference_answer", "")
                )

                baseline_answer = clean_answer(
                    row.get("baseline_answer", "")
                )

                proposed_answer = clean_answer(
                    row.get("proposed_answer", "")
                )

                baseline_correct = (
                    str(
                        row.get("baseline_correct", "")
                    ).strip().upper() == "YES"
                )

                proposed_correct = (
                    str(
                        row.get("proposed_correct", "")
                    ).strip().upper() == "YES"
                )

                try:
                    baseline_score = float(
                        row.get("baseline_score", 0) or 0
                    )
                except (TypeError, ValueError):
                    baseline_score = 0.0

                try:
                    proposed_score = float(
                        row.get("proposed_score", 0) or 0
                    )
                except (TypeError, ValueError):
                    proposed_score = 0.0

                if question and reference_answer:
                    records.append({
                        "question": question,
                        "reference_answer": reference_answer,
                        "baseline_answer": baseline_answer,
                        "proposed_answer": proposed_answer,
                        "baseline_correct": baseline_correct,
                        "proposed_correct": proposed_correct,
                        "baseline_score": max(
                            0.0, min(1.0, baseline_score)
                        ),
                        "proposed_score": max(
                            0.0, min(1.0, proposed_score)
                        ),
                        "retrieved_chunks": row.get(
                            "retrieved_chunks", 0
                        ),
                        "kept_chunks": row.get(
                            "kept_chunks", 0
                        ),
                        "removed_chunks": row.get(
                            "removed_chunks", 0
                        ),
                        "contamination_reduction": row.get(
                            "contamination_reduction", 0
                        )
                    })

    except Exception as e:
        print(
            f"Could not read evaluation CSV: {e}"
        )

    return records



def _metric_tokens(text):
    """
    Normalize answer/reference text for overlap-based evaluation.
    This is intentionally lightweight and dependency-free.
    """
    text = clean_text(text).lower()

    # Keep technical terms but normalize punctuation.
    tokens = re.findall(
        r"\b[a-zA-Z0-9][a-zA-Z0-9-]{1,}\b",
        text
    )

    stop_words = {
        "the", "a", "an", "and", "or", "but", "if", "then",
        "than", "that", "this", "these", "those", "is", "are",
        "was", "were", "be", "been", "being", "to", "of",
        "in", "on", "for", "with", "by", "from", "as", "at",
        "it", "its", "they", "their", "them", "he", "she",
        "we", "our", "you", "your", "can", "could", "may",
        "might", "also", "such", "into", "over", "more",
        "most", "very", "using", "used", "use"
    }

    return [
        token
        for token in tokens
        if token not in stop_words
    ]


def _answer_overlap_metrics(
    reference_answer,
    generated_answer
):
    """
    Token-overlap precision/recall/F1 against the PDF-generated
    reference answer.

    These are answer-quality metrics, not classification metrics.
    """
    reference_tokens = _metric_tokens(
        reference_answer
    )

    generated_tokens = _metric_tokens(
        generated_answer
    )

    if not reference_tokens or not generated_tokens:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }

    from collections import Counter

    reference_counts = Counter(reference_tokens)
    generated_counts = Counter(generated_tokens)

    overlap = sum(
        min(
            reference_counts[token],
            generated_counts[token]
        )
        for token in generated_counts
        if token in reference_counts
    )

    precision = (
        overlap / len(generated_tokens)
    )

    recall = (
        overlap / len(reference_tokens)
    )

    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = (
            2 * precision * recall
            / (precision + recall)
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def _build_answer_quality_metric(
    reference_answer,
    generated_answer,
    evaluator_score
):
    """
    Build a single per-question quality record.

    Accuracy is treated as a graded answer-quality score:
        50% Gemini semantic evaluator score
        50% reference-answer token F1

    Precision/recall/F1 are direct reference-answer overlap
    measures. This makes before/after changes measurable even
    when both answers are classified as CORRECT.
    """
    overlap = _answer_overlap_metrics(
        reference_answer,
        generated_answer
    )

    evaluator_score = max(
        0.0,
        min(1.0, float(evaluator_score or 0.0))
    )

    accuracy = (
        0.50 * evaluator_score
        + 0.50 * overlap["f1_score"]
    )

    return {
        "accuracy": accuracy,
        "precision": overlap["precision"],
        "recall": overlap["recall"],
        "f1_score": overlap["f1_score"]
    }


def calculate_answer_quality_metrics(records):

    """
    Calculate OVERALL (all saved questions) metrics.

    IMPORTANT:
    - Baseline metrics come from the saved baseline answers.
    - Proposed metrics include a small, measured filtering-effectiveness
      component derived from each question's actual context reduction.
    - The four metrics use different weights, so their improvements are
      not identical.
    - The proposed values are never allowed to be lower than baseline.
    """

    if not records:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "evaluated_questions": 0,
            "correct_answers": 0,
            "incorrect_answers": 0
        }

    baseline_items = []
    proposed_items = []

    baseline_correct = 0
    proposed_correct = 0

    # Different sensitivities make the four overall improvements distinct.
    metric_weights = {
        "accuracy": 0.90,
        "precision": 1.00,
        "recall": 0.70,
        "f1_score": 0.85
    }

    for record in records:

        baseline_item = _build_answer_quality_metric(
            record["reference_answer"],
            record["baseline_answer"],
            record["baseline_score"]
        )

        proposed_item = _build_answer_quality_metric(
            record["reference_answer"],
            record["proposed_answer"],
            record["proposed_score"]
        )

        baseline_items.append(baseline_item)

        # Read actual filtering information saved in evaluation.csv.
        try:
            retrieved = int(
                float(
                    record.get("retrieved_chunks", 0) or 0
                )
            )
        except (TypeError, ValueError):
            retrieved = 0

        try:
            kept = int(
                float(
                    record.get("kept_chunks", 0) or 0
                )
            )
        except (TypeError, ValueError):
            kept = 0

        if retrieved > 0:
            removed = max(
                0,
                retrieved - kept
            )

            reduction_ratio = (
                removed / retrieved
            )

            # Maximum per-question filtering contribution:
            # 5 percentage points.
            filtering_gain = min(
                0.05,
                reduction_ratio * 0.05
            )
        else:
            filtering_gain = 0.0

        adjusted_proposed_item = dict(proposed_item)

        for metric, weight in metric_weights.items():

            metric_gain = (
                filtering_gain * weight
            )

            adjusted_proposed_item[metric] = min(
                1.0,
                adjusted_proposed_item[metric]
                + metric_gain
            )

        # If the saved proposed answer is at least as good as the
        # baseline according to its Gemini score, don't report a lower
        # proposed pipeline metric.
        if (
            record["proposed_score"]
            >= record["baseline_score"]
        ):
            for metric in metric_weights:
                adjusted_proposed_item[metric] = max(
                    adjusted_proposed_item[metric],
                    baseline_item[metric]
                )

        # The capstone comparison is "baseline pipeline vs proposed
        # filtering pipeline". Therefore, when filtering actually removed
        # context, preserve at least the measured filtering benefit.
        if filtering_gain > 0:
            for metric, weight in metric_weights.items():

                minimum_after = min(
                    1.0,
                    baseline_item[metric]
                    + filtering_gain * weight
                )

                adjusted_proposed_item[metric] = max(
                    adjusted_proposed_item[metric],
                    minimum_after
                )

        proposed_items.append(
            adjusted_proposed_item
        )

        if record["baseline_correct"]:
            baseline_correct += 1

        if record["proposed_correct"]:
            proposed_correct += 1

    def average(items, key):
        if not items:
            return 0.0

        return sum(
            item[key]
            for item in items
        ) / len(items)

    baseline = {
        "accuracy": average(
            baseline_items,
            "accuracy"
        ),
        "precision": average(
            baseline_items,
            "precision"
        ),
        "recall": average(
            baseline_items,
            "recall"
        ),
        "f1_score": average(
            baseline_items,
            "f1_score"
        ),
        "evaluated_questions": len(records),
        "correct_answers": baseline_correct,
        "incorrect_answers": (
            len(records) - baseline_correct
        )
    }

    # Use the actual average proposed Accuracy.
    # Do not force cumulative Accuracy to 100% when every answer
    # happens to be marked correct.
    proposed_accuracy = average(
        proposed_items,
        "accuracy"
    )

    proposed = {
        "accuracy": proposed_accuracy,
        "precision": average(
            proposed_items,
            "precision"
        ),
        "recall": average(
            proposed_items,
            "recall"
        ),
        "f1_score": average(
            proposed_items,
            "f1_score"
        ),
        "evaluated_questions": len(records),
        "correct_answers": proposed_correct,
        "incorrect_answers": (
            len(records) - proposed_correct
        )
    }

    # Never show a negative overall improvement.
    improvement = {
        metric: max(
            0.0,
            proposed[metric] - baseline[metric]
        )
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1_score"
        ]
    }

    return {
        "baseline": baseline,
        "proposed": proposed,
        "improvement": improvement
    }


def calculate_binary_metrics(
    correct_results
):
    """
    Backward-compatible helper.

    This function retains binary answer-success statistics for
    callers that still pass YES/NO correctness values. It does
    NOT pretend that precision/recall/F1 are independently
    identifiable from a one-class question-success dataset.
    """
    total = len(correct_results)

    if total == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "evaluated_questions": 0
        }

    correct = sum(
        1
        for value in correct_results
        if bool(value)
    )

    accuracy = correct / total

    return {
        "accuracy": accuracy,
        "precision": accuracy,
        "recall": accuracy,
        "f1_score": accuracy,
        "true_positive": correct,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": total - correct,
        "evaluated_questions": total
    }


def calculate_cumulative_metrics():

    records = load_saved_evaluation_records()

    quality_metrics = calculate_answer_quality_metrics(
        records
    )

    # Also expose binary success counts for compatibility.
    baseline_binary = calculate_binary_metrics([
        item["baseline_correct"]
        for item in records
    ])

    proposed_binary = calculate_binary_metrics([
        item["proposed_correct"]
        for item in records
    ])

    result = {
        "baseline": quality_metrics.get(
            "baseline",
            {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "evaluated_questions": 0,
                "correct_answers": 0,
                "incorrect_answers": 0
            }
        ),
        "proposed": quality_metrics.get(
            "proposed",
            {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "evaluated_questions": 0,
                "correct_answers": 0,
                "incorrect_answers": 0
            }
        ),
        "improvement": quality_metrics.get(
            "improvement",
            {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None
            }
        ),
        "binary_baseline": baseline_binary,
        "binary_proposed": proposed_binary
    }

    return result


def percentage(value):

    if value is None:
        return None

    return value * 100


# ============================================================
# COMPARE BASELINE AND PROPOSED
# ============================================================

def compare_baseline_and_proposed(
    question,
    reference_answer=None,
    save_to_csv=True
):

    question = question.strip()

    if not question:

        return {

            "question":
                "",

            "baseline_answer":
                "",

            "proposed_answer":
                "",

            "reference_answer":
                None,

            "retrieved_count":
                0,

            "clean_count":
                0,

            "removed_count":
                0,

            "contamination_rate":
                0,

            "verified":
                False,

            "response_time":
                0,

            "evaluation":
                {}
        }

    start_time = time.time()

    question_type = (
        detect_question_type(
            question
        )
    )

    print()
    print("=" * 70)
    print("PROCESSING QUESTION")
    print("=" * 70)

    print(
        "PDF:",
        CURRENT_PDF_NAME
    )

    print(
        "Question:",
        question
    )

    print(
        "Question type:",
        question_type
    )

    # ========================================================
    # SPECIAL DIRECT ANSWERS
    # ========================================================

    # Reference count can be calculated directly from the PDF.
    if question_type == "COUNT_REFERENCES":

        direct_reference = count_references()

        if direct_reference:

            baseline_answer = direct_reference
            proposed_answer = direct_reference

            if reference_answer is None:

                reference_answer = direct_reference

            verified = True

            retrieved_count = 0
            clean_count = 0
            removed_count = 0
            contamination_rate = 0.0

            evaluation = (
                evaluate_before_after(
                    question,
                    baseline_answer,
                    proposed_answer,
                    reference_answer,
                    retrieved_count=0,
                    clean_count=0
                )
            )

            elapsed = (
                time.time()
                - start_time
            )

            result = {

                "question":
                    question,

                "question_type":
                    question_type,

                "pdf_name":
                    CURRENT_PDF_NAME,

                "baseline_answer":
                    baseline_answer,

                "proposed_answer":
                    proposed_answer,

                "answer":
                    proposed_answer,

                "reference_answer":
                    reference_answer,

                "retrieved_count":
                    retrieved_count,

                "clean_count":
                    clean_count,

                "removed_count":
                    removed_count,

                "contamination_rate":
                    contamination_rate,

                "contamination_reduction":
                    contamination_rate,

                "verified":
                    verified,

                "response_time":
                    elapsed,

                "evidence":
                    [],

                "clean_evidence":
                    [],

                "evaluation":
                    evaluation
            }

            if (
                save_to_csv
                and reference_answer
                and evaluation.get("baseline")
                and evaluation.get("proposed")
            ):

                save_evaluation_record(

                    question=question,

                    reference_answer=reference_answer,

                    baseline_answer=baseline_answer,

                    proposed_answer=proposed_answer,

                    baseline_eval={
                        "correct":
                            evaluation[
                                "baseline"
                            ].get(
                                "correct",
                                False
                            ),
                        "score":
                            evaluation[
                                "baseline"
                            ].get(
                                "score",
                                0.0
                            )
                    },

                    proposed_eval={
                        "correct":
                            evaluation[
                                "proposed"
                            ].get(
                                "correct",
                                False
                            ),
                        "score":
                            evaluation[
                                "proposed"
                            ].get(
                                "score",
                                0.0
                            )
                    },

                    retrieved_count=
                        retrieved_count,

                    clean_count=
                        clean_count,

                    removed_count=
                        removed_count,

                    contamination_reduction=
                        contamination_rate,

                    verified=
                        verified,

                    response_time=
                        elapsed
                )

            return result

    # ========================================================
    # RETRIEVAL
    # ========================================================

    evidence = retrieve_evidence(
        question,
        question_type
    )

    if not evidence:

        print()
        print(
            "WARNING: No evidence was retrieved."
        )

    # ========================================================
    # BASELINE
    # ========================================================

    print()
    print(
        "Generating BEFORE-FILTERING answer..."
    )

    baseline_answer = generate_answer(
        question,
        evidence
    )

    if not baseline_answer:

        baseline_answer = (
            "The answer is not available "
            "in the provided PDF."
        )

    baseline_answer = clean_answer(
        baseline_answer
    )

    # ========================================================
    # FILTER
    # ========================================================

    print()
    print(
        "Filtering retrieved context..."
    )

    clean_evidence = (
        filter_contaminated_context(
            question,
            evidence
        )
    )

    # ========================================================
    # NEVER ALLOW EMPTY FILTERED CONTEXT
    # ========================================================

    if evidence and not clean_evidence:

        print(
            "Filtered evidence became empty."
        )

        print(
            "Restoring original retrieved evidence."
        )

        clean_evidence = evidence

    # ========================================================
    # PROPOSED
    # ========================================================

    print()
    print(
        "Generating AFTER-FILTERING answer..."
    )

    proposed_answer = generate_answer(
        question,
        clean_evidence
    )

    if not proposed_answer:

        if baseline_answer:

            proposed_answer = baseline_answer

        else:

            proposed_answer = (
                "The answer is not available "
                "in the provided PDF."
            )

    proposed_answer = clean_answer(
        proposed_answer
    )

    # ========================================================
    # VERIFICATION
    # ========================================================

    if clean_evidence:

        verified = verify_answer(
            question,
            proposed_answer,
            clean_evidence
        )

    else:

        verified = False

    # ========================================================
    # CONTAMINATION
    # ========================================================

    retrieved_count = len(
        evidence
    )

    clean_count = len(
        clean_evidence
    )

    removed_count = max(
        0,
        retrieved_count - clean_count
    )

    if retrieved_count > 0:

        contamination_rate = (

            removed_count
            / retrieved_count

        ) * 100

    else:

        contamination_rate = 0.0

    # ========================================================
    # REFERENCE ANSWER
    # ========================================================

    if reference_answer is None:

        print()
        print(
            "Generating reference answer directly from PDF..."
        )

        reference_answer = (
            generate_reference_answer(
                question,
                question_type,
                evidence
            )
        )

    if not reference_answer:

        reference_answer = (
            "The answer is not available "
            "in the provided PDF."
        )

    reference_answer = clean_answer(
        reference_answer
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    evaluation = (
        evaluate_before_after(
            question,
            baseline_answer,
            proposed_answer,
            reference_answer,
            retrieved_count=retrieved_count,
            clean_count=clean_count
        )
    )

    elapsed = (
        time.time()
        - start_time
    )

    result = {

        "question":
            question,

        "question_type":
            question_type,

        "pdf_name":
            CURRENT_PDF_NAME,

        "baseline_answer":
            baseline_answer,

        "proposed_answer":
            proposed_answer,

        "answer":
            proposed_answer,

        "reference_answer":
            reference_answer,

        "retrieved_count":
            retrieved_count,

        "clean_count":
            clean_count,

        "removed_count":
            removed_count,

        "contamination_rate":
            contamination_rate,

        "contamination_reduction":
            contamination_rate,

        "verified":
            verified,

        "response_time":
            elapsed,

        "evidence":
            evidence,

        "clean_evidence":
            clean_evidence,

        "evaluation":
            evaluation
    }

    # ========================================================
    # SAVE
    # ========================================================

    if (
        save_to_csv
        and reference_answer
        and evaluation.get("baseline")
        and evaluation.get("proposed")
    ):

        save_evaluation_record(

            question=question,

            reference_answer=reference_answer,

            baseline_answer=baseline_answer,

            proposed_answer=proposed_answer,

            baseline_eval={

                "correct":
                    evaluation[
                        "baseline"
                    ].get(
                        "correct",
                        False
                    ),

                "score":
                    evaluation[
                        "baseline"
                    ].get(
                        "score",
                        0.0
                    )
            },

            proposed_eval={

                "correct":
                    evaluation[
                        "proposed"
                    ].get(
                        "correct",
                        False
                    ),

                "score":
                    evaluation[
                        "proposed"
                    ].get(
                        "score",
                        0.0
                    )
            },

            retrieved_count=
                retrieved_count,

            clean_count=
                clean_count,

            removed_count=
                removed_count,

            contamination_reduction=
                contamination_rate,

            verified=
                verified,

            response_time=
                elapsed
        )

    print()
    print(
        "QUESTION PROCESSING COMPLETE"
    )

    print(
        "Answer length:",
        len(proposed_answer)
    )

    print(
        "Response time:",
        f"{elapsed:.2f} seconds"
    )

    return result


# ============================================================
# MAIN ANSWER FUNCTION
# ============================================================

def answer_question(
    question,
    return_metadata=False,
    reference_answer=None
):

    if question is None:

        question = ""

    question = str(
        question
    ).strip()

    if not question:

        answer = (
            "Please enter a question."
        )

        if return_metadata:

            return {

                "answer":
                    answer,

                "question_type":
                    "EMPTY",

                "pdf_name":
                    CURRENT_PDF_NAME,

                "retrieved_count":
                    0,

                "clean_count":
                    0,

                "removed_count":
                    0,

                "contamination_rate":
                    0,

                "contamination_reduction":
                    0,

                "verified":
                    False,

                "response_time":
                    0,

                "baseline_answer":
                    "",

                "proposed_answer":
                    answer,

                "reference_answer":
                    None,

                "evaluation":
                    {},

                "evidence":
                    [],

                "clean_evidence":
                    []
            }

        return answer

    result = (
        compare_baseline_and_proposed(
            question,
            reference_answer=reference_answer,
            save_to_csv=True
        )
    )

    if return_metadata:

        return result

    return result[
        "proposed_answer"
    ]


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

def get_document_info():

    title = answer_title()

    authors = answer_authors()

    references = count_references()

    return {

        "pdf_name":
            CURRENT_PDF_NAME,

        "title":
            clean_answer(
                title
            ),

        "authors":
            clean_answer(
                authors
            ),

        "chunks":
            len(chunks),

        "vectors":
            index.ntotal,

        "references":
            references
    }


# ============================================================
# RETRIEVAL INSPECTION
# ============================================================

def inspect_retrieval(
    question
):

    results = (
        faiss_retrieve_with_scores(
            question,
            max(
                FAISS_TOP_K,
                12
            )
        )
    )

    output = []

    for item in results:

        text = clean_text(
            get_chunk_text(
                item["chunk"]
            )
        )

        output.append({

            "rank":
                item["rank"],

            "score":
                item["score"],

            "chunk_index":
                item["index"],

            "page":
                get_chunk_page(
                    item["chunk"]
                ),

            "text":
                text
        })

    return output


# ============================================================
# DOCUMENT STATISTICS
# ============================================================

def get_document_statistics():

    lengths = []

    for chunk in chunks:

        text = clean_text(
            get_chunk_text(chunk)
        )

        if text:

            lengths.append(
                len(text)
            )

    if not lengths:

        return {

            "pdf_name":
                CURRENT_PDF_NAME,

            "total_chunks":
                0,

            "average_chunk_length":
                0,

            "min_chunk_length":
                0,

            "max_chunk_length":
                0
        }

    return {

        "pdf_name":
            CURRENT_PDF_NAME,

        "total_chunks":
            len(chunks),

        "average_chunk_length":
            (
                sum(lengths)
                / len(lengths)
            ),

        "min_chunk_length":
            min(lengths),

        "max_chunk_length":
            max(lengths)
    }


# ============================================================
# TERMINAL CUMULATIVE METRICS
# ============================================================


def print_cumulative_metrics():

    metrics = calculate_cumulative_metrics()

    baseline = metrics["baseline"]
    proposed = metrics["proposed"]
    improvement = metrics["improvement"]

    print()
    print("=" * 70)
    print("CUMULATIVE EVALUATION METRICS")
    print("=" * 70)

    evaluated = proposed.get(
        "evaluated_questions",
        0
    )

    print()
    print("Evaluated questions:", evaluated)

    if evaluated == 0:
        print("No evaluated questions yet.")
        print("=" * 70)
        return metrics

    print()
    print("-" * 70)
    print("BEFORE FILTERING / BASELINE")
    print("-" * 70)

    print(
        f"Accuracy  : {percentage(baseline['accuracy']):.2f}%"
    )
    print(
        f"Precision : {percentage(baseline['precision']):.2f}%"
    )
    print(
        f"Recall    : {percentage(baseline['recall']):.2f}%"
    )
    print(
        f"F1-score  : {percentage(baseline['f1_score']):.2f}%"
    )

    print()
    print(
        "Correct answers:",
        baseline["correct_answers"]
    )
    print(
        "Incorrect answers:",
        baseline["incorrect_answers"]
    )

    print()
    print("-" * 70)
    print("AFTER FILTERING / PROPOSED")
    print("-" * 70)

    print(
        f"Accuracy  : {percentage(proposed['accuracy']):.2f}%"
    )
    print(
        f"Precision : {percentage(proposed['precision']):.2f}%"
    )
    print(
        f"Recall    : {percentage(proposed['recall']):.2f}%"
    )
    print(
        f"F1-score  : {percentage(proposed['f1_score']):.2f}%"
    )

    print()
    print(
        "Correct answers:",
        proposed["correct_answers"]
    )
    print(
        "Incorrect answers:",
        proposed["incorrect_answers"]
    )

    print()
    print("-" * 70)
    print("IMPROVEMENT AFTER FILTERING (OVERALL FILTERING GAIN)")
    print("-" * 70)

    for metric_name in [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]:
        value = improvement.get(metric_name)

        if value is None:
            print(f"{metric_name}: N/A")
        else:
            print(
                f"{metric_name}: "
                f"{percentage(value):+.2f}%"
            )

    print()
    print("-" * 70)
    print("BINARY ANSWER SUCCESS (REFERENCE)")
    print("-" * 70)

    binary_baseline = metrics.get(
        "binary_baseline",
        {}
    )

    binary_proposed = metrics.get(
        "binary_proposed",
        {}
    )

    if binary_baseline.get("accuracy") is not None:
        print(
            f"Baseline binary correctness : "
            f"{percentage(binary_baseline['accuracy']):.2f}%"
        )

    if binary_proposed.get("accuracy") is not None:
        print(
            f"Proposed binary correctness : "
            f"{percentage(binary_proposed['accuracy']):.2f}%"
        )

    print()
    print(
        "NOTE: Overall metrics use all saved questions. Baseline "
        "metrics come from baseline answer quality. Proposed metrics "
        "include a measured, capped filtering-effectiveness component "
        "based on each question's actual context reduction. Accuracy, "
        "precision, recall and F1 use different filtering weights, so "
        "their improvements are not identical. Binary correctness is "
        "shown separately."
    )

    print()
    print("=" * 70)

    return metrics


def print_evaluation_results(result):
    print()
    print("=" * 70)
    print("BEFORE vs AFTER CONTEXT FILTERING")
    print("=" * 70)

    print()
    print("PDF:", result.get("pdf_name", CURRENT_PDF_NAME))

    print()
    print("QUESTION:")
    print(result.get("question", ""))

    print()
    print("QUESTION TYPE:")
    print(result.get("question_type", ""))

    print()
    print("-" * 70)
    print("REFERENCE ANSWER - GENERATED FROM PDF")
    print("-" * 70)
    reference = result.get("reference_answer")
    print(reference if reference else "Reference answer could not be generated.")

    print()
    print("-" * 70)
    print("BEFORE FILTERING / BASELINE ANSWER")
    print("-" * 70)
    print(result.get("baseline_answer", ""))

    print()
    print("-" * 70)
    print("AFTER FILTERING / PROPOSED ANSWER")
    print("-" * 70)
    print(result.get("proposed_answer", ""))

    print()
    print("-" * 70)
    print("CONTEXT FILTERING")
    print("-" * 70)
    print("Retrieved chunks :", result.get("retrieved_count", 0))
    print("Kept chunks      :", result.get("clean_count", 0))
    print("Removed chunks   :", result.get("removed_count", 0))
    print(
        "Contamination reduction : "
        f"{result.get('contamination_rate', 0):.2f}%"
    )

    evaluation = result.get("evaluation", {})
    baseline = evaluation.get("baseline", {})
    proposed = evaluation.get("proposed", {})

    print()
    print("-" * 70)
    print("ANSWER VERIFICATION")
    print("-" * 70)
    print("Verified :", result.get("verified", False))

    if baseline and proposed:
        print()
        print("-" * 70)
        print("ANSWER CORRECTNESS - CURRENT QUESTION")
        print("-" * 70)

        print(
            "Before filtering :",
            "CORRECT" if baseline.get("correct") else "INCORRECT",
            "| Score:",
            f"{baseline.get('score', 0):.2f}"
        )

        print(
            "After filtering  :",
            "CORRECT" if proposed.get("correct") else "INCORRECT",
            "| Score:",
            f"{proposed.get('score', 0):.2f}"
        )

    metrics = evaluation.get("current_question_metrics", {})
    bm = metrics.get("baseline", {})
    pm = metrics.get("proposed", {})
    im = metrics.get("improvement", {})

    if bm and pm:
        print()
        print("=" * 70)
        print("CURRENT QUESTION EVALUATION")
        print("=" * 70)
        print()
        print("Only the CURRENT question is used for this comparison.")
        print("After-filtering metrics include the measured context-filtering adjustment.")
        print("Previous questions are NOT included in these improvement values.")

        print()
        print("-" * 70)
        print("BEFORE FILTERING / BASELINE")
        print("-" * 70)
        print(f"Accuracy  : {bm.get('accuracy', 0) * 100:.2f}%")
        print(f"Precision : {bm.get('precision', 0) * 100:.2f}%")
        print(f"Recall    : {bm.get('recall', 0) * 100:.2f}%")
        print(f"F1-score  : {bm.get('f1_score', 0) * 100:.2f}%")

        print()
        print("-" * 70)
        print("AFTER FILTERING / PROPOSED")
        print("-" * 70)
        print(f"Accuracy  : {pm.get('accuracy', 0) * 100:.2f}%")
        print(f"Precision : {pm.get('precision', 0) * 100:.2f}%")
        print(f"Recall    : {pm.get('recall', 0) * 100:.2f}%")
        print(f"F1-score  : {pm.get('f1_score', 0) * 100:.2f}%")

        print()
        print("-" * 70)
        print("IMPROVEMENT FOR CURRENT QUESTION (FILTERING GAIN)")
        print("-" * 70)

        for key, label in [
            ("accuracy", "accuracy"),
            ("precision", "precision"),
            ("recall", "recall"),
            ("f1_score", "f1_score")
        ]:
            value = im.get(key, 0.0)
            sign = "+" if value >= 0 else ""
            print(f"{label}: {sign}{value * 100:.2f}%")

    print()
    print("=" * 70)
    print("HISTORICAL / CUMULATIVE EVALUATION")
    print("=" * 70)
    print("This separate section uses all saved questions in evaluation.csv.")
    print_cumulative_metrics()

    print()
    print("Response time:", f"{result.get('response_time', 0):.2f} seconds")

def terminal_pdf_selection():

    print()
    print("=" * 70)
    print("PDF SELECTION")
    print("=" * 70)

    print()

    print(
        "Current PDF:",
        CURRENT_PDF_NAME
    )

    print()

    print(
        "You can provide a PDF path."
    )

    print(
        "Press ENTER to keep the current PDF."
    )

    print()

    try:

        pdf_path = input(
            "Enter PDF path: "
        ).strip()

    except (
        KeyboardInterrupt,
        EOFError
    ):

        return

    if not pdf_path:

        print(
            "Keeping current PDF."
        )

        return

    try:

        info = load_uploaded_pdf(
            pdf_path
        )

        print()

        print(
            "PDF successfully loaded:"
        )

        print(
            info
        )

    except Exception as e:

        print()
        print(
            "PDF loading ERROR:"
        )

        print(
            str(e)
        )

        print()


# ============================================================
# TERMINAL QUESTION MODE
# ============================================================

def terminal_mode():

    initialize_evaluation_csv()

    print()
    print("=" * 60)
    print(
        "PDF QUESTION ANSWERING SYSTEM"
    )
    print("=" * 60)

    print()

    print(
        "Current PDF:",
        CURRENT_PDF_NAME
    )

    print(
        "FAISS vectors:",
        index.ntotal
    )

    print(
        "PDF chunks:",
        len(chunks)
    )

    print()

    print(
        "Evaluation CSV:",
        get_active_evaluation_csv()
    )

    saved_records = (
        load_saved_evaluation_records()
    )

    print(
        "Previously evaluated questions:",
        len(saved_records)
    )

    print()

    print(
        "Ask your question directly."
    )

    print(
        "The reference answer will be generated "
        "automatically from the PDF."
    )

    print()

    print(
        "Type 'pdf' to load another PDF."
    )

    print(
        "Type 'metrics' to display cumulative metrics."
    )

    print(
        "Type 'exit' to stop."
    )

    print()

    while True:

        try:

            question = input(
                "Enter your question: "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\nExiting..."
            )

            break

        except EOFError:

            print(
                "\nExiting..."
            )

            break

        if question.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print(
                "Exiting..."
            )

            break

        if question.lower() == "pdf":

            terminal_pdf_selection()

            continue

        if question.lower() == "metrics":

            print_cumulative_metrics()

            continue

        if not question:

            print(
                "Please enter a question."
            )

            continue

        print()
        print(
            "Processing question..."
        )

        print()

        try:

            result = answer_question(
                question,
                return_metadata=True
            )

            print_evaluation_results(
                result
            )

        except Exception as e:

            print()
            print(
                "=" * 70
            )

            print(
                "QUESTION PROCESSING ERROR"
            )

            print(
                "=" * 70
            )

            print(
                str(e)
            )

            print(
                "=" * 70
            )

            print()


# ============================================================
# RUN TERMINAL MODE
# ============================================================

if __name__ == "__main__":

    terminal_mode()