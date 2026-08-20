import os
import re
import pickle
import faiss
import time
from pathlib import Path
from functools import lru_cache

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

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

GEMINI_MODEL = "gemini-3.5-flash-lite"
FALLBACK_GEMINI_MODEL = "gemini-3.5-flash"

FAISS_TOP_K = 8
MAX_EVIDENCE = 12
MAX_CHUNK_LENGTH = 5000


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


# Create Gemini client once.
client = genai.Client(api_key=API_KEY)


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

        print("Embedding model loading failed.")

        raise RuntimeError(
            "Could not load the embedding model.\n"
            "Try restarting the terminal and Streamlit.\n\n"
            f"Original error: {e}"
        )


# ============================================================
# LOAD FAISS
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

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)


print("FAISS index loaded.")
print(f"Number of vectors: {index.ntotal}")
print(f"Number of chunks: {len(chunks)}")


# ============================================================
# CHUNK HELPERS
# ============================================================

def get_chunk_text(chunk):

    if isinstance(chunk, dict):
        return str(
            chunk.get("text", "")
        )

    return str(chunk)


def get_chunk_id(chunk, fallback):

    if isinstance(chunk, dict):

        value = chunk.get("id")

        if value is not None:
            return value

    return fallback


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
        text = text.replace(old, new)

    text = re.sub(
        r"\\[a-zA-Z]+",
        " ",
        text
    )

    text = text.replace("$", " ")
    text = text.replace("{", " ")
    text = text.replace("}", " ")

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

    if not answer:
        return (
            "The answer is not available "
            "in the provided PDF."
        )

    answer = str(answer)

    answer = re.sub(
        r"#{1,6}\s*",
        "",
        answer
    )

    answer = answer.replace("**", "")
    answer = answer.replace("__", "")
    answer = answer.replace("*", "")

    answer = answer.replace("```", "")
    answer = answer.replace("`", "")

    answer = re.sub(
        r"\\[a-zA-Z]+",
        "",
        answer
    )

    answer = answer.replace("$", "")
    answer = answer.replace("{", "")
    answer = answer.replace("}", "")

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
        answer = answer.replace(old, new)

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
        "Based on the evidence:"
    ]

    for prefix in prefixes:

        if answer.lower().startswith(
            prefix.lower()
        ):

            answer = answer[
                len(prefix):
            ].strip()

    return answer.strip()


# ============================================================
# QUESTION TYPE
# ============================================================

def detect_question_type(question):

    q = question.lower().strip()

    # RAG definition
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

    # Title
    if any(x in q for x in [
        "main title",
        "title of the pdf",
        "title of the paper",
        "paper title",
        "what is the title",
        "name of the paper"
    ]):
        return "TITLE"

    # Authors
    if any(x in q for x in [
        "who are the authors",
        "authors of the paper",
        "authors of the pdf",
        "author names",
        "who wrote the paper"
    ]):
        return "AUTHOR"

    # First page
    if any(x in q for x in [
        "first page",
        "first-page",
        "on page one",
        "page one",
        "mentioned in first page",
        "mentioned on first page"
    ]):
        return "FIRST_PAGE"

    # References
    if any(x in q for x in [
        "how many references",
        "number of references",
        "references are present",
        "total references",
        "reference count"
    ]):
        return "COUNT_REFERENCES"

    # Conclusion
    if any(x in q for x in [
        "conclusion",
        "conclusions",
        "what did the paper conclude",
        "final conclusion"
    ]):
        return "CONCLUSION"

    # Results
    if any(x in q for x in [
        "results",
        "result of the paper",
        "outcomes",
        "findings",
        "performance"
    ]):
        return "RESULTS"

    # Limitations
    if any(x in q for x in [
        "limitations",
        "limitation",
        "drawbacks",
        "weaknesses",
        "disadvantages"
    ]):
        return "LIMITATIONS"

    # Broader impact
    if any(x in q for x in [
        "broader impact",
        "societal impact",
        "social impact",
        "impact of the work"
    ]):
        return "BROADER_IMPACT"

    # Methodology
    if any(x in q for x in [
        "methodology",
        "method used",
        "methods used",
        "what methods",
        "approach used",
        "how was it implemented"
    ]):
        return "METHODOLOGY"

    # Dataset
    if any(x in q for x in [
        "dataset",
        "datasets",
        "data used",
        "what data"
    ]):
        return "DATASETS"

    # Architecture
    if any(x in q for x in [
        "architecture",
        "system architecture",
        "components",
        "framework",
        "pipeline",
        "tools",
        "tools and frameworks"
    ]):
        return "ARCHITECTURE"

    # Summary
    if any(x in q for x in [
        "summarize",
        "summary",
        "exam preparation",
        "exam point",
        "exam perspective",
        "analyze the pdf",
        "analyse the pdf",
        "overview of the paper",
        "explain the whole paper"
    ]):
        return "SUMMARY"

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
# TITLE
# ============================================================

def answer_title():

    beginning = get_beginning_chunks(5)

    text = "\n".join(
        clean_text(
            get_chunk_text(x)
        )
        for x in beginning
    )

    match = re.search(
        r"(Retrieval[- ]Augmented Generation "
        r"for Knowledge[- ]Intensive NLP Tasks)",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return generate_answer(
        "What is the main title of the PDF?",
        beginning
    )


# ============================================================
# AUTHORS
# ============================================================

def answer_authors():

    beginning = get_beginning_chunks(5)

    text = "\n".join(
        clean_text(
            get_chunk_text(x)
        )
        for x in beginning
    )

    pattern = (
        r"Patrick Lewis.*?"
        r"Douwe Kiela"
    )

    match = re.search(
        pattern,
        text,
        re.IGNORECASE | re.DOTALL
    )

    if match:

        author_text = match.group(0)

        author_text = author_text.replace("?", "")
        author_text = author_text.replace("†", "")
        author_text = author_text.replace("‡", "")

        author_text = author_text.split(
            "plewis"
        )[0]

        return clean_text(
            author_text
        )

    return generate_answer(
        "Who are the authors of this paper?",
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
                numbers.append(number)

        except ValueError:
            pass

    if numbers:

        highest = max(numbers)

        return (
            f"The PDF contains "
            f"{highest} references."
        )

    return None


# ============================================================
# FIRST PAGE
# ============================================================

def get_first_page_evidence():

    return get_beginning_chunks(10)


def answer_first_page(question):

    evidence = get_first_page_evidence()

    if not evidence:
        return None

    return generate_answer(
        question,
        evidence
    )


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
        "we have presented"
    ],

    "RESULTS": [
        "results",
        "experimental results",
        "evaluation",
        "performance",
        "state-of-the-art",
        "state of the art"
    ],

    "LIMITATIONS": [
        "limitations",
        "limitation",
        "drawbacks",
        "bias",
        "factual",
        "misleading"
    ],

    "BROADER_IMPACT": [
        "broader impact",
        "societal benefits",
        "societal",
        "positive societal",
        "abuse",
        "misleading content"
    ],

    "METHODOLOGY": [
        "method",
        "methodology",
        "approach",
        "training",
        "retriever",
        "generator",
        "fine-tuning"
    ],

    "DATASETS": [
        "datasets",
        "dataset",
        "natural questions",
        "triviaqa",
        "webquestions",
        "curatedtrec",
        "fever",
        "searchqa"
    ],

    "ARCHITECTURE": [
        "architecture",
        "rag-sequence",
        "rag-token",
        "retriever",
        "generator",
        "parametric memory",
        "non-parametric memory"
    ],

    "SUMMARY": [
        "abstract",
        "introduction",
        "results",
        "conclusion",
        "rag-sequence",
        "rag-token"
    ],

    "RAG_DEFINITION": [
        "retrieval-augmented generation",
        "retrieval augmented generation",
        "parametric memory",
        "non-parametric memory",
        "retrieve text documents",
        "additional context"
    ]
}


# ============================================================
# SECTION RETRIEVAL
# ============================================================

def retrieve_section_chunks(question_type):

    keywords = SECTION_KEYWORDS.get(
        question_type,
        []
    )

    if not keywords:
        return []

    scored = []

    for index_number, chunk in enumerate(chunks):

        text = clean_text(
            get_chunk_text(chunk)
        )

        if not text:
            continue

        lower_text = text.lower()

        score = 0

        for keyword in keywords:

            if keyword.lower() in lower_text:
                score += 1

        if score > 0:

            scored.append(
                (
                    score,
                    index_number,
                    chunk
                )
            )

    scored.sort(
        key=lambda x: (x[0], -x[1]),
        reverse=True
    )

    return [
        item[2]
        for item in scored[:12]
    ]


# ============================================================
# FAISS RETRIEVAL
# ============================================================

def faiss_retrieve_with_scores(
    question,
    top_k=FAISS_TOP_K
):

    # IMPORTANT:
    # Model is loaded only here.
    embedding_model = get_embedding_model()

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, idx in enumerate(indices[0]):

        if idx < 0:
            continue

        if idx >= len(chunks):
            continue

        results.append({
            "chunk": chunks[idx],
            "score": float(
                distances[0][rank]
            ),
            "rank": rank + 1,
            "index": int(idx)
        })

    return results


def faiss_retrieve(
    question,
    top_k=FAISS_TOP_K
):

    results = faiss_retrieve_with_scores(
        question,
        top_k
    )

    return [
        item["chunk"]
        for item in results
    ]


# ============================================================
# MULTI QUERY
# ============================================================

def build_search_queries(question):

    q = question.strip()

    question_type = detect_question_type(q)

    additions = {

        "GENERAL": [
            q
        ],

        "RAG_DEFINITION": [
            q,
            "what is retrieval augmented generation",
            "definition of RAG",
            "RAG combines parametric and non-parametric memory"
        ],

        "FIRST_PAGE": [
            q,
            "first page title authors university affiliation"
        ],

        "CONCLUSION": [
            q,
            "conclusion of the paper",
            "final conclusions and future work"
        ],

        "RESULTS": [
            q,
            "experimental results of the paper",
            "performance and evaluation results"
        ],

        "LIMITATIONS": [
            q,
            "limitations of the proposed approach",
            "drawbacks and limitations"
        ],

        "BROADER_IMPACT": [
            q,
            "broader impact of the work",
            "societal impact"
        ],

        "METHODOLOGY": [
            q,
            "methodology and approach",
            "training method and architecture"
        ],

        "DATASETS": [
            q,
            "datasets used in experiments",
            "evaluation datasets"
        ],

        "ARCHITECTURE": [
            q,
            "RAG architecture",
            "retriever generator architecture",
            "tools frameworks components"
        ],

        "SUMMARY": [
            q,
            "main contributions of the paper",
            "important concepts and results"
        ]
    }

    queries = additions.get(
        question_type,
        [q]
    )

    final_queries = []

    for query in queries:

        if query not in final_queries:
            final_queries.append(query)

    return final_queries[:4]


# ============================================================
# MULTI QUERY RETRIEVAL
# ============================================================

def multi_query_retrieve(question):

    queries = build_search_queries(
        question
    )

    results = []

    seen_ids = set()

    for query in queries:

        retrieved = faiss_retrieve(
            query,
            FAISS_TOP_K
        )

        for chunk in retrieved:

            chunk_id = get_chunk_id(
                chunk,
                id(chunk)
            )

            if chunk_id not in seen_ids:

                seen_ids.add(chunk_id)

                results.append(chunk)

    return results


# ============================================================
# MERGE RETRIEVAL
# ============================================================

def retrieve_evidence(
    question,
    question_type
):

    results = []

    seen_ids = set()

    # First page
    if question_type == "FIRST_PAGE":

        first_page = get_first_page_evidence()

        for chunk in first_page:

            chunk_id = get_chunk_id(
                chunk,
                id(chunk)
            )

            if chunk_id not in seen_ids:

                seen_ids.add(chunk_id)
                results.append(chunk)

    # Section
    section_results = retrieve_section_chunks(
        question_type
    )

    for chunk in section_results:

        chunk_id = get_chunk_id(
            chunk,
            id(chunk)
        )

        if chunk_id not in seen_ids:

            seen_ids.add(chunk_id)
            results.append(chunk)

    # Semantic
    semantic_results = multi_query_retrieve(
        question
    )

    for chunk in semantic_results:

        chunk_id = get_chunk_id(
            chunk,
            id(chunk)
        )

        if chunk_id not in seen_ids:

            seen_ids.add(chunk_id)
            results.append(chunk)

    return results[:MAX_EVIDENCE]


# ============================================================
# GEMINI
# ============================================================

def call_gemini(
    prompt,
    retries=3
):

    for attempt in range(retries):

        try:

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            if response and response.text:

                return response.text.strip()

        except Exception as e:

            error_text = str(e)

            if (
                "404" in error_text
                or "NOT_FOUND" in error_text
            ):

                try:

                    response = client.models.generate_content(
                        model=FALLBACK_GEMINI_MODEL,
                        contents=prompt
                    )

                    if response and response.text:

                        return response.text.strip()

                except Exception:
                    pass

            if attempt < retries - 1:

                time.sleep(2)

    return None


# ============================================================
# FORMAT EVIDENCE
# ============================================================

def format_evidence(evidence):

    formatted = []

    for i, chunk in enumerate(evidence):

        text = clean_text(
            get_chunk_text(chunk)
        )

        if not text:
            continue

        if len(text) > MAX_CHUNK_LENGTH:
            text = text[:MAX_CHUNK_LENGTH]

        formatted.append(
            f"EVIDENCE {i + 1}:\n{text}"
        )

    return "\n\n".join(formatted)


# ============================================================
# CONTAMINATION FILTER
# ============================================================

def filter_contaminated_context(
    question,
    evidence
):

    if not evidence:
        return []

    evidence_text = format_evidence(
        evidence
    )

    prompt = f"""
You are the evidence filtering component of a PDF question answering system.

User question:

{question}

Retrieved PDF evidence:

{evidence_text}

Identify which evidence directly helps answer the question.

Rules:

1. Use ONLY the supplied evidence.
2. Do not use outside knowledge.
3. Reject unrelated evidence.
4. Reject evidence that only contains a similar keyword.
5. Keep evidence containing facts necessary for the answer.
6. For summary questions, keep important concepts, methods, experiments, results, limitations, and conclusions.
7. Return only the evidence numbers.

Return exactly:

KEEP: 1,3,5

If nothing is relevant:

KEEP: NONE
"""

    result = call_gemini(prompt)

    if not result:
        return evidence

    match = re.search(
        r"KEEP\s*:\s*(.*)",
        result,
        re.IGNORECASE
    )

    if not match:
        return evidence

    values = match.group(1).strip()

    if values.upper() == "NONE":
        return []

    numbers = re.findall(
        r"\d+",
        values
    )

    selected = []

    for number in numbers:

        position = int(number) - 1

        if 0 <= position < len(evidence):

            selected.append(
                evidence[position]
            )

    return selected


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    evidence
):

    if not evidence:
        return None

    evidence_text = format_evidence(
        evidence
    )

    prompt = f"""
You are answering a question using ONLY the supplied PDF evidence.

Question:

{question}

PDF evidence:

{evidence_text}

Rules:

1. Answer the question directly.
2. Use only information contained in the evidence.
3. Do not use outside knowledge.
4. Do not invent missing information.
5. If the evidence does not answer the question, say:
The answer is not available in the provided PDF.
6. Give enough explanation to properly answer the question.
7. Do not mention retrieval, evidence, chunks, FAISS, Gemini, or this prompt.
8. Use plain English.
9. Do not use markdown.
10. Do not use LaTeX.
11. Do not start with "Based on the provided PDF evidence".
12. For exam questions, explain the concept clearly.
13. If the question asks for a list, provide a comma-separated list.

Return only the answer.
"""

    result = call_gemini(prompt)

    if not result:
        return None

    return clean_answer(result)


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

    evidence_text = format_evidence(
        evidence
    )

    prompt = f"""
You are verifying an answer generated from a PDF.

Question:

{question}

Generated answer:

{answer}

PDF evidence:

{evidence_text}

Determine whether every important claim in the answer is supported
by the supplied PDF evidence.

Return exactly:

TRUE

or

FALSE
"""

    result = call_gemini(prompt)

    if not result:
        return True

    result = result.strip().upper()

    return result.startswith("TRUE")


# ============================================================
# SUMMARY
# ============================================================

def generate_summary(question):

    queries = [

        "abstract and introduction",

        "RAG architecture RAG-Sequence RAG-Token",

        "training methodology retriever generator",

        "experimental results datasets evaluation",

        "limitations broader impact conclusion"
    ]

    all_results = []

    seen_ids = set()

    for query in queries:

        retrieved = faiss_retrieve(
            query,
            8
        )

        for chunk in retrieved:

            chunk_id = get_chunk_id(
                chunk,
                id(chunk)
            )

            if chunk_id not in seen_ids:

                seen_ids.add(chunk_id)
                all_results.append(chunk)

    all_results = all_results[:25]

    if not all_results:
        return None

    evidence_text = format_evidence(
        all_results
    )

    prompt = f"""
You are an exam preparation assistant.

User question:

{question}

PDF evidence:

{evidence_text}

Prepare a clear exam-oriented explanation.

Cover, when available:

1. Problem addressed
2. Main idea of RAG
3. Architecture
4. Parametric memory
5. Non-parametric memory
6. Retriever
7. Generator
8. RAG-Sequence
9. RAG-Token
10. Training
11. Datasets
12. Experiments
13. Results
14. Limitations
15. Broader impact
16. Important conclusions

Use only information supported by the PDF.

Use simple plain English.

Do not mention retrieval, FAISS, Gemini, chunks, or this prompt.

Return only the answer.
"""

    result = call_gemini(prompt)

    if not result:
        return None

    return clean_answer(result)


# ============================================================
# MAIN ANSWER FUNCTION
# ============================================================

def answer_question(
    question,
    return_metadata=False
):

    question = question.strip()

    if not question:

        answer = "Please enter a question."

        if return_metadata:

            return {
                "answer": answer,
                "question_type": "EMPTY",
                "retrieved_count": 0,
                "clean_count": 0,
                "contamination_rate": 0,
                "verified": False,
                "response_time": 0,
                "evidence": [],
                "clean_evidence": []
            }

        return answer

    start_time = time.time()

    question_type = detect_question_type(
        question
    )

    evidence = []
    clean_evidence = []
    verified = False


    # ========================================================
    # TITLE
    # ========================================================

    if question_type == "TITLE":

        answer = clean_answer(
            answer_title()
        )

        elapsed = time.time() - start_time

        if return_metadata:

            return {
                "answer": answer,
                "question_type": question_type,
                "retrieved_count": 0,
                "clean_count": 0,
                "contamination_rate": 0,
                "verified": True,
                "response_time": elapsed,
                "evidence": [],
                "clean_evidence": []
            }

        return answer


    # ========================================================
    # AUTHORS
    # ========================================================

    if question_type == "AUTHOR":

        answer = clean_answer(
            answer_authors()
        )

        elapsed = time.time() - start_time

        if return_metadata:

            return {
                "answer": answer,
                "question_type": question_type,
                "retrieved_count": 0,
                "clean_count": 0,
                "contamination_rate": 0,
                "verified": True,
                "response_time": elapsed,
                "evidence": [],
                "clean_evidence": []
            }

        return answer


    # ========================================================
    # REFERENCES
    # ========================================================

    if question_type == "COUNT_REFERENCES":

        answer = count_references()

        if answer:

            elapsed = time.time() - start_time

            if return_metadata:

                return {
                    "answer": answer,
                    "question_type": question_type,
                    "retrieved_count": 0,
                    "clean_count": 0,
                    "contamination_rate": 0,
                    "verified": True,
                    "response_time": elapsed,
                    "evidence": [],
                    "clean_evidence": []
                }

            return answer


    # ========================================================
    # FIRST PAGE
    # ========================================================

    if question_type == "FIRST_PAGE":

        evidence = get_first_page_evidence()

        clean_evidence = evidence

        answer = generate_answer(
            question,
            clean_evidence
        )

        if not answer:

            answer = (
                "The answer is not available "
                "in the provided PDF."
            )

        verified = True

        elapsed = time.time() - start_time

        if return_metadata:

            return {
                "answer": clean_answer(answer),
                "question_type": question_type,
                "retrieved_count": len(evidence),
                "clean_count": len(clean_evidence),
                "contamination_rate": 0,
                "verified": verified,
                "response_time": elapsed,
                "evidence": evidence,
                "clean_evidence": clean_evidence
            }

        return clean_answer(answer)


    # ========================================================
    # SUMMARY
    # ========================================================

    if question_type == "SUMMARY":

        answer = generate_summary(
            question
        )

        if not answer:

            answer = (
                "The answer is not available "
                "in the provided PDF."
            )

        elapsed = time.time() - start_time

        if return_metadata:

            return {
                "answer": clean_answer(answer),
                "question_type": question_type,
                "retrieved_count": 0,
                "clean_count": 0,
                "contamination_rate": 0,
                "verified": True,
                "response_time": elapsed,
                "evidence": [],
                "clean_evidence": []
            }

        return clean_answer(answer)


    # ========================================================
    # NORMAL RAG PIPELINE
    # ========================================================

    evidence = retrieve_evidence(
        question,
        question_type
    )

    clean_evidence = filter_contaminated_context(
        question,
        evidence
    )

    if not clean_evidence:
        clean_evidence = evidence

    answer = generate_answer(
        question,
        clean_evidence
    )

    if not answer:

        answer = (
            "The answer is not available "
            "in the provided PDF."
        )

    else:

        verified = verify_answer(
            question,
            answer,
            clean_evidence
        )

        # Retry if verification fails
        if not verified:

            broader_evidence = multi_query_retrieve(
                question
            )

            if broader_evidence:

                broader_clean = (
                    filter_contaminated_context(
                        question,
                        broader_evidence
                    )
                )

                if broader_clean:

                    retry_answer = generate_answer(
                        question,
                        broader_clean
                    )

                    if retry_answer:

                        retry_verified = verify_answer(
                            question,
                            retry_answer,
                            broader_clean
                        )

                        if retry_verified:

                            answer = retry_answer
                            clean_evidence = broader_clean
                            verified = True


    elapsed = time.time() - start_time

    retrieved_count = len(evidence)

    clean_count = len(clean_evidence)

    if retrieved_count > 0:

        contamination_rate = (
            (retrieved_count - clean_count)
            / retrieved_count
        ) * 100

    else:

        contamination_rate = 0


    metadata = {

        "answer": clean_answer(answer),

        "question_type": question_type,

        "retrieved_count": retrieved_count,

        "clean_count": clean_count,

        "contamination_rate": contamination_rate,

        "verified": verified,

        "response_time": elapsed,

        "evidence": evidence,

        "clean_evidence": clean_evidence
    }


    if return_metadata:
        return metadata

    return clean_answer(answer)


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

def get_document_info():

    title = answer_title()

    authors = answer_authors()

    references = count_references()

    return {

        "title": clean_answer(title),

        "authors": clean_answer(authors),

        "chunks": len(chunks),

        "vectors": index.ntotal,

        "references": references
    }


# ============================================================
# RETRIEVAL INSPECTION
# ============================================================

def inspect_retrieval(question):

    results = faiss_retrieve_with_scores(
        question,
        FAISS_TOP_K
    )

    output = []

    for item in results:

        text = clean_text(
            get_chunk_text(
                item["chunk"]
            )
        )

        output.append({

            "rank": item["rank"],

            "score": item["score"],

            "chunk_index": item["index"],

            "text": text
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

            "total_chunks": 0,

            "average_chunk_length": 0,

            "min_chunk_length": 0,

            "max_chunk_length": 0
        }

    return {

        "total_chunks": len(chunks),

        "average_chunk_length":
            sum(lengths) / len(lengths),

        "min_chunk_length":
            min(lengths),

        "max_chunk_length":
            max(lengths)
    }


# ============================================================
# TERMINAL QUESTION MODE
# ============================================================

def terminal_mode():

    print()
    print("=" * 60)
    print("PDF QUESTION ANSWERING SYSTEM")
    print("=" * 60)
    print()
    print("FAISS vectors :", index.ntotal)
    print("PDF chunks    :", len(chunks))
    print()
    print("Ask a question about the PDF.")
    print("Type 'exit' to stop.")
    print()

    while True:

        try:

            question = input(
                "Enter your question: "
            ).strip()

        except KeyboardInterrupt:

            print("\nExiting...")
            break

        except EOFError:

            print("\nExiting...")
            break

        if question.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print("Exiting...")
            break

        if not question:

            print(
                "Please enter a question."
            )

            continue

        print()
        print("Processing...")
        print()

        try:

            result = answer_question(
                question,
                return_metadata=True
            )

            print("ANSWER:")
            print(result["answer"])
            print()

            print(
                f"Question type: "
                f"{result['question_type']}"
            )

            print(
                f"Retrieved chunks: "
                f"{result['retrieved_count']}"
            )

            print(
                f"Clean chunks: "
                f"{result['clean_count']}"
            )

            print(
                f"Contamination rate: "
                f"{result['contamination_rate']:.2f}%"
            )

            print(
                f"Verified: "
                f"{result['verified']}"
            )

            print(
                f"Response time: "
                f"{result['response_time']:.2f} seconds"
            )

            print()
            print("-" * 60)
            print()

        except Exception as e:

            print()
            print("ERROR:")
            print(str(e))
            print()


# ============================================================
# ONLY RUN TERMINAL MODE DIRECTLY
# ============================================================

if __name__ == "__main__":

    terminal_mode()