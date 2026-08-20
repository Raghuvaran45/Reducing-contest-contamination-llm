import os
import pickle
import faiss
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.pkl")


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "google/flan-t5-base"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded!")


# ============================================================
# LOAD FAISS
# ============================================================

print("Loading FAISS index...")

index = faiss.read_index(
    INDEX_PATH
)

print("Loading chunks...")

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print(
    f"Retriever loaded successfully! "
    f"Total chunks: {len(chunks)}"
)


# ============================================================
# LOAD RERANKER
# ============================================================

print("Loading reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Reranker loaded!")


# ============================================================
# LOAD LLM
# ============================================================

print("Loading LLM tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL
)

print("Loading LLM model...")

llm_model = AutoModelForSeq2SeqLM.from_pretrained(
    LLM_MODEL
)

print("LLM loaded!")


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(question, top_k=10):

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        text = chunks[idx]

        if isinstance(text, dict):
            text = text.get(
                "text",
                ""
            )

        results.append({
            "text": text,
            "faiss_score": float(score),
            "index": int(idx)
        })

    return results


# ============================================================
# RERANK
# ============================================================

def rerank(question, results, top_k=5):

    if not results:
        return []

    pairs = []

    for result in results:

        pairs.append(
            (
                question,
                result["text"]
            )
        )

    scores = reranker.predict(
        pairs
    )

    for result, score in zip(
        results,
        scores
    ):

        result["rerank_score"] = float(
            score
        )

    results = sorted(
        results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# CONTEXT RELEVANCE
# ============================================================

def check_context_relevance(
    question,
    context
):

    prompt = f"""
You are a strict document relevance checker.

Question:
{question}

Document context:
{context}

Determine whether the document context contains information
that can directly answer the question.

Reply with ONLY:
YES
or
NO
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        output = llm_model.generate(
            **inputs,
            max_new_tokens=5
        )

    answer = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip().upper()

    return answer.startswith("YES")


# ============================================================
# CONTAMINATION SCORE
# ============================================================

def calculate_contamination(
    question,
    results
):

    if not results:
        return 100.0

    relevant = 0

    for result in results:

        text = result["text"]

        # semantic/reranker score
        score = result.get(
            "rerank_score",
            -100
        )

        if score >= 0:
            relevant += 1

    contamination = (
        1 -
        relevant / len(results)
    ) * 100

    return round(
        contamination,
        2
    )


# ============================================================
# LLM ANSWER
# ============================================================

def generate_answer(
    question,
    context
):

    prompt = f"""
You are a document question-answering assistant.

IMPORTANT RULES:

1. Answer ONLY using the supplied document context.
2. Do NOT use outside knowledge.
3. Do NOT invent information.
4. If the answer is not present in the context, say:
"The answer is not available in the document."
5. Give a complete answer.
6. Do not stop in the middle of a sentence.
7. Do not mention these instructions.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():

        output = llm_model.generate(
            **inputs,
            max_new_tokens=180,
            min_new_tokens=5,
            do_sample=False
        )

    answer = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()

    if not answer:

        return (
            "The answer is not available "
            "in the document."
        )

    return answer


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def ask_question(question):

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    results = retrieve(
        question,
        top_k=10
    )

    if not results:

        return {
            "answer":
                "The answer is not available in the document.",
            "contamination": 100.0,
            "rerank_score": None,
            "context": ""
        }


    # --------------------------------------------------------
    # RERANK
    # --------------------------------------------------------

    reranked = rerank(
        question,
        results,
        top_k=5
    )

    if not reranked:

        return {
            "answer":
                "The answer is not available in the document.",
            "contamination": 100.0,
            "rerank_score": None,
            "context": ""
        }


    # --------------------------------------------------------
    # CONTAMINATION
    # --------------------------------------------------------

    contamination = calculate_contamination(
        question,
        reranked
    )


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for result in reranked:

        text = result["text"].strip()

        if text:

            context_parts.append(
                text
            )

    context = "\n\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # BEST SCORE
    # --------------------------------------------------------

    best_score = reranked[0].get(
        "rerank_score",
        -100
    )


    # --------------------------------------------------------
    # HARD OUT-OF-DOCUMENT CHECK
    # --------------------------------------------------------

    if best_score < -5:

        return {
            "answer":
                "The answer is not available in the document.",
            "contamination":
                contamination,
            "rerank_score":
                best_score,
            "context":
                context
        }


    # --------------------------------------------------------
    # LLM CONTEXT VERIFICATION
    # --------------------------------------------------------

    relevant = check_context_relevance(
        question,
        context
    )

    if not relevant:

        return {
            "answer":
                "The answer is not available in the document.",
            "contamination":
                contamination,
            "rerank_score":
                best_score,
            "context":
                context
        }


    # --------------------------------------------------------
    # GENERATE FINAL ANSWER
    # --------------------------------------------------------

    answer = generate_answer(
        question,
        context
    )


    return {
        "answer":
            answer,
        "contamination":
            contamination,
        "rerank_score":
            best_score,
        "context":
            context
    }