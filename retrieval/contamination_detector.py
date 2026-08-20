from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

import re


# ============================================================
# LOAD LLM
# ============================================================

MODEL_NAME = "google/flan-t5-base"

print("Loading contamination detection LLM...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print("Contamination detector loaded successfully!")


# ============================================================
# LLM CONTAMINATION CHECK
# ============================================================

def llm_check_chunk(question, chunk_text):

    prompt = f"""
Classify whether the passage is relevant to answering
the question.

Question:
{question}

Passage:
{chunk_text}

Reply with exactly one word:

RELEVANT

or

IRRELEVANT
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip().upper()

    if "RELEVANT" in answer and "IRRELEVANT" not in answer:
        return "relevant"

    return "irrelevant"


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def find_duplicates(results):

    seen = set()

    duplicate_indexes = set()

    for i, result in enumerate(results):

        text = normalize_text(
            result.get("text", "")
        )

        if not text:
            continue

        if text in seen:

            duplicate_indexes.add(i)

        else:

            seen.add(text)

    return duplicate_indexes


# ============================================================
# PROMPT INJECTION DETECTION
# ============================================================

def contains_prompt_injection(text):

    injection_patterns = [

        "ignore previous instructions",

        "ignore all previous instructions",

        "ignore the instructions",

        "system prompt",

        "you are chatgpt",

        "you are an ai assistant",

        "follow these instructions",

        "disregard previous",

        "disregard the instructions",

        "forget previous instructions"
    ]

    text = text.lower()

    for pattern in injection_patterns:

        if pattern in text:

            return True

    return False


# ============================================================
# MAIN CONTAMINATION FILTER
# ============================================================

def filter_context(
    question,
    results
):

    if not results:

        return [], 100.0


    total = len(results)

    contaminated = 0

    clean_results = []


    # ========================================================
    # DUPLICATE CHECK
    # ========================================================

    duplicate_indexes = find_duplicates(
        results
    )


    # ========================================================
    # CHECK EACH CHUNK
    # ========================================================

    for i, result in enumerate(results):

        text = result.get(
            "text",
            ""
        ).strip()


        # ----------------------------------------------------
        # Empty chunk
        # ----------------------------------------------------

        if not text:

            contaminated += 1

            continue


        # ----------------------------------------------------
        # Duplicate
        # ----------------------------------------------------

        if i in duplicate_indexes:

            contaminated += 1

            continue


        # ----------------------------------------------------
        # Prompt injection
        # ----------------------------------------------------

        if contains_prompt_injection(text):

            contaminated += 1

            continue


        # ----------------------------------------------------
        # LLM relevance check
        # ----------------------------------------------------

        relevance = llm_check_chunk(
            question,
            text
        )


        # ----------------------------------------------------
        # Irrelevant
        # ----------------------------------------------------

        if relevance == "irrelevant":

            contaminated += 1

            continue


        # ----------------------------------------------------
        # Clean chunk
        # ----------------------------------------------------

        result["contamination"] = "clean"

        clean_results.append(
            result
        )


    # ========================================================
    # CONTAMINATION SCORE
    # ========================================================

    contamination_score = (
        contaminated / total
    ) * 100


    contamination_score = round(
        contamination_score,
        2
    )


    return (
        clean_results,
        contamination_score
    )