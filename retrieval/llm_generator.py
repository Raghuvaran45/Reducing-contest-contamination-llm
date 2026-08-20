from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# ============================================================
# LOAD LLM
# ============================================================

MODEL_NAME = "google/flan-t5-base"

print("Loading LLM tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading LLM model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

model.eval()

print("LLM loaded successfully!")


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_llm_answer(question, context):

    if not context or not context.strip():

        return "The answer is not available in the document."


    # --------------------------------------------------------
    # LIMIT CONTEXT SIZE
    # --------------------------------------------------------

    context = context.strip()

    # Keep enough context for detailed answers
    if len(context) > 12000:
        context = context[:12000]


    # ========================================================
    # STRICT RAG PROMPT
    # ========================================================

    prompt = f"""
You are a document question-answering assistant.

You MUST answer the question using ONLY the information
provided in the DOCUMENT CONTEXT.

Do NOT use your own knowledge.
Do NOT invent information.
Do NOT guess.
Do NOT use information that is not present in the context.

If the answer cannot be found in the DOCUMENT CONTEXT,
respond exactly:

The answer is not available in the document.

When the answer is available:
- Give the complete answer.
- Preserve important technical details.
- Do not stop in the middle of a sentence.
- Do not repeat the same information.
- For definition questions, give the definition clearly.
- For method questions, list the relevant methods.
- For summary questions, combine the important information
  from the provided context.
- Use simple and clear language.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


    # ========================================================
    # TOKENIZE
    # ========================================================

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )


    # ========================================================
    # GENERATE
    # ========================================================

    with torch.no_grad():

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],

            # Larger value prevents short/truncated answers
            max_new_tokens=300,

            # Prevent extremely short answers
            min_new_tokens=10,

            # Better controlled generation
            num_beams=4,

            # Avoid repeating phrases
            no_repeat_ngram_size=3,

            # Stop when EOS token is produced
            early_stopping=True
        )


    # ========================================================
    # DECODE
    # ========================================================

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()


    # ========================================================
    # CLEAN ANSWER
    # ========================================================

    if not answer:

        return "The answer is not available in the document."


    # Remove accidental prompt-like outputs
    if answer.lower().startswith("answer:"):

        answer = answer[7:].strip()


    return answer