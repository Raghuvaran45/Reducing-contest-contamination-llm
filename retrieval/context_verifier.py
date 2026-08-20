from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-base"

print("Loading context verification LLM...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Context verification LLM loaded successfully!")


def verify_context(question, context):

    prompt = f"""
Question: {question}

Context from the PDF:
{context}

Does this context contain information that can answer the question?

Answer only:
YES
or
NO
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=3,
        do_sample=False
    )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip().upper()

    print("Verification Result:", result)

    if "YES" in result:
        return True

    if "NO" in result:
        return False

    # Don't reject useful retrieved context
    return True