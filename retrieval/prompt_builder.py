def build_prompt(question, context):
    
    prompt = f"""
Answer the question using only the information in the context.

Context:
{context}

Question:
{question}

Give a short, factual answer.
Do not say "Model" unless the context clearly states that as the answer.
If the context does not contain the answer, say:
The answer is not available in the document.

Answer:
"""

    return prompt