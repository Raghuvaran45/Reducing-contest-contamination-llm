from sentence_transformers import CrossEncoder


print("Loading reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker loaded successfully!")


def rerank(
    question,
    results,
    intent="general",
    top_k=5
):

    if not results:
        return []

    pairs = []

    for result in results:

        pairs.append([
            question,
            result.get("text", "")
        ])

    scores = reranker.predict(
        pairs
    )

    for result, score in zip(
        results,
        scores
    ):

        result["rerank_score"] = float(score)

    results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return results[:top_k]