from retriever import retrieve
from intent_detector import detect_intent
from contamination_detector import filter_context
from reranker import rerank
from context_verifier import verify_context
from llm_generator import generate_llm_answer


print("=" * 60)
print("RAG QUESTION ANSWERING SYSTEM")
print("=" * 60)

print("\nType 'exit' to stop the program.")


while True:

    # ========================================================
    # QUESTION
    # ========================================================

    question = input(
        "\nAsk your question: "
    ).strip()


    if question.lower() in ["exit", "quit", "q"]:

        print("\nRAG system stopped.")
        print("\nThank you for using the RAG system.")

        break


    if not question:

        print("\nPlease enter a question.")

        continue


    # ========================================================
    # INTENT
    # ========================================================

    intent = detect_intent(question)

    print(
        "\nDetected Intent:",
        intent
    )


    # ========================================================
    # RETRIEVAL
    # ========================================================

    results = retrieve(
        question,
        intent,
        top_k=15
    )


    if not results:

        print(
            "\nAnswer: The answer is not available "
            "in the document."
        )

        continue


    # ========================================================
    # CONTAMINATION DETECTION
    # ========================================================

    clean_results, contamination_score = filter_context(
        question,
        results
    )


    print(
        "\nContamination Score:",
        contamination_score,
        "%"
    )


    if not clean_results:

        print(
            "\nAnswer: The answer is not available "
            "in the document."
        )

        continue


    # ========================================================
    # RE-RANKING
    # ========================================================

    reranked_results = rerank(
        question,
        clean_results,
        intent,
        top_k=8
    )


    if not reranked_results:

        print(
            "\nAnswer: The answer is not available "
            "in the document."
        )

        continue


    # ========================================================
    # BEST SCORE
    # ========================================================

    best_score = reranked_results[0].get(
        "rerank_score",
        -999
    )


    print(
        "\nBest Rerank Score:",
        round(best_score, 4)
    )


    # ========================================================
    # VERY LOW RELEVANCE
    # ========================================================

    if best_score < -5:

        print(
            "\nContext Verification: NOT RELEVANT"
        )

        print(
            "\nAnswer: The answer is not available "
            "in the document."
        )

        continue


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context_parts = []


    for result in reranked_results:

        text = result.get(
            "text",
            ""
        ).strip()


        if text:

            context_parts.append(text)


    context = "\n\n".join(
        context_parts
    )


    if not context:

        print(
            "\nAnswer: The answer is not available "
            "in the document."
        )

        continue


    # ========================================================
    # CONTEXT VERIFICATION
    # ========================================================

    is_relevant = verify_context(
        question,
        context
    )


    if is_relevant:

        print(
            "\nContext Verification: RELEVANT"
        )

    else:

        print(
            "\nContext Verification: NOT RELEVANT"
        )


    # ========================================================
    # REJECT ONLY IF BOTH CHECKS FAIL
    # ========================================================

    if not is_relevant and best_score < 0:

        print(
            "\nAnswer: The answer is not available "
            "in the document."
        )

        continue


    # ========================================================
    # LLM ANSWER
    # ========================================================

    answer = generate_llm_answer(
        question,
        context
    )


    # ========================================================
    # FINAL ANSWER
    # ========================================================

    print(
        "\nAnswer:",
        answer
    )