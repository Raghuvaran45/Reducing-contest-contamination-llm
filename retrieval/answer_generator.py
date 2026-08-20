import re


def generate_answer(question, results):

    if not results:
        return "I could not find relevant information in the document."


    # Use the best retrieved chunk
    context = results[0]["text"].strip()


    # Split into sentences
    sentences = re.split(
        r'(?<=[.!?])\s+',
        context
    )


    question_words = set(
        re.findall(
            r'\b[a-zA-Z]{3,}\b',
            question.lower()
        )
    )


    best_sentence = ""
    best_score = -1


    for sentence in sentences:

        sentence_words = set(
            re.findall(
                r'\b[a-zA-Z]{3,}\b',
                sentence.lower()
            )
        )

        score = len(
            question_words.intersection(
                sentence_words
            )
        )


        if score > best_score:

            best_score = score
            best_sentence = sentence.strip()


    if best_sentence:

        return best_sentence

    return context[:300]