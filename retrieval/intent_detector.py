def detect_intent(question):
    
    question = question.lower().strip()

    if any(word in question for word in [
        "title",
        "paper name",
        "document name",
        "name of paper"
    ]):
        return "metadata"

    elif any(word in question for word in [
        "author",
        "authors",
        "written by",
        "who wrote",
        "researchers"
    ]):
        return "author"

    elif any(word in question for word in [
        "summary",
        "summarize",
        "overview",
        "main idea"
    ]):
        return "summary"

    elif any(word in question for word in [
        "what is",
        "define",
        "definition",
        "explain",
        "meaning"
    ]):
        return "explanation"

    elif any(word in question for word in [
        "architecture",
        "model",
        "method",
        "approach",
        "algorithm"
    ]):
        return "technical"

    else:
        return "general"