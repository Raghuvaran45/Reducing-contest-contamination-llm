import os
from google import genai


api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY is not set.")
    exit()


client = genai.Client(
    api_key=api_key
)


response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain RAG in two simple sentences."
)


print("\nGemini response:")
print(response.text)