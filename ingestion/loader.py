import os
import fitz  # PyMuPDF


def load_pdf(file_path):
    """
    Extract text from a PDF.
    """
    document = fitz.open(file_path)

    text = ""

    for page in document:
        text += page.get_text()

    return text


def load_all_pdfs(folder_path):
    """
    Load all PDFs from a folder.
    """
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(folder_path, file)

            text = load_pdf(full_path)

            documents.append({
                "filename": file,
                "text": text
            })

    return documents