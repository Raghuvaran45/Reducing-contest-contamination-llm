
# Reducing Context Contamination in LLMs

## 📌 Project Overview

Large Language Models (LLMs) can give incorrect or less relevant answers when they receive too much information, especially when the retrieved context contains unrelated or unnecessary content.

This project focuses on **reducing context contamination in Large Language Models** by improving the way relevant information is retrieved and filtered before it is given to the language model.

The system uses a **Retrieval-Augmented Generation (RAG)** approach. Instead of sending an entire document to the language model, the document is divided into smaller chunks, relevant chunks are retrieved for a question, and unnecessary or potentially distracting context is filtered before generating the final answer.

The main goal is simple:

> **Retrieve the right information, remove unnecessary context, and generate a more reliable answer.**

---

## 🎯 Objectives

The main objectives of this project are:

- Reduce irrelevant information from retrieved context.
- Improve the quality of context provided to an LLM.
- Compare answers generated before and after context filtering.
- Evaluate retrieval and generation separately.
- Measure the effect of context filtering using multiple evaluation metrics.
- Support different types of documents and questions.
- Provide an easy-to-use Streamlit interface.
- Allow users to upload and analyze their own PDF documents.

---

## 🧠 How the System Works

The overall workflow of the system is:

```text
                PDF Document
                     │
                     ▼
              Text Extraction
                     │
                     ▼
               Text Chunking
                     │
                     ▼
          Sentence Transformer
             Embeddings
                     │
                     ▼
                  FAISS
             Vector Retrieval
                     │
                     ▼
             Relevant Context
                     │
            ┌────────┴────────┐
            │                 │
            ▼                 ▼
     Before Filtering   After Filtering
            │                 │
            └────────┬────────┘
                     ▼
                LLM Generation
                     │
                     ▼
              Answer Generation
                     │
                     ▼
            Evaluation & Metrics
```

The system first retrieves potentially useful chunks from the PDF. It then applies context filtering to identify and retain the information that is most useful for answering the question.

The answers produced using the original retrieved context and the filtered context are then compared.

---

## 🔍 Context Contamination

Context contamination happens when irrelevant, redundant, misleading, or weakly related information is included in the context provided to an LLM.

For example, suppose a PDF contains information about:

- Diabetes
- Heart disease
- Kidney disease
- Treatment recommendations
- Medical history

If the user asks:

> "What are the precautions for diabetes?"

the system should mainly provide the information related to diabetes and precautions.

Sending large amounts of unrelated information can make it harder for the LLM to identify the important evidence.

This project attempts to reduce that problem through adaptive context filtering.

---

## 📚 Retrieval-Augmented Generation (RAG)

The project follows a RAG-based architecture.

The PDF is first converted into text and divided into manageable chunks. Each chunk is converted into an embedding using a Sentence Transformer model.

FAISS is then used to efficiently search for chunks that are semantically related to the user's question.

The retrieved chunks are passed through the context filtering stage before the final answer is generated.

This approach allows the system to work with information contained in the uploaded document instead of relying only on the model's internal knowledge.

---

## 🧹 Context Filtering

One of the main parts of this project is the context filtering stage.

The system evaluates the retrieved information and tries to remove unnecessary context while preserving useful evidence.

The filtering process is designed to be adaptive rather than simply keeping a fixed number of chunks.

The system also considers whether the filtered context still contains enough information to answer the question. If filtering removes important evidence, the system can retain or restore the necessary context.

This helps avoid a situation where filtering improves the amount of removed information but makes the final answer worse.

---

## 📊 Evaluation

The project evaluates both **retrieval quality** and **answer generation quality**.

### Retrieval Metrics

The following metrics are used for retrieval evaluation:

- Accuracy
- Precision
- Recall
- F1-score

The retrieval evaluation focuses on whether useful evidence is successfully retrieved rather than treating every non-retrieved chunk as a negative example.

### Generation Metrics

The generated answers are evaluated using:

- Accuracy
- BLEU-4
- ROUGE-1 F1
- ROUGE-2 F1
- ROUGE-L F1
- Perplexity

The system compares the results **before filtering** and **after filtering**.

---

## 📈 Before vs After Filtering

The application provides a combined comparison of the results.

The evaluation table contains:

| Metric | Retrieval Before | Retrieval After | Retrieval Improvement | Generation Before | Generation After | Generation Improvement |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | | | | | | |
| Precision | | | | | | |
| Recall | | | | | | |
| F1-score | | | | | | |
| BLEU-4 | | | | | | |
| ROUGE-1 F1 | | | | | | |
| ROUGE-2 F1 | | | | | | |
| ROUGE-L F1 | | | | | | |
| Perplexity | | | | | | |

For most metrics, a higher value indicates better performance.

For **Perplexity**, a lower value is considered better because it indicates that the generated text is more predictable according to the evaluation model.

---

## 📄 PDF Support

The application allows users to upload their own PDF documents.

The system dynamically processes the uploaded document instead of requiring a fixed document.

This makes the application useful for different types of documents and questions.

The same core RAG pipeline is used across the different case-study categories.

---

## 🏥 Medical Case Study

The application provides a medical-oriented workflow for documents containing medical information.

Depending on the information available in the document, the system can identify:

- Disease or condition
- Diagnosis or findings
- Precautions
- Recommendations

The system is designed to answer based on the uploaded document.

It does **not** independently diagnose a patient or replace professional medical advice.

---

## ⚖️ Legal Case Study

The application also supports legal documents.

For legal questions, the system can extract information such as:

- IPC sections
- Case facts
- Allegations
- Findings
- Punishment or sentence mentioned in the document

The system attempts to preserve the terminology used in the source document.

It does not independently determine a punishment simply from an IPC section or invent legal facts that are not supported by the document.

---

## 🖥️ Streamlit Application

The project includes a Streamlit-based frontend.

The user can:

1. Select a case-study category.
2. Upload a PDF.
3. Process the document.
4. Enter a question.
5. Retrieve relevant information.
6. Filter unnecessary context.
7. Generate an answer.
8. Compare before and after results.
9. View evaluation metrics.

The interface is designed to make the complete RAG pipeline easier to use without requiring users to interact directly with the Python code.

---

## 💻 Terminal Mode

The retrieval pipeline can also be used directly from the terminal.

The terminal workflow allows the user to:

- Select the case-study category.
- Provide a PDF.
- Process the document.
- Ask questions.
- View retrieval information.
- Evaluate results.

This is useful for testing and debugging the RAG pipeline independently from the Streamlit frontend.

---

## 🛠️ Technologies Used

The main technologies used in this project include:

- **Python**
- **Streamlit**
- **FAISS**
- **Sentence Transformers**
- **PyMuPDF**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **NLTK**
- **ROUGE**
- **Transformers**
- **PyTorch**
- **Large Language Model API**

### Embedding Model

The project uses:

```text
BAAI/bge-small-en-v1.5
```

for generating semantic embeddings for document chunks and user questions.

### Vector Database

**FAISS** is used for efficient similarity search over the document embeddings.

---

## 📁 Project Structure

```text
Reducing-contest-contamination-llm/
│
├── app.py
├── requirements.txt
├── README.md
├── evaluation.csv
│
├── retrieval/
│   ├── __init__.py
│   └── retrieval_pipeline.py
│
├── medical/
│   ├── __init__.py
│   └── medical_analysis.py
│
└── legal/
    ├── __init__.py
    └── legal_analysis.py
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

### 2. Move into the project directory

```bash
cd Reducing-contest-contamination-llm
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install the dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 API Configuration

The project uses an external LLM API for answer generation.

The API key should **never be directly written inside the source code or uploaded to GitHub**.

For local development, configure the required API key as an environment variable.

For Streamlit Cloud, add the API key through the application's **Secrets** settings.

Example:

```toml
API_KEY = "your-api-key"
```

Use the exact environment variable required by the selected LLM provider.

---

## ▶️ Running the Streamlit Application

After installing the dependencies, run:

```bash
streamlit run app.py
```

The application will open in the browser.

---

## ▶️ Running the Retrieval Pipeline

The retrieval pipeline can also be executed directly:

```bash
python retrieval/retrieval_pipeline.py
```

or, depending on the project directory:

```bash
python -m retrieval.retrieval_pipeline
```

---

## ☁️ Streamlit Cloud Deployment

The project can be deployed using Streamlit Cloud.

Basic deployment steps:

1. Push the project to GitHub.
2. Open Streamlit Cloud.
3. Create a new application.
4. Select the GitHub repository.
5. Select `app.py` as the main file.
6. Deploy the application.
7. Add required API keys through Streamlit Secrets.
8. Reboot the application after configuration changes.

The dependencies are installed automatically from:

```text
requirements.txt
```

---

## 🔒 Security

API keys and other sensitive credentials should not be committed to GitHub.

Do not write credentials directly into:

```text
app.py
retrieval_pipeline.py
README.md
requirements.txt
```

Instead, use environment variables or Streamlit Secrets.

---

## 🌟 Key Features

### Adaptive Retrieval

The system retrieves information based on the user's question instead of processing the complete document every time.

### Context Filtering

Irrelevant or weakly related context is reduced before generation.

### Before/After Comparison

The system compares performance before and after filtering.

### Multiple Evaluation Metrics

Several retrieval and generation metrics are used instead of depending on a single score.

### PDF Upload

Users can upload different PDF documents and process them dynamically.

### Multiple Case Studies

The same RAG architecture can be used for:

- General documents
- Medical documents
- Legal documents

### Interactive Frontend

The complete workflow is available through a Streamlit interface.

---

## 🎓 Project Motivation

As LLMs become more widely used with external documents, simply retrieving more information does not always lead to better answers.

Providing an LLM with too much irrelevant information can make it harder to identify the evidence that actually matters.

This project explores whether **better context selection can improve the reliability and quality of LLM-generated answers**.

Rather than focusing only on retrieving more information, the project focuses on retrieving **useful information** and reducing unnecessary context.

---

## 🚀 Future Improvements

Some possible future improvements include:

- Support for more embedding models.
- Support for additional LLM providers.
- More advanced semantic filtering.
- Better document-level evaluation datasets.
- Improved multilingual document support.
- More detailed visualization of retrieval results.
- Larger-scale benchmarking across different document types.
- More advanced hallucination detection.
- Improved evaluation of long-form answers.

---

## 👨‍💻 Conclusion

**Reducing Context Contamination in LLMs** is a RAG-based project that explores how filtering unnecessary information can affect the quality of retrieval and LLM-generated answers.

The project combines document processing, semantic search, context filtering, LLM generation, evaluation metrics, and an interactive Streamlit frontend into a single system.

The main idea behind the project is:

> **Better context can lead to better answers.**

Instead of simply giving an LLM more information, this project focuses on giving it the **right information at the right time**.
