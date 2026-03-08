# Legal RAG Assistant

A Retrieval-Augmented Generation (RAG) powered legal document assistant that lets you upload PDF legal documents and ask natural language questions about them. Backed by an AutoGen agent, a FAISS vector store, and the LLaMA 3.3 70B model served via Groq.

---

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Demo

> Upload any legal PDF (e.g. a rental agreement, NDA, or contract) through the web UI and ask questions like:
> - *"Can the landlord increase rent during the lease term?"*
> - *"What are the termination clauses?"*
> - *"Summarise the obligations of the tenant."*

---

## Features

- **PDF ingestion** — Upload any legal PDF directly from the browser.
- **Semantic search** — Documents are chunked, embedded, and stored in a FAISS vector store for fast similarity search.
- **Agentic Q&A** — An AutoGen `AssistantAgent` decides when to call the retrieval tool and synthesises a grounded answer.
- **Conversational web UI** — Clean chat interface served by Flask.
- **CLI mode** — Run queries directly from the terminal without the web server.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | LLaMA 3.3 70B (via [Groq](https://groq.com)) |
| Agent framework | [AutoGen](https://github.com/microsoft/autogen) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace / sentence-transformers) |
| Vector store | FAISS (CPU) |
| Document parsing | PyMuPDF (`fitz`) |
| Orchestration | LangChain (text splitter, document loaders) |
| Web framework | Flask |

---

## How It Works

```
User uploads PDF
      │
      ▼
PyMuPDF extracts text
      │
      ▼
RecursiveCharacterTextSplitter chunks the text (1000 chars, 200 overlap)
      │
      ▼
HuggingFace embeddings (all-MiniLM-L6-v2) encode each chunk
      │
      ▼
FAISS index persisted to ./rag_faiss_store/
      │
User asks a question
      │
      ▼
AutoGen AssistantAgent calls retrieve_legal_context tool
      │
      ▼
Top-3 semantically similar chunks retrieved from FAISS
      │
      ▼
LLaMA 3.3 70B (Groq) synthesises a grounded answer
      │
      ▼
Answer returned to the user
```

---

## Project Structure

```
Legal Rag Assistant/
├── app.py                  # Flask app + AutoGen agent (web interface)
├── mainchat.py             # AutoGen agent (CLI interface)
├── rag_index_builder.py    # PDF ingestion & FAISS index builder
├── tools.py                # RAG retrieval tool used by the agent
├── main.py                 # Entry point placeholder
├── pyproject.toml          # Project metadata & dependencies
├── requirements.txt        # pip-compatible dependency list
├── templates/
│   └── index.html          # Chat web UI
├── docs/                   # Place sample PDFs here
├── rag_faiss_store/        # Persisted FAISS index (auto-generated)
│   └── index.faiss
└── README.md
```

---

## Prerequisites

- Python 3.12 or higher
- A free [Groq API key](https://console.groq.com)
- `pip` or `uv` for package management

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/legal-rag-assistant.git
cd legal-rag-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Or using uv (faster):

```bash
uv sync
```

---

## Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> **Never commit your `.env` file.** Add it to `.gitignore`:
> ```
> .env
> rag_faiss_store/
> __pycache__/
> .venv/
> ```

---

## Usage

### Web Application

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

1. Upload a legal PDF using the upload button.
2. Wait for the index to build (confirmation shown in UI).
3. Type your question in the chat box and press **Send**.

### CLI (no web server)

Build the index first:

```bash
python rag_index_builder.py
```

Then run a query interactively via `mainchat.py`, or modify `tools.py` directly:

```bash
python tools.py
```

## License

This project is licensed under the [MIT License](LICENSE).
