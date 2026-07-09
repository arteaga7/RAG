# RAG
A RAG (Retrieval-Augmented Generation) sysmtem using:

- LLM: `llama-3.3-70b-versatile` via `GROQ_API_KEY`
- Embeddings: `nomic-embed-text` via Ollama (locally)
- Vector store: `ChromaDB`

Requirements: **GROQ API KEY** needed.

## 🌎 Repository Structure
```bash
RAG/
├── app.py
├── .gitignore
├── env/
└── requirements.txt
└── book_embedder.py        # To convert data to emdeddings
└── books/                  # Paste your documents `.txt`, `.md` or `.pdf` here
└── templates/              # Needed for Flask
```

## 🚀 How to run
1. Clone this repository:
```bash
git clone https://github.com/arteaga7/RAG.git
```
2. Set virtual environment and install dependencies.

For Windows:
```bash
python -m venv env
env/Scripts/activate
pip install -r requirements.txt
```
For Linux:
```bash
python -m venv env && source env/bin/activate && pip install -r requirements.txt
```
3. Download and install Ollama from: https://ollama.com/download.
4. Download the the embedder. In a terminal run:
```bash
ollama pull nomic-embed-text
```
5. Paste your documents `.txt`, `.md` or `.pdf` in "books/".
6. Build the embeddings. In the root of this project, run:
```bash
python book_embedder.py
```

7. Run "app.py":
```bash
python app.py
```
