# dkuchat-rag-pipeline
A bilingual (English & Chinese) agentic RAG pipeline for question answering over DKU documents, built with DSPy, FAISS, BM25, and a locally hosted LLM.

---

## folder and file Structure

```
dkuchat-rag-pipeline/
├── data/                          # Source documents (PDF, DOCX)
├── index/                         # FAISS vector index + BM25 index + chunks
├── ingest/
│   └── ingest.py                  # chunking process
├── agent/
│   └── agent.py                   # DSPy RAG agent
├── tools/
│   ├── vector_search.py           # FAISS through vector search
│   ├── keyword_search.py          # BM25 keyword search
│   └── internet_search.py        # DuckDuckGo internet search
├── llm_server/
│   └── start_server.sh            # Script to start local LLM server
├── ui/
│   └── app.py                     # Gradio web interface
├── .env                           # Environment variables (not committed)
├── .env.example                   # Example environment variables
├── requirements.txt               # Python dependencies
└── README.md
```

---


## Setup Instructions

## 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/dkuchat-rag-pipeline.git
cd dkuchat-rag-pipeline
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Install and start Ollama
Download from **ollama.com/download** then:
```bash
ollama pull qwen2.5:1.5b
ollama serve
```

## 4. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env` with your settings.

## 5. Run ingestion (build the index)
```bash
python ingest/ingest.py
```

## 6. Start the application
```bash
# Terminal 1 — LLM server
bash llm_server/start_server.sh

# Terminal 2 — Web UI
python ui/app.py --model qwen2.5:1.5b
```

---

##  Dependencies

Install all dependencies with:
```bash
pip install -r requirements.txt
```

Key packages:
- `dspy-ai` — agentic LLM pipeline framework
- `faiss-cpu` — vector similarity search
- `rank-bm25` — BM25 keyword search
- `sentence-transformers` — embedding model
- `pdfplumber` — PDF text extraction
- `python-docx` — DOCX text extraction
- `gradio` — web UI
- `duckduckgo-search` — internet search
- `python-dotenv` — environment variable loading

---

## LLM Models Tested

| `Qwen/Qwen2.5-1.5B-Instruct` | Ollama --> Fast, works well on CPU/Mac |
| `Qwen/Qwen2.5-1.5B-Instruct` | MLX --> As it was apple Silicon there was no difficulty |
| `Qwen/Qwen3-8B` | Ollama / MLX --> Better quality, slower |

> **To note!** vLLM 0.19.1 had shared memory broadcast bug on macOS (POSIX shm limit). Ollama is used as the local LLM server after instructor approval.

---

## Environment Variables

Create a `.env` file in the project root:

```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:1.5b
OPENAI_API_KEY=not-needed
```

---


## Embedding Models Tested
| `BAAI/bge-small-en-v1.5` | 384 dimension| Fast, good quality |


## Notes
- The index must be rebuilt (`python ingest/ingest.py`) if documents change
- First query after server start is slower due to model warmup