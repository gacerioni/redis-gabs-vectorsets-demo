
# Redis IMDB Vector Search Demo

This project demonstrates how to use **Redis Vector Sets (VSS)** in Redis 8 (2025 preview) to perform semantic search on the IMDB Top 1000 movie dataset, using **sentence-transformers** for embeddings and optionally enriching responses with **OpenAI GPT-4** via hybrid RAG.

## 📦 Project Structure

- `ingest_imdb_embeddings.py`  
  Loads the IMDB dataset, generates 384-dimensional embeddings using HuggingFace's `all-MiniLM-L6-v2`, and stores them in Redis Vector Sets along with movie metadata (`title`, `genre`, `year`, `IMDB_rating`).

- `main.py`  
  CLI tool to run semantic search over the vector set. It takes user input, vectorizes it, and finds similar movies using Redis `VSIM`.

- `main_with_openai_llm_agent.py`  
  Extended search using OpenAI Chat API (`gpt-4o`, etc.) to answer movie-related questions with both Redis-based semantic context and LLM knowledge (hybrid RAG).

- `main_gradio_llm_ui.py`  
  (New!) A **Gradio web interface** that wraps the hybrid RAG logic, allowing users to interact with the system via a browser instead of the CLI.

## 🛠️ Requirements

- Python 3.12+
- Redis 8.0 (with vector sets enabled)
- Virtualenv recommended

### Install dependencies

```bash
pip install -r requirements.txt
```

## 🧪 Usage

### 1. Prepare your `.env` file:

```dotenv
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
IMDB_CSV_PATH=./data/imdb_top_1000.csv
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
TOP_K=5
```

### 2. Ingest IMDB data into Redis:

```bash
python ingest_imdb_embeddings.py
```

### 3. Run local semantic search:

```bash
python main.py
```

### 4. Run semantic search + OpenAI hybrid LLM:

```bash
python main_with_openai_llm_agent.py
```

### 5. Run semantic search + OpenAI hybrid LLM (Web UI):

```bash
python main_gradio_llm_ui.py
```

Ask things like:
- `crime movie with revenge`
- `epic fantasy adventure`
- `what is the godfather movies scores in imdb?`

## 🧠 What It Demonstrates

- Redis Vector Sets (`VADD`, `VSIM`, `VSETATTR`, `VGETATTR`)
- HuggingFace `sentence-transformers` for sentence embeddings
- Chat-based RAG pattern using OpenAI GPT
- Light hybrid search pipeline combining fast vector recall with LLM reasoning

---

Feel free to modify or extend this as a Redis hackathon demo, internal POC, or educational lab.
