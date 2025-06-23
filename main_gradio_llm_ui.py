import os
import json
import numpy as np
import redis
import gradio as gr
from dotenv import load_dotenv
from typing import List
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# === Load environment ===
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Config ===
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
VECTOR_SET_NAME = "movies"
VECTOR_DIM = 384
TOP_K = int(os.getenv("TOP_K", "5"))

# === Load clients ===
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def embed_query(text: str) -> np.ndarray:
    return model.encode([text])[0]


def query_redis_vss(query_vector: np.ndarray, top_k: int = 5) -> List[dict]:
    try:
        response = redis_client.execute_command(
            "VSIM", VECTOR_SET_NAME,
            "VALUES", VECTOR_DIM, *query_vector.tolist(),
            "WITHSCORES", "COUNT", top_k
        )
        results = []
        for i in range(0, len(response), 2):
            item_id = response[i]
            score = float(response[i + 1])
            raw = redis_client.execute_command("VGETATTR", VECTOR_SET_NAME, item_id)
            metadata = json.loads(raw) if raw else {}
            metadata["id"] = item_id
            metadata["score"] = score
            results.append(metadata)
            print("DEBUG: Found item:", item_id, "with score:", score, "and metadata:", metadata)
        return results
    except Exception as e:
        return [{"error": str(e)}]


def build_context_from_results(results: List[dict]) -> str:
    if not results or "error" in results[0]:
        return "No results from Redis."
    lines = ["Here are some similar movies found via semantic vector search:"]
    for r in results:
        lines.append(
            f"- {r.get('title', r['id'])} ({r.get('year', '?')}) - Genre: {r.get('genre', 'Unknown')} [score: {r['score']:.4f}]")
    return "\n".join(lines)


def stream_openai(system_msg: str, user_msg: str):
    try:
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        stream = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            stream=True,
        )
        result = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                result += delta
                yield result
    except Exception as e:
        yield f"[ERROR] OpenAI streaming failed: {e}"


def run_llm_pipeline(user_input: str):
    vector = embed_query(user_input)
    results = query_redis_vss(vector, top_k=TOP_K)
    context = build_context_from_results(results)

    system_prompt = (
        "You are an AI movie assistant using a hybrid retrieval system. "
        "You should answer using both the Redis vector search context provided and your internal knowledge. "
        "The Redis dataset includes only the top 1000 IMDB movies, so feel free to expand with other examples as needed."
    )
    final_query = f"{context}\n\nUser query: {user_input}"

    # 👇 make sure run_llm_pipeline is a generator by yielding each chunk
    for chunk in stream_openai(system_prompt, final_query):
        yield chunk


if __name__ == "__main__":
    demo = gr.Interface(
        fn=run_llm_pipeline,
        inputs=gr.Textbox(label="Describe the movie you want", lines=6),
        outputs=gr.Markdown(label="AI Recommendation"),
        title="🎬 Redis Vector Search + OpenAI RAG",
        description="Type a query to get AI-powered movie recommendations using Redis Vector Sets and OpenAI GPT.",
        flagging_mode="never"
    )
    demo.launch()