import os

import json
import numpy as np
from typing import List
import redis
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# === Load environment ===
load_dotenv()

# === Config ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Must come before model load
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_SET_NAME = "movies"
VECTOR_DIM = 384
TOP_K = int(os.getenv("TOP_K", "5"))

# === Load clients ===
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
client = OpenAI(api_key=OPENAI_API_KEY)


def embed_query(text: str) -> np.ndarray:
    """
    Embed user query into a 384-dim vector using SentenceTransformer model.
    """
    return model.encode([text])[0]


def query_redis_vss(query_vector: np.ndarray, top_k: int = 5) -> List[dict]:
    try:
        response = redis_client.execute_command(
            "VSIM", VECTOR_SET_NAME,
            "VALUES", VECTOR_DIM, *query_vector.tolist(),
            "WITHSCORES", "COUNT", top_k
        )
        results = []
        print("\n[DEBUG] Raw Redis VSIM results:\n")
        for i in range(0, len(response), 2):
            item_id = response[i]
            score = float(response[i + 1])
            raw = redis_client.execute_command("VGETATTR", VECTOR_SET_NAME, item_id)
            metadata = json.loads(raw) if raw else {}
            metadata["id"] = item_id
            metadata["score"] = score
            results.append(metadata)
            print(json.dumps(metadata, indent=2))  # ✅ clean debug print

        return results
    except Exception as e:
        print(f"[ERROR] Redis VSIM failed: {e}")
        return []


def build_context_from_results(results: List[dict]) -> str:
    """
    Construct a natural language context block for LLM from Redis results.
    """
    lines = ["Here are some similar movies found via semantic vector search:"]
    for r in results:
        lines.append(
            f"- {r.get('title', r['id'])} ({r.get('year', '?')}) - Genre: {r.get('genre', 'Unknown')} [score: {r['score']:.4f}]"
        )
    return "\n".join(lines)


def ask_openai(system_msg: str, user_msg: str) -> str:
    """
    Use OpenAI Chat API to generate a reply with hybrid context.
    """
    try:
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] OpenAI API call failed: {e}"


if __name__ == "__main__":
    print("[INFO] Redis VSS + OpenAI RAG demo")

    while True:
        query = input("\nDescribe the type of movie you want (or 'exit'): ").strip()
        if query.lower() in ("exit", "quit"):
            break

        vector = embed_query(query)
        similar_movies = query_redis_vss(vector, top_k=TOP_K)
        context = build_context_from_results(similar_movies)

        system_prompt = (
            "You are an AI movie assistant using a hybrid retrieval system. "
            "You should answer using both the Redis vector search context provided and your internal knowledge. "
            "The Redis dataset includes only the top 1000 IMDB movies, so feel free to expand with other examples as needed."
        )

        final_query = f"{context}\n\nUser query: {query}"

        print("\n[INFO] Asking OpenAI with semantic context...")
        reply = ask_openai(system_prompt, final_query)

        print("\n[OPENAI REPLY]")
        print(reply)