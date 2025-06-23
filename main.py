import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import redis

# === Load environment variables ===
load_dotenv()

# === Constants ===
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
VECTOR_SET_NAME = "movies"
VECTOR_DIM = 384
TOP_K = int(os.getenv("TOP_K", "5"))

# === Load model and Redis ===
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
redis_client = redis.Redis.from_url(REDIS_URL)


def embed_query(text: str) -> np.ndarray:
    """
    Embed the input query using SentenceTransformer model.
    """
    return model.encode([text])[0]


def query_redis_vss(query_vector: np.ndarray, top_k: int = 5):
    """
    Run a VSIM query against Redis Vector Set.
    """
    try:
        response = redis_client.execute_command(
            "VSIM", VECTOR_SET_NAME,
            "VALUES", VECTOR_DIM, *query_vector.tolist(),
            "WITHSCORES", "COUNT", top_k
        )

        print(f"\n[INFO] Top {top_k} similar items:\n")
        for i in range(0, len(response), 2):
            item_id = response[i]
            score = float(response[i + 1])
            print(f"• {item_id} (score: {score:.4f})")

    except Exception as e:
        print(f"[ERROR] VSIM query failed: {e}")


if __name__ == "__main__":
    print("[INFO] Semantic Search: IMDB Redis Vector Set\n")

    while True:
        query = input("Enter a movie description (or 'exit'): ").strip()
        if query.lower() in ("exit", "quit"):
            break

        vector = embed_query(query)
        query_redis_vss(vector, top_k=TOP_K)