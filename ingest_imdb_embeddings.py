import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import redis

# === Load environment variables ===
load_dotenv()

# === Constants ===
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CSV_PATH = os.getenv("IMDB_CSV_PATH", "./data/imdb_top_1000.csv")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# === Load SentenceTransformer model once ===
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# === Connect to Redis ===
redis_client = redis.Redis.from_url(REDIS_URL)


def generate_description(row: pd.Series) -> str:
    """
    Create a natural language sentence describing the movie.
    This text will be used to generate the embedding.
    """
    title = row.get("Series_Title", "").strip()
    genre = row.get("Genre", "").strip()
    year = row.get("Released_Year", "").strip()
    director = row.get("Director", "").strip()
    overview = row.get("Overview", "").strip()

    return f"{title} is a {genre} movie released in {year}, directed by {director}. {overview}"


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate a 384-dimensional embedding for the given input text.
    """
    return model.encode([text])[0]


def push_to_redis_vss(title: str, vector: np.ndarray, attributes: Dict):
    """
    Push an embedding and its attributes to Redis Vector Set.
    """
    try:
        redis_client.execute_command("VADD", "movies", "VALUES", 384, *vector.tolist(), f"id:{title}")
        redis_client.execute_command("VSETATTR", "movies", f"id:{title}", json.dumps(attributes))
    except Exception as e:
        print(f"[ERROR] Failed to push {title} to Redis: {e}")


def ingest_csv(csv_path: str) -> List[Tuple[str, np.ndarray]]:
    """
    Loop through the CSV file and push each movie embedding to Redis.
    """
    df = pd.read_csv(csv_path)
    results = []

    for idx, row in df.iterrows():
        try:
            description = generate_description(row)
            embedding = generate_embedding(description)
            title = row.get("Series_Title", f"movie_{idx}").strip()

            # Optional: attributes to store as metadata
            attributes = {
                "title": title,
                "genre": row.get("Genre", "").strip(),
                "year": int(row.get("Released_Year", "0")),
                "IMDB_rating": float(row.get("IMDB_Rating", "0.0"))
            }

            push_to_redis_vss(title, embedding, attributes)
            results.append((title, embedding))
        except Exception as e:
            print(f"[WARN] Failed to process row {idx}: {e}")
            continue

    return results


if __name__ == "__main__":
    print(f"[INFO] Using model: {EMBEDDING_MODEL_NAME}")
    print(f"[INFO] Loading dataset from: {CSV_PATH}")
    print(f"[INFO] Connecting to Redis at: {REDIS_URL}")

    vectors = ingest_csv(CSV_PATH)

    print(f"[INFO] Processed {len(vectors)} movies.")
    if vectors:
        title, emb = vectors[0]
        print(f"[INFO] First example title: {title}")
        print(f"[INFO] Embedding shape: {emb.shape}")
        print(f"[INFO] Embedding preview: {emb[:10]} ...")