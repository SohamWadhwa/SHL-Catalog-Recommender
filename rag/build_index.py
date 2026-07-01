"""
Run once: python rag/build_index.py
Reads catalog/catalog.json, embeds every assessment, saves:
  - rag/index.faiss      (FAISS flat L2 index)
  - rag/metadata.pkl     (list of dicts, same order as index rows)
"""

import json
import pickle
import pathlib
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).parent.parent
CATALOG_PATH = ROOT / "catalog" / "shl_product_catalog.json"
INDEX_PATH   = pathlib.Path(__file__).parent / "index.faiss"
META_PATH    = pathlib.Path(__file__).parent / "metadata.pkl"

KEY_TO_TYPE = {
    "Knowledge & Skills":           "K",
    "Personality & Behavior":       "P",
    "Ability & Aptitude":           "A",
    "Simulations":                  "S",
    "Competencies":                 "C",
    "Biodata & Situational Judgment": "B",
    "Development & 360":            "D",
    "Assessment Exercises":         "E",
}

def build_document(item: dict) -> str:
    """
    Combine all useful fields into one string for embedding.
    More signal = better retrieval.
    """
    parts = [
        f"Name: {item['name']}",
        f"Description: {item.get('description', '')}",
        f"Test types: {', '.join(item.get('keys', []))}",
        f"Job levels: {', '.join(item.get('job_levels', []))}",
    ]
    if item.get("duration"):
        parts.append(f"Duration: {item['duration']}")
    if item.get("adaptive") == "yes":
        parts.append("Adaptive: yes")
    return " | ".join(parts)


def derive_test_type(keys: list[str]) -> str:
    """Return comma-separated type codes, e.g. 'K' or 'P,C'."""
    codes = [KEY_TO_TYPE.get(k, "K") for k in keys]
    seen = set()
    result = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return ",".join(result) if result else "K"

def main():
    # 1. Load catalog
    with open(CATALOG_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"Loaded {len(raw)} assessments from catalog.")

    # 2. Build clean metadata list (what the agent will return to users)
    metadata = []
    documents = []
    for item in raw:
        meta = {
            "name":       item["name"],
            "url":        item["link"],
            "test_type":  derive_test_type(item.get("keys", [])),
            "description": item.get("description", ""),
            "job_levels": item.get("job_levels", []),
            "duration":   item.get("duration", ""),
            "remote":     item.get("remote", "yes"),
            "adaptive":   item.get("adaptive", "no"),
            "keys":       item.get("keys", []),
        }
        metadata.append(meta)
        documents.append(build_document(item))

    # 3. Embed
    print("Loading embedding model (all-MiniLM-L6-v2)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding documents…")
    embeddings = model.encode(documents, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")
    print(f"Embeddings shape: {embeddings.shape}")

    # 4. Build FAISS index (Inner Product == cosine sim when normalized)
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    # 5. Save
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved index → {INDEX_PATH}")
    print(f"Saved metadata → {META_PATH}")
    print("Phase 2 build complete ✓")


if __name__ == "__main__":
    main()