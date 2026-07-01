"""
Retriever — loaded once at API startup, queried per request.
Usage:
    from rag.retriever import Retriever
    r = Retriever()
    results = r.query("Java developer stakeholder communication", top_k=10)
"""

import pickle
import pathlib
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = pathlib.Path(__file__).parent / "index.faiss"
META_PATH  = pathlib.Path(__file__).parent / "metadata.pkl"


class Retriever:
    def __init__(self):
        print("Loading FAISS index…")
        self.index = faiss.read_index(str(INDEX_PATH))

        print("Loading metadata…")
        with open(META_PATH, "rb") as f:
            self.metadata: list[dict] = pickle.load(f)

        print("Loading embedding model…")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"Retriever ready. {self.index.ntotal} assessments indexed.")

    def query(self, text: str, top_k: int = 20) -> list[dict]:
        """
        Returns top_k metadata dicts ranked by cosine similarity.
        Always returns at most top_k items.
        """
        vec = self.model.encode([text], normalize_embeddings=True)
        vec = np.array(vec, dtype="float32")

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:           # FAISS sentinel for "not found"
                continue
            entry = dict(self.metadata[idx])   # shallow copy
            entry["_score"] = float(score)
            results.append(entry)
        return results

    def get_by_name(self, name: str) -> dict | None:
        """Exact-ish lookup by name for compare queries."""
        name_lower = name.lower().strip()
        for item in self.metadata:
            if item["name"].lower().strip() == name_lower:
                return item
        # fuzzy fallback: check if name is a substring
        for item in self.metadata:
            if name_lower in item["name"].lower():
                return item
        return None