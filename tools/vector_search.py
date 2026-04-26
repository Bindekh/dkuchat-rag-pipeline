import pickle, faiss, numpy as np
from sentence_transformers import SentenceTransformer

class VectorSearchTool:
    def __init__(self, index_path, chunks_path, model_name="BAAI/bge-small-en-v1.5"):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        emb = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(emb)
        scores, ids = self.index.search(emb, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "score": float(score),
                "method": "vector"
            })
        return results