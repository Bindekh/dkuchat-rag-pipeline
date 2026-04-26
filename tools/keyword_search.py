import pickle

class KeywordSearchTool:
    def __init__(self, bm25_path, chunks_path):
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_ids:
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "score": float(scores[idx]),
                "method": "keyword"
            })
        return results