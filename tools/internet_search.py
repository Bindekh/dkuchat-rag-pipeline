from duckduckgo_search import DDGS

class InternetSearchTool:
    def __init__(self):
        self.ddgs = DDGS()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        results = []
        try:
            hits = self.ddgs.text(query, max_results=top_k)
            for item in hits:
                results.append({
                    "text": item.get("body", ""),
                    "source": item.get("href", ""),
                    "page": "N/A",
                    "score": 1.0,
                    "method": "internet"
                })
        except Exception as e:
            print(f"  [Internet] DuckDuckGo search failed: {e}")
        return results