import os
import dspy
from tools.vector_search import VectorSearchTool
from tools.keyword_search import KeywordSearchTool
from tools.internet_search import InternetSearchTool

# --- DSPy Signatures ---

class ToolSelector(dspy.Signature):
    """Given a user question, decide which search tools to use.
    Output a comma-separated list of: vector, keyword, internet"""
    question: str = dspy.InputField(desc="User question in English or Chinese")
    tools_to_use: str = dspy.OutputField(desc="Comma-separated tools: vector, keyword, internet")

class AnswerWithCitations(dspy.Signature):
    """Answer the question using the retrieved context.
    IMPORTANT: You MUST respond in the same language as specified in the 'language' field.
    If language is 'English', answer in English only. No Chinese characters.
    If language is 'Chinese', answer in Chinese only.
    Always cite sources using format: [source: filename, page: N]"""
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="Retrieved passages with source info")
    language: str = dspy.InputField(desc="Language to respond in: 'English' or 'Chinese'")
    answer: str = dspy.OutputField(desc="Answer in the specified language with inline citations")

# --- DSPy Agent Module ---

class RAGAgent(dspy.Module):
    def __init__(self, index_dir="index/"):
        super().__init__()
        self.tool_selector = dspy.Predict(ToolSelector)
        self.answerer = dspy.Predict(AnswerWithCitations)

        self.vector_tool = VectorSearchTool(
            index_path=f"{index_dir}/faiss_BAAI_bge-small-en-v1.5.index",
            chunks_path=f"{index_dir}/chunks.pkl"
        )
        self.keyword_tool = KeywordSearchTool(
            bm25_path=f"{index_dir}/bm25.pkl",
            chunks_path=f"{index_dir}/chunks.pkl"
        )
        self.internet_tool = InternetSearchTool()

    def forward(self, question: str) -> dspy.Prediction:
        # Step 1: detect language from question
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in question)
        language = "Chinese" if has_chinese else "English"

        # Step 2: select tools
        selection = self.tool_selector(question=question)
        tools = [t.strip() for t in selection.tools_to_use.split(",")]

        # Step 3: run selected tools — reduced top_k to limit context size
        all_results = []
        if "vector" in tools:
            all_results += self.vector_tool.search(question, top_k=2)
        if "keyword" in tools:
            all_results += self.keyword_tool.search(question, top_k=1)
        if "internet" in tools:
            all_results += self.internet_tool.search(question, top_k=1)

        # Step 4: format context with hard 600-word limit
        context_parts = []
        total_words = 0
        for r in all_results:
            chunk_text = (
                f"[Source: {r['source']}, Page: {r['page']}, "
                f"Method: {r['method']}]\n{r['text']}"
            )
            chunk_words = len(chunk_text.split())
            if total_words + chunk_words > 600:
                break
            context_parts.append(chunk_text)
            total_words += chunk_words

        context = "\n\n---\n\n".join(context_parts)

        # Step 5: generate answer with citations in correct language
        result = self.answerer(
            question=question,
            context=context,
            language=language
        )
        return dspy.Prediction(
            answer=result.answer,
            sources=all_results,
            tools_used=tools
        )


def setup_dspy(model_name=None, base_url=None):
    """Configure DSPy to use local LLM via OpenAI-compatible endpoint."""
    model_name = model_name or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    base_url   = base_url   or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")

    lm = dspy.LM(
        model=f"openai/{model_name}",
        api_base=base_url,
        api_key="not-needed",
        max_tokens=512,
        temperature=0.1,
        cache=False
    )
    dspy.configure(lm=lm)
    print(f"[DSPy] Using model : {model_name}")
    print(f"[DSPy] Server URL  : {base_url}")
    return model_name