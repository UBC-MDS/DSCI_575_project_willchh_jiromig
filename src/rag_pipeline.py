import os
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from src.prompts import DEFAULT_PROMPT_NAME, PROMPT_VARIANTS, build_context, build_web_context
from src.retrievers_lc import wrap_retriever
from src.tools import web_search_snippets

load_dotenv()

DEFAULT_LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def load_llm(
    model_id: str = DEFAULT_LLM_MODEL,
    max_new_tokens: int = 512,
    provider: str = "auto",
    temperature: float = 0.1,
    do_sample: bool = True,
    seed: int = 42,
) -> BaseChatModel:
    """Create a ChatHuggingFace LLM from the HF Inference API."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN in .env — see .env.example")
    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        max_new_tokens=max_new_tokens,
        provider=provider,
        huggingfacehub_api_token=token,
        temperature=temperature,
        do_sample=do_sample,
        seed=seed,
    )
    return ChatHuggingFace(llm=endpoint)


class RAGPipeline:
    """RAG pipeline composing retriever, context builder, prompt, and LLM via LCEL."""

    def __init__(
        self,
        bm25: Any,
        semantic: Any,
        retriever_name: str = "Hybrid",
        prompt_name: str = DEFAULT_PROMPT_NAME,
        llm: Optional[BaseChatModel] = None,
        top_k: int = 5,
    ):
        """Build the LCEL chain from retriever, prompt variant, and LLM."""
        self.top_k = top_k
        self.retriever = wrap_retriever(retriever_name, bm25, semantic, top_k=top_k)
        self.prompt = PROMPT_VARIANTS[prompt_name]
        self.llm = llm if llm is not None else load_llm()
        self.chain = self.prompt | self.llm | StrOutputParser()

    def answer(self, query: str, use_web_search: bool = False) -> dict:
        """Run the full RAG pipeline and return answer text with source documents.

        When ``use_web_search`` is true, Tavily snippets are fetched and placed in
        a labeled Web Context block in the prompt. Tavily failures are caught and
        surfaced via ``web_warning`` so the RAG path still produces an answer.
        """
        docs = self.retriever.invoke(query)[: self.top_k]
        web_sources: list[str] = []
        web_warning: Optional[str] = None
        if use_web_search:
            try:
                web_sources = web_search_snippets(query, max_results=self.top_k)
            except Exception as exc:  # noqa: BLE001
                web_warning = f"Web search failed: {exc}"

        text = self.chain.invoke(
            {
                "context": build_context(docs),
                "web_context": build_web_context(web_sources),
                "question": query,
            }
        )
        return {
            "answer": text,
            "sources": [{"page_content": d.page_content, **d.metadata} for d in docs],
            "web_sources": web_sources,
            "web_warning": web_warning,
        }
