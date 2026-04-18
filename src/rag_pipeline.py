import os
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from src.prompts import DEFAULT_PROMPT_NAME, PROMPT_VARIANTS, build_context
from src.retrievers_lc import wrap_retriever

load_dotenv()

DEFAULT_LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def load_llm(
    model_id: str = DEFAULT_LLM_MODEL,
    max_new_tokens: int = 512,
    provider: str = "auto",
) -> BaseChatModel:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN in .env — see .env.example")
    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        max_new_tokens=max_new_tokens,
        provider=provider,
        huggingfacehub_api_token=token,
    )
    return ChatHuggingFace(llm=endpoint)


class RAGPipeline:
    def __init__(
        self,
        bm25: Any,
        semantic: Any,
        retriever_name: str = "Hybrid",
        prompt_name: str = DEFAULT_PROMPT_NAME,
        llm: Optional[BaseChatModel] = None,
        top_k: int = 5,
    ):
        self.retriever = wrap_retriever(retriever_name, bm25, semantic, top_k=top_k)
        self.prompt = PROMPT_VARIANTS[prompt_name]
        self.llm = llm if llm is not None else load_llm()
        self.chain = (
            {
                "context": self.retriever | (lambda docs: build_context(docs)),
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def answer(self, query: str) -> dict:
        docs = self.retriever.invoke(query)
        text = self.chain.invoke(query)
        return {
            "answer": text,
            "sources": [{"page_content": d.page_content, **d.metadata} for d in docs],
        }
