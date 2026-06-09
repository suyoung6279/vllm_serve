from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from rank_bm25 import BM25Okapi

from rag.embeddings import DEFAULT_HF_EMBEDDING_MODEL, make_embedder
from rag.qdrant_store import config_from_env, make_client


@dataclass
class RagSettings:
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    collection: str | None = None
    prefer_grpc: bool = False
    embedding_provider: str = "huggingface"
    embedding_model: str = DEFAULT_HF_EMBEDDING_MODEL
    embedding_backend: str = os.getenv("RAG_HF_EMBEDDING_BACKEND", "transformers")
    dimensions: int | None = None
    device: str = "auto"
    answer_provider: str = os.getenv("RAG_ANSWER_PROVIDER", "huggingface")
    answer_model: str = os.getenv("RAG_ANSWER_MODEL", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    answer_revision: str | None = os.getenv("RAG_ANSWER_REVISION") or os.getenv("RAG_ANSWER_MODEL_REVISION") or None
    answer_max_new_tokens: int = int(os.getenv("RAG_ANSWER_MAX_NEW_TOKENS", "384"))
    answer_torch_dtype: str = os.getenv("RAG_ANSWER_TORCH_DTYPE", "auto")
    answer_device_map: str = os.getenv("RAG_ANSWER_DEVICE_MAP", "auto")
    openai_api_key_env: str = "OPENAI_API_KEY"
    hf_token_env: str = "HF_TOKEN"
    cohere_model: str = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
    top_k: int = 3
    dense_k: int = 20
    dense_fetch_k: int = 50
    bm25_k: int = 20
    rrf_k: int = 60
    rrf_top_k: int = 20
    rerank_top_n: int = 10
    max_tokens_per_doc: int = 4096
    where_doc_id: str | None = None
    where_source_contains: str | None = None
    exclude_chunk_types: str = "image_caption,chart_caption"
    max_context_chars: int = 9000
    temperature: float = 0.0
    skip_rerank: bool = True
    news_mode: str = "fallback"
    news_display: int = 3
    news_sort: str = "sim"
    news_max_context_chars: int = 1500
    news_answer_max_new_tokens: int = int(os.getenv("RAG_NEWS_ANSWER_MAX_NEW_TOKENS", "512"))
    news_fallback_dense_threshold: float = 0.35


def load_query_module() -> ModuleType:
    from pathlib import Path

    module_path = Path(__file__).resolve().parent.parent / "04_query_qdrant_openai.py"
    spec = importlib.util.spec_from_file_location("query_qdrant_api", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RagService:
    def __init__(self, settings: RagSettings | None = None) -> None:
        self.settings = settings or RagSettings()
        self.query_module: ModuleType | None = None
        self.embedder: Any = None
        self.client: Any = None
        self.collection: str | None = None
        self.records: list[Any] = []
        self.bm25: BM25Okapi | None = None
        self.answer_client: Any | None = None
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return

        self.query_module = load_query_module()
        self.embedder = make_embedder(
            provider=self.settings.embedding_provider,
            model=self.settings.embedding_model,
            dimensions=self.settings.dimensions,
            device=self.settings.device,
            openai_api_key_env=self.settings.openai_api_key_env,
            hf_token_env=self.settings.hf_token_env,
            embedding_backend=self.settings.embedding_backend,
        )

        config = config_from_env(
            self.settings.qdrant_url,
            self.settings.qdrant_api_key,
            self.settings.collection,
            self.settings.prefer_grpc,
        )
        self.client = make_client(config)
        self.collection = config.collection
        self.records = self.query_module.load_corpus(self.client, self.collection)
        self.bm25 = BM25Okapi(
            [self.query_module.tokenize(self.query_module.retrieval_text(record)) for record in self.records]
        )
        self.answer_client = self.query_module.make_answer_client(self._request_args(""))
        self.loaded = True

    def _request_args(self, question: str) -> Any:
        if self.query_module is None:
            raise RuntimeError("RagService is not loaded.")

        return self.query_module.argparse.Namespace(
            query=question,
            qdrant_url=self.settings.qdrant_url,
            qdrant_api_key=self.settings.qdrant_api_key,
            collection=self.settings.collection,
            prefer_grpc=self.settings.prefer_grpc,
            embedding_provider=self.settings.embedding_provider,
            embedding_model=self.settings.embedding_model,
            embedding_backend=self.settings.embedding_backend,
            dimensions=self.settings.dimensions,
            device=self.settings.device,
            answer_provider=self.settings.answer_provider,
            answer_model=self.settings.answer_model,
            answer_revision=self.settings.answer_revision,
            answer_max_new_tokens=self.settings.answer_max_new_tokens,
            answer_torch_dtype=self.settings.answer_torch_dtype,
            answer_device_map=self.settings.answer_device_map,
            openai_api_key_env=self.settings.openai_api_key_env,
            hf_token_env=self.settings.hf_token_env,
            cohere_model=self.settings.cohere_model,
            top_k=self.settings.top_k,
            dense_k=self.settings.dense_k,
            dense_fetch_k=self.settings.dense_fetch_k,
            bm25_k=self.settings.bm25_k,
            rrf_k=self.settings.rrf_k,
            rrf_top_k=self.settings.rrf_top_k,
            rerank_top_n=self.settings.rerank_top_n,
            max_tokens_per_doc=self.settings.max_tokens_per_doc,
            where_doc_id=self.settings.where_doc_id,
            where_source_contains=self.settings.where_source_contains,
            exclude_chunk_types=self.settings.exclude_chunk_types,
            max_context_chars=self.settings.max_context_chars,
            preview_chars=700,
            temperature=self.settings.temperature,
            skip_rerank=self.settings.skip_rerank,
            no_answer=False,
            hide_context=True,
            news_mode=self.settings.news_mode,
            news_display=self.settings.news_display,
            news_sort=self.settings.news_sort,
            news_max_context_chars=self.settings.news_max_context_chars,
            news_answer_max_new_tokens=self.settings.news_answer_max_new_tokens,
            news_fallback_dense_threshold=self.settings.news_fallback_dense_threshold,
        )

    def ask(self, question: str) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("RagService is not loaded.")
        if self.query_module is None or self.bm25 is None or self.answer_client is None:
            raise RuntimeError("RagService is missing required runtime objects.")

        question = question.strip()
        if not question:
            raise ValueError("question is empty")

        started = time.perf_counter()
        args = self._request_args(question)
        if self.client is None or self.collection is None:
            raise RuntimeError("RagService is missing Qdrant runtime objects.")
        route = "news" if args.news_mode != "off" and (args.news_mode == "always" or self.query_module.is_news_intent(args.query)) else "rag"
        if route == "news":
            hits = []
            news_items = self.query_module.maybe_search_news(args)
        else:
            hits = self.query_module.retrieve(args, self.embedder, self.client, self.collection, self.records, self.bm25)
            news_items = self.query_module.maybe_search_news(args, hits)
        answer_hits = [] if news_items else hits
        answer = self.query_module.run_with_max_new_tokens(
            self.answer_client,
            self.settings.news_answer_max_new_tokens if news_items else None,
            lambda: self.query_module.generate_answer(
                self.answer_client,
                self.settings.answer_model,
                question,
                answer_hits,
                self.settings.max_context_chars,
                self.settings.temperature,
                news_items,
                self.settings.news_max_context_chars,
            ),
        )
        elapsed = time.perf_counter() - started
        sources = [] if news_items else self.query_module.answer_source_references(hits)
        news_sources = self.query_module.news_source_references(news_items)
        generate_stats = getattr(self.answer_client, "last_generate_stats", {})

        return {
            "answer": answer,
            "sources": sources,
            "news_sources": news_sources,
            "elapsed_sec": round(elapsed, 3),
            "route": route,
            "generate_stats": generate_stats,
        }
