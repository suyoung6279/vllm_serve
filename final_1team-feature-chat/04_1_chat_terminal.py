#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from rag.embeddings import DEFAULT_HF_EMBEDDING_MODEL, make_embedder
from rag.naver_news import news_source_references
from rag.qdrant_store import config_from_env, make_client


def load_query_module() -> ModuleType:
    module_path = Path(__file__).with_name("04_query_qdrant_openai.py")
    spec = importlib.util.spec_from_file_location("query_qdrant", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive terminal chatbot over Qdrant.")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--qdrant-api-key", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--prefer-grpc", action="store_true")
    parser.add_argument("--embedding-provider", choices=["huggingface", "openai"], default="huggingface")
    parser.add_argument("--embedding-model", "--model", dest="embedding_model", default=DEFAULT_HF_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-backend",
        choices=["sentence-transformers", "transformers"],
        default=os.getenv("RAG_HF_EMBEDDING_BACKEND", "transformers"),
        help="Local Hugging Face embedding backend. Use transformers to avoid sentence-transformers encode issues.",
    )
    parser.add_argument("--dimensions", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--answer-provider", choices=["huggingface", "openai"], default=os.getenv("RAG_ANSWER_PROVIDER", "huggingface"))
    parser.add_argument("--answer-model", default=os.getenv("RAG_ANSWER_MODEL", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"))
    parser.add_argument("--answer-revision", default=os.getenv("RAG_ANSWER_REVISION") or os.getenv("RAG_ANSWER_MODEL_REVISION") or None)
    parser.add_argument("--answer-max-new-tokens", type=int, default=int(os.getenv("RAG_ANSWER_MAX_NEW_TOKENS", "384")))
    parser.add_argument("--answer-torch-dtype", default=os.getenv("RAG_ANSWER_TORCH_DTYPE", "auto"))
    parser.add_argument("--answer-device-map", default=os.getenv("RAG_ANSWER_DEVICE_MAP", "auto"))
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--cohere-model", default=os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--dense-k", type=int, default=20)
    parser.add_argument("--dense-fetch-k", type=int, default=50)
    parser.add_argument("--bm25-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rrf-top-k", type=int, default=20)
    parser.add_argument("--rerank-top-n", type=int, default=10)
    parser.add_argument("--max-tokens-per-doc", type=int, default=4096)
    parser.add_argument("--where-doc-id", default=None)
    parser.add_argument("--where-source-contains", default=None)
    parser.add_argument("--exclude-chunk-types", default="image_caption,chart_caption")
    parser.add_argument("--max-context-chars", type=int, default=9000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--news-mode", choices=["fallback", "always", "off"], default="fallback")
    parser.add_argument("--no-news", dest="news_mode", action="store_const", const="off")
    parser.add_argument("--news-display", type=int, default=3)
    parser.add_argument("--news-sort", choices=["sim", "date"], default="sim")
    parser.add_argument("--news-max-context-chars", type=int, default=1500)
    parser.add_argument("--news-answer-max-new-tokens", type=int, default=int(os.getenv("RAG_NEWS_ANSWER_MAX_NEW_TOKENS", "512")))
    parser.add_argument("--news-fallback-dense-threshold", type=float, default=0.35)
    return parser.parse_args()

def main() -> int:
    configure_output_encoding()
    load_dotenv()
    args = parse_args()
    query_module = load_query_module()

    print("[초기화] 임베딩 모델 로드 중...")
    embedder = make_embedder(
        provider=args.embedding_provider,
        model=args.embedding_model,
        dimensions=args.dimensions,
        device=args.device,
        openai_api_key_env=args.openai_api_key_env,
        hf_token_env=args.hf_token_env,
        embedding_backend=args.embedding_backend,
    )
    print(f"[임베딩] provider={embedder.provider} model={embedder.model} device={embedder.device or 'remote/api'} dims={embedder.dimensions or 'model default'}")

    config = config_from_env(args.qdrant_url, args.qdrant_api_key, args.collection, args.prefer_grpc)
    client = make_client(config)
    collection = config.collection
    records = query_module.load_corpus(client, collection)
    bm25 = BM25Okapi([query_module.tokenize(query_module.retrieval_text(record)) for record in records])
    print("[초기화] 답변 모델 로드 중...")
    answer_client = query_module.make_answer_client(args)
    print(f"[답변 모델] provider={args.answer_provider} model={args.answer_model}")

    print("[준비 완료] 질문을 입력하세요. 종료: exit, quit, q")
    while True:
        try:
            query = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            break

        start = time.perf_counter()
        args.query = query
        try:
            if args.news_mode != "off" and (args.news_mode == "always" or query_module.is_news_intent(args.query)):
                hits = []
                news_items = query_module.maybe_search_news(args)
            else:
                hits = query_module.retrieve(args, embedder, client, collection, records, bm25)
                news_items = query_module.maybe_search_news(args, hits)
            answer_hits = [] if news_items else hits
            answer = query_module.run_with_max_new_tokens(
                answer_client,
                args.news_answer_max_new_tokens if news_items else None,
                lambda: query_module.generate_answer(
                    answer_client,
                    args.answer_model,
                    query,
                    answer_hits,
                    args.max_context_chars,
                    args.temperature,
                    news_items,
                    args.news_max_context_chars,
                ),
            )
            elapsed = time.perf_counter() - start
            print("\n[답변]")
            print(answer)
            if news_items:
                print("\n[Naver 뉴스 결과]")
                for ref in news_source_references(news_items):
                    print(ref)
            print(f"\n[답변 시간] {elapsed:.2f}초")
        except Exception as exc:
            elapsed = time.perf_counter() - start
            print(f"[ERROR] {type(exc).__name__}: {exc!r}")
            print(traceback.format_exc())
            print(f"[소요 시간] {elapsed:.2f}초")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
