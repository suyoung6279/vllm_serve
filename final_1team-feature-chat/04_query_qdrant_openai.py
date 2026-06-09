#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

from rag.embeddings import DEFAULT_HF_EMBEDDING_MODEL, make_embedder
from rag.naver_news import NewsItem, news_context, news_source_references, search_naver_news
from rag.qdrant_store import config_from_env, make_client, scroll_records, search_points

try:
    import cohere
except Exception:
    cohere = None


TOKEN_RE = re.compile(r"[가-힣]{2,}|[A-Za-z][A-Za-z0-9_-]+|\d+(?:\.\d+)?")
DEFAULT_HF_ANSWER_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
DEFAULT_HF_ANSWER_REVISION = None


@dataclass
class CorpusRecord:
    chunk_id: str
    document: str
    metadata: dict[str, Any]


@dataclass
class SearchHit:
    rank: int
    chunk_id: str
    document: str
    metadata: dict[str, Any]
    source: str
    score: float
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    distance: float | None = None


def configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid RAG query over a Qdrant collection.")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--qdrant-api-key", default=None)
    parser.add_argument("--collection", type=str, default=None)
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
    parser.add_argument("--answer-model", default=os.getenv("RAG_ANSWER_MODEL", DEFAULT_HF_ANSWER_MODEL))
    parser.add_argument(
        "--answer-revision",
        default=os.getenv("RAG_ANSWER_REVISION") or os.getenv("RAG_ANSWER_MODEL_REVISION") or DEFAULT_HF_ANSWER_REVISION,
    )
    parser.add_argument("--answer-max-new-tokens", type=int, default=int(os.getenv("RAG_ANSWER_MAX_NEW_TOKENS", "384")))
    parser.add_argument("--answer-torch-dtype", default=os.getenv("RAG_ANSWER_TORCH_DTYPE", "auto"))
    parser.add_argument("--answer-device-map", default=os.getenv("RAG_ANSWER_DEVICE_MAP", "auto"))
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--cohere-model", default=os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"))
    parser.add_argument("--query", required=True)
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
    parser.add_argument("--preview-chars", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--no-answer", action="store_true")
    parser.add_argument("--hide-context", action="store_true")
    parser.add_argument("--news-mode", choices=["fallback", "always", "off"], default="fallback")
    parser.add_argument("--no-news", dest="news_mode", action="store_const", const="off")
    parser.add_argument("--news-display", type=int, default=3)
    parser.add_argument("--news-sort", choices=["sim", "date"], default="sim")
    parser.add_argument("--news-max-context-chars", type=int, default=1500)
    parser.add_argument("--news-answer-max-new-tokens", type=int, default=int(os.getenv("RAG_NEWS_ANSWER_MAX_NEW_TOKENS", "512")))
    parser.add_argument("--news-fallback-dense-threshold", type=float, default=0.35)
    return parser.parse_args()


def make_openai_client(env_name: str) -> OpenAI:
    api_key = os.getenv(env_name)
    if not api_key:
        raise RuntimeError(f"Missing {env_name}. Put it in .env or export it.")
    return OpenAI(api_key=api_key)


class HuggingFaceAnswerClient:
    def __init__(
        self,
        model_name: str,
        hf_token_env: str = "HF_TOKEN",
        revision: str | None = None,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        max_new_tokens: int = 768,
    ) -> None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("Local HF answer generation requires torch and transformers.") from exc

        token = os.getenv(hf_token_env) or None
        revision = revision or None
        self.model_name = model_name
        self.revision = revision
        self.torch = torch
        tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if token:
            tokenizer_kwargs["token"] = token
        if revision:
            tokenizer_kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": device_map,
        }
        if token:
            model_kwargs["token"] = token
        if revision:
            model_kwargs["revision"] = revision
        if torch_dtype == "auto":
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = getattr(torch, str(torch_dtype), torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.last_generate_stats: dict[str, Any] = {}
        self._patch_causal_mask_aliases()

    def _patch_causal_mask_aliases(self) -> None:
        patched: set[int] = set()

        def make_wrapper(func: Any) -> Any:
            try:
                params = inspect.signature(func).parameters
            except Exception:
                return func
            accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
            accepts_input = "input_embeds" in params
            accepts_inputs = "inputs_embeds" in params
            accepted_names = set(params)
            if accepts_kwargs and accepts_input == accepts_inputs:
                return func

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if accepts_inputs and "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                    kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
                elif accepts_input and "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
                    kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
                if not accepts_kwargs:
                    kwargs = {key: value for key, value in kwargs.items() if key in accepted_names}
                return func(*args, **kwargs)

            return wrapper

        for module in list(sys.modules.values()):
            module_name = getattr(module, "__name__", "")
            if "transformers_modules" not in module_name and "modeling_exaone" not in module_name:
                continue
            for attr in ("create_causal_mask", "create_sliding_window_causal_mask"):
                func = getattr(module, attr, None)
                if func is None or id(func) in patched:
                    continue
                wrapped = make_wrapper(func)
                if wrapped is not func:
                    setattr(module, attr, wrapped)
                    patched.add(id(wrapped))

    def generate(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            template_output = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=False,
            )
        except Exception as exc:
            raise RuntimeError("answer.generate.apply_chat_template failed") from exc

        if hasattr(template_output, "keys") and "input_ids" in template_output:
            input_ids = template_output["input_ids"]
        else:
            input_ids = template_output
        if not hasattr(input_ids, "shape"):
            input_ids = self.torch.tensor([input_ids], dtype=self.torch.long)
        input_ids = input_ids.to(self.model.device)
        prompt_length = input_ids.shape[-1]
        print(f"[생성 입력] input_ids={tuple(input_ids.shape)}")

        do_sample = temperature > 0
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
        try:
            started = time.perf_counter()
            with self.torch.inference_mode():
                outputs = self.model.generate(input_ids, **generate_kwargs)
            generate_sec = time.perf_counter() - started
        except Exception as exc:
            raise RuntimeError(f"answer.generate.model_generate failed: {type(exc).__name__}: {exc}") from exc

        try:
            generated = outputs[0][prompt_length:]
            self.last_generate_stats = {
                "input_tokens": int(prompt_length),
                "generated_tokens": int(generated.shape[-1]),
                "generate_sec": round(generate_sec, 3),
                "max_new_tokens": int(self.max_new_tokens),
            }
            print(
                "[생성 시간] "
                f"{self.last_generate_stats['generate_sec']:.3f}초 "
                f"입력토큰={self.last_generate_stats['input_tokens']} "
                f"생성토큰={self.last_generate_stats['generated_tokens']} "
                f"최대생성={self.last_generate_stats['max_new_tokens']}"
            )
            return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as exc:
            raise RuntimeError(f"answer.generate.decode failed: {type(exc).__name__}: {exc}") from exc


def make_answer_client(args: argparse.Namespace) -> Any:
    answer_provider = getattr(args, "answer_provider", "huggingface")
    if answer_provider == "openai":
        return make_openai_client(getattr(args, "openai_api_key_env", "OPENAI_API_KEY"))
    return HuggingFaceAnswerClient(
        getattr(args, "answer_model", DEFAULT_HF_ANSWER_MODEL),
        hf_token_env=getattr(args, "hf_token_env", "HF_TOKEN"),
        revision=getattr(args, "answer_revision", DEFAULT_HF_ANSWER_REVISION),
        torch_dtype=getattr(args, "answer_torch_dtype", "auto"),
        device_map=getattr(args, "answer_device_map", "auto"),
        max_new_tokens=getattr(args, "answer_max_new_tokens", 768),
    )


def run_answer_client(client: Any, model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    if hasattr(client, "generate"):
        return client.generate(system_prompt, user_prompt, temperature)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )
    return response.choices[0].message.content or ""


def run_with_max_new_tokens(client: Any, max_new_tokens: int | None, fn: Any) -> Any:
    if max_new_tokens is None or not hasattr(client, "max_new_tokens"):
        return fn()
    original = client.max_new_tokens
    client.max_new_tokens = max_new_tokens
    try:
        return fn()
    finally:
        client.max_new_tokens = original


def get_cohere_client() -> Any:
    api_key = os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY")
    if not api_key:
        raise RuntimeError("Missing COHERE_API_KEY. Add it to .env or run with --skip-rerank.")
    if cohere is None:
        raise RuntimeError("cohere package is not installed. Run: python -m pip install cohere")
    if hasattr(cohere, "ClientV2"):
        return cohere.ClientV2(api_key=api_key)
    return cohere.Client(api_key)


def normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            stripped = value.strip()
            if (stripped.startswith("[") and stripped.endswith("]")) or (stripped.startswith("{") and stripped.endswith("}")):
                try:
                    out[key] = json.loads(stripped)
                    continue
                except Exception:
                    pass
        out[key] = value
    return out


def source_filename(meta: dict[str, Any]) -> str:
    name = str(meta.get("source_filename") or meta.get("source") or meta.get("file_name") or "unknown.pdf")
    return Path(name).name


def page_label(meta: dict[str, Any]) -> str:
    start = meta.get("page_start") or meta.get("page_no") or meta.get("page")
    end = meta.get("page_end") or start
    if start is None:
        return ""
    if end and end != start:
        return f"{start}-{end}"
    return str(start)


def source_label(meta: dict[str, Any], include_page: bool = True) -> str:
    filename = source_filename(meta)
    if include_page and page_label(meta):
        return f"{filename} p.{page_label(meta)}"
    return filename


def source_matches(meta: dict[str, Any], source_contains: str | None) -> bool:
    return not source_contains or source_contains.lower() in source_filename(meta).lower()


def chunk_type_allowed(meta: dict[str, Any], excluded_types: set[str]) -> bool:
    return str(meta.get("chunk_type") or meta.get("block_type") or "") not in excluded_types


def normalize_context_text(text: str) -> str:
    return " ".join((text or "").split())


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_RE.findall((text or "").lower()):
        tokens.append(token)
        if re.fullmatch(r"[가-힣]{3,}", token):
            tokens.extend(token[i : i + 2] for i in range(len(token) - 1))
            if len(token) >= 5:
                tokens.extend(token[i : i + 3] for i in range(len(token) - 2))
    return tokens


def retrieval_text(record: CorpusRecord) -> str:
    meta = record.metadata
    parts = [source_filename(meta), page_label(meta), str(meta.get("chunk_type") or "")]
    heading = meta.get("heading") or meta.get("article_title")
    if heading:
        parts.append(str(heading))
    return " | ".join(part for part in parts if part) + "\n" + record.document


def load_corpus(client: Any, collection: str, batch_size: int = 1000) -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    for record in scroll_records(client, collection, batch_size=batch_size):
        records.append(CorpusRecord(record.chunk_id, record.document, normalize_metadata(record.metadata)))
    return records


def distance_to_similarity(distance: float | None) -> float:
    if distance is None:
        return 0.0
    if 0 <= distance <= 2.5:
        return max(0.0, min(1.0, 1.0 - distance / 2.0))
    return 1.0 / (1.0 + distance)


def dense_search(client: Any, collection: str, query_embedding: list[float], args: argparse.Namespace, excluded_types: set[str]) -> list[SearchHit]:
    points = search_points(
        client,
        collection,
        query_embedding,
        limit=max(args.dense_fetch_k, args.dense_k),
        doc_id=args.where_doc_id,
    )
    hits: list[SearchHit] = []
    for point in points:
        payload = getattr(point, "payload", None) or {}
        raw_meta = payload.get("metadata") if isinstance(payload, dict) else {}
        clean_meta = normalize_metadata(raw_meta if isinstance(raw_meta, dict) else {})
        if not source_matches(clean_meta, args.where_source_contains):
            continue
        if not chunk_type_allowed(clean_meta, excluded_types):
            continue
        rank = len(hits) + 1
        raw_score = getattr(point, "score", None)
        score = float(raw_score) if raw_score is not None else 0.0
        chunk_id = str(payload.get("chunk_id") or clean_meta.get("chunk_id") or getattr(point, "id", ""))
        doc = str(payload.get("document") or "")
        hits.append(SearchHit(rank, chunk_id, doc, clean_meta, "dense", score, dense_rank=rank, dense_score=score, distance=None))
        if len(hits) >= args.dense_k:
            break
    return hits


def bm25_search(records: list[CorpusRecord], bm25: BM25Okapi, query: str, args: argparse.Namespace, excluded_types: set[str]) -> list[SearchHit]:
    scores = bm25.get_scores(tokenize(query))
    candidate_indices = [
        i
        for i, record in enumerate(records)
        if source_matches(record.metadata, args.where_source_contains)
        and chunk_type_allowed(record.metadata, excluded_types)
        and (not args.where_doc_id or record.metadata.get("doc_id") == args.where_doc_id)
    ]
    candidate_indices.sort(key=lambda i: float(scores[i]), reverse=True)
    hits: list[SearchHit] = []
    for i in candidate_indices[: args.bm25_k]:
        record = records[i]
        rank = len(hits) + 1
        score = float(scores[i])
        hits.append(SearchHit(rank, record.chunk_id, record.document, record.metadata, "bm25", score, bm25_rank=rank, bm25_score=score))
    return hits


def rrf_fuse(dense_hits: list[SearchHit], bm25_hits: list[SearchHit], rrf_k: int, top_k: int) -> list[SearchHit]:
    by_id: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    dense_by_id = {hit.chunk_id: hit for hit in dense_hits}
    bm25_by_id = {hit.chunk_id: hit for hit in bm25_hits}
    for hit in dense_hits:
        by_id[hit.chunk_id] = hit
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + hit.rank)
    for hit in bm25_hits:
        by_id.setdefault(hit.chunk_id, hit)
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + hit.rank)

    fused: list[SearchHit] = []
    for chunk_id in sorted(scores, key=scores.get, reverse=True)[:top_k]:
        base = by_id[chunk_id]
        dense = dense_by_id.get(chunk_id)
        bm25 = bm25_by_id.get(chunk_id)
        rank = len(fused) + 1
        fused.append(
            SearchHit(
                rank=rank,
                chunk_id=base.chunk_id,
                document=base.document,
                metadata=base.metadata,
                source="rrf",
                score=float(scores[chunk_id]),
                dense_rank=dense.dense_rank if dense else None,
                dense_score=dense.dense_score if dense else None,
                bm25_rank=bm25.bm25_rank if bm25 else None,
                bm25_score=bm25.bm25_score if bm25 else None,
                rrf_rank=rank,
                rrf_score=float(scores[chunk_id]),
                distance=dense.distance if dense else None,
            )
        )
    return fused


def format_for_cohere(hit: SearchHit, max_chars: int = 12000) -> str:
    return f"source: {source_label(hit.metadata)}\ntext: {normalize_context_text(hit.document)[:max_chars]}"


def cohere_rerank(co_client: Any, query: str, candidates: list[SearchHit], args: argparse.Namespace) -> list[SearchHit]:
    if not candidates:
        return []
    docs = [format_for_cohere(hit) for hit in candidates]
    top_n = min(args.rerank_top_n, len(docs))
    try:
        response = co_client.rerank(
            model=args.cohere_model,
            query=query,
            documents=docs,
            top_n=top_n,
            max_tokens_per_doc=args.max_tokens_per_doc,
        )
    except TypeError:
        response = co_client.rerank(model=args.cohere_model, query=query, documents=docs, top_n=top_n)

    results = getattr(response, "results", None)
    if results is None and isinstance(response, dict):
        results = response.get("results", [])
    reranked: list[SearchHit] = []
    for item in results or []:
        idx = getattr(item, "index", None)
        score = getattr(item, "relevance_score", None)
        if idx is None and isinstance(item, dict):
            idx = item.get("index")
            score = item.get("relevance_score")
        if idx is None:
            continue
        base = candidates[int(idx)]
        rank = len(reranked) + 1
        reranked.append(
            SearchHit(
                rank=rank,
                chunk_id=base.chunk_id,
                document=base.document,
                metadata=base.metadata,
                source="rerank",
                score=float(score or 0.0),
                dense_rank=base.dense_rank,
                dense_score=base.dense_score,
                bm25_rank=base.bm25_rank,
                bm25_score=base.bm25_score,
                rrf_rank=base.rrf_rank,
                rrf_score=base.rrf_score,
                rerank_score=float(score or 0.0),
                distance=base.distance,
            )
        )
    return reranked


def fmt_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def build_context(hits: list[SearchHit], max_chars: int) -> str:
    parts: list[str] = []
    used = 0
    for hit in hits:
        text = normalize_context_text(hit.document)
        if not text:
            continue
        block = f"[{hit.rank}] 출처: {source_label(hit.metadata)}\n{text}"
        if used + len(block) > max_chars:
            block = block[: max(0, max_chars - used)].rstrip()
        if block:
            parts.append(block)
            used += len(block) + 2
        if used >= max_chars:
            break
    return "\n\n".join(parts)


def unique_source_filenames(hits: list[SearchHit]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for hit in hits:
        name = source_filename(hit.metadata)
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def answer_source_references(hits: list[SearchHit]) -> list[str]:
    references: list[str] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        filename = source_filename(hit.metadata)
        page = page_label(hit.metadata)
        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)
        page_text = f"{page}page" if page else "page unknown"
        references.append(f"[{len(references) + 1}] {filename} {page_text}")
    return references


def generate_answer(client: OpenAI, model: str, query: str, hits: list[SearchHit], max_context_chars: int, temperature: float) -> str:
    context = build_context(hits, max_context_chars)
    if not context:
        return "검색된 근거가 없어 답변할 수 없습니다."
    source_refs = "\n".join(answer_source_references(hits))
    system_prompt = (
        "너는 한국어 RAG 챗봇이다. 제공된 근거 안에서만 답변한다. "
        "근거에서 확인되지 않는 내용은 모른다고 말한다. "
        "답변 끝에 '출처:'를 쓰고, 각 출처를 '[1] 파일명.pdf 10page \n' 형식으로 적는다. "
        "출처에는 파일명과 페이지 외에 경로, 청크 번호, 점수는 쓰지 않는다."
        "근거가 확인되지 않는 내용의 출처는 명시하지 않는다."
    )
    user_prompt = (
        f"질문:\n{query}\n\n"
        f"검색 근거:\n{context}\n\n"
        f"사용 가능한 출처 표기:\n{source_refs}\n\n"
        "위 근거만 사용해서 한국어로 답변해줘."
    )
    return run_answer_client(client, model, system_prompt, user_prompt, temperature)


def generate_answer(
    client: OpenAI,
    model: str,
    query: str,
    hits: list[SearchHit],
    max_context_chars: int,
    temperature: float,
    news_items: list[NewsItem] | None = None,
    news_max_context_chars: int = 3000,
) -> str:
    context = build_context(hits, max_context_chars)
    news_items = news_items or []
    external_context = news_context(news_items, news_max_context_chars)
    if news_items:
        context = ""
    if not context and not external_context:
        return "검색된 내부문서와 뉴스 근거가 없어 답변할 수 없습니다."

    source_refs = "\n".join(answer_source_references(hits if not news_items else []))
    news_refs = "\n".join(news_source_references(news_items))
    system_prompt = """
너는 한국어 RAG 챗봇이다.
제공된 내부문서 근거와 Naver 뉴스 근거 안에서만 답한다.
내부문서 근거가 없고 뉴스 근거만 있으면 뉴스 근거만 사용한다.
뉴스 근거가 없고 내부문서 근거만 있으면 내부문서 근거만 사용한다.
둘 다 있는 경우에는 내부문서와 뉴스 근거를 구분해서 사용한다.

확인되지 않은 내용은 추측하지 말고, 모르면 모른다고 답한다.
본문에는 URL을 직접 쓰지 않는다.
본문에서 출처가 필요한 문장 끝에는 [1], [2]처럼 번호만 붙인다.
답변 맨 아래에만 '출처:' 섹션을 만들고 전체 출처 목록을 표시한다.

뉴스 질의일 경우:
- 기사 제목을 단순 나열하지 않는다.
- 먼저 최근 흐름을 2~3문장으로 종합 요약한다.
- 그다음 핵심 이슈를 3개 이하로 묶어서 설명한다.
- 각 이슈 설명 끝에 관련 출처 번호를 붙인다.
- 출처 섹션에는 '[번호] 뉴스 제목 - 언론사 - URL' 형식으로 표시한다.

내부문서 질의일 경우:
- 문서 근거를 기준으로 답한다.
- 조항, 수치, 조건, 절차가 있으면 빠뜨리지 않는다.
- 출처 섹션에는 '[번호] 문서명 p.페이지' 형식으로 표시한다.
"""
    user_prompt = (
        f"질문:\n{query}\n\n"
        f"[내부문서 근거]\n{context or '(없음)'}\n\n"
        f"[뉴스 근거]\n{external_context or '(없음)'}\n\n"
        f"[사용 가능한 내부문서 출처]\n{source_refs or '(없음)'}\n\n"
        f"[사용 가능한 뉴스 출처]\n{news_refs or '(없음)'}\n\n"
        "위 근거만 사용해서 한국어로 답변해줘."
    )
    return run_answer_client(client, model, system_prompt, user_prompt, temperature)


NEWS_INTENT_KEYWORDS = ("뉴스", "최근", "최신", "오늘", "어제", "요즘", "시사", "속보", "동향", "보도", "기사")


def is_news_intent(query: str) -> bool:
    return any(keyword in query for keyword in NEWS_INTENT_KEYWORDS)


def internal_answer_confident(hits: list[SearchHit], min_dense_score: float) -> bool:
    if not hits:
        return False
    best_dense = max((hit.dense_score or 0.0 for hit in hits), default=0.0)
    return best_dense >= min_dense_score


def should_search_news(args: argparse.Namespace, hits: list[SearchHit]) -> bool:
    mode = getattr(args, "news_mode", "fallback")
    if mode == "off":
        return False
    if mode == "always":
        return True
    if is_news_intent(args.query):
        return True
    return not internal_answer_confident(hits, args.news_fallback_dense_threshold)


def maybe_search_news(args: argparse.Namespace, hits: list[SearchHit] | None = None) -> list[NewsItem]:
    if hits is not None and not should_search_news(args, hits):
        return []
    try:
        return search_naver_news(args.query, display=args.news_display, sort=args.news_sort)
    except Exception as exc:
        print(f"[WARN] Naver 뉴스 검색 실패: {exc}", file=sys.stderr)
        return []


def print_hits(hits: list[SearchHit], preview_chars: int, hide_context: bool) -> None:
    print("\n[최종 검색 결과]")
    if not hits:
        print("검색 결과가 없습니다.")
        return
    for hit in hits:
        print("=" * 100)
        print(
            f"[{hit.rank}] {source_label(hit.metadata)} "
            f"rerank={fmt_score(hit.rerank_score)} rrf={fmt_score(hit.rrf_score)} "
            f"dense={fmt_score(hit.dense_score)} bm25={fmt_score(hit.bm25_score)}"
        )
        print(f"id={hit.chunk_id}")
        if not hide_context:
            print("-" * 100)
            print(normalize_context_text(hit.document)[:preview_chars])


def retrieve(args: argparse.Namespace, embedder: Any, client: Any, collection: str, records: list[CorpusRecord], bm25: BM25Okapi) -> list[SearchHit]:
    total_started = time.perf_counter()
    excluded_types = parse_csv(args.exclude_chunk_types)
    started = time.perf_counter()
    query_embedding = embedder.embed_query(args.query)
    embed_sec = time.perf_counter() - started
    started = time.perf_counter()
    dense_hits = dense_search(client, collection, query_embedding, args, excluded_types)
    dense_sec = time.perf_counter() - started
    started = time.perf_counter()
    bm25_hits = bm25_search(records, bm25, args.query, args, excluded_types)
    bm25_sec = time.perf_counter() - started
    started = time.perf_counter()
    rrf_hits = rrf_fuse(dense_hits, bm25_hits, args.rrf_k, args.rrf_top_k)
    rrf_sec = time.perf_counter() - started
    if not args.skip_rerank:
        started = time.perf_counter()
        co_client = get_cohere_client()
        final_candidates = cohere_rerank(co_client, args.query, rrf_hits, args)
        rerank_sec = time.perf_counter() - started
    else:
        final_candidates = rrf_hits
        rerank_sec = 0.0
    final_hits = final_candidates[: args.top_k]
    total_sec = time.perf_counter() - total_started
    context_chars = sum(len(normalize_context_text(hit.document)) for hit in final_hits)
    print(
        "[검색 시간] "
        f"total={total_sec:.3f}초 "
        f"embed={embed_sec:.3f}초 "
        f"dense={dense_sec:.3f}초({len(dense_hits)}건) "
        f"bm25={bm25_sec:.3f}초({len(bm25_hits)}건) "
        f"rrf={rrf_sec:.3f}초({len(rrf_hits)}건) "
        f"rerank={rerank_sec:.3f}초 "
        f"final={len(final_hits)}건 "
        f"context_chars={context_chars}"
    )
    return final_hits


def main() -> int:
    configure_output_encoding()
    load_dotenv()
    args = parse_args()

    embedder = make_embedder(
        provider=args.embedding_provider,
        model=args.embedding_model,
        dimensions=args.dimensions,
        device=args.device,
        openai_api_key_env=args.openai_api_key_env,
        hf_token_env=args.hf_token_env,
        embedding_backend=args.embedding_backend,
    )
    config = config_from_env(args.qdrant_url, args.qdrant_api_key, args.collection, args.prefer_grpc)
    client = make_client(config)
    collection = config.collection
    records = load_corpus(client, collection)
    bm25 = BM25Okapi([tokenize(retrieval_text(record)) for record in records])

    print("[검색 파이프라인] Dense + BM25 -> RRF -> " + ("Rerank" if not args.skip_rerank else "RRF final"))
    print(f"[임베딩] provider={embedder.provider} model={embedder.model} device={embedder.device or 'remote/api'} dims={embedder.dimensions or 'model default'}")

    if args.news_mode != "off" and (args.news_mode == "always" or is_news_intent(args.query)):
        final_hits = []
        news_items = maybe_search_news(args)
    else:
        final_hits = retrieve(args, embedder, client, collection, records, bm25)
        news_items = maybe_search_news(args, final_hits)
    answer_hits = [] if news_items else final_hits
    if not args.no_answer:
        answer_client = make_answer_client(args)
        print(f"[답변 모델] provider={args.answer_provider} model={args.answer_model}")
        print("\n[답변]")
        print(
            run_with_max_new_tokens(
                answer_client,
                args.news_answer_max_new_tokens if news_items else None,
                lambda: generate_answer(
                    answer_client,
                    args.answer_model,
                    args.query,
                    answer_hits,
                    args.max_context_chars,
                    args.temperature,
                    news_items,
                    args.news_max_context_chars,
                ),
            )
        )

    if news_items:
        print("\n[Naver 뉴스 결과]")
        for ref in news_source_references(news_items):
            print(ref)
    else:
        print_hits(final_hits, args.preview_chars, args.hide_context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
