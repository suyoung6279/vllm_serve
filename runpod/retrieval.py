from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

from config import FEATURE_CHAT_DIR
from schemas import PreparationRequest


@dataclass
class FeatureChatRagBundle:
    query_module: Any
    embedder: Any
    client: Any
    collection: str
    records: list[Any]
    bm25: Any


feature_chat_rag_bundle: FeatureChatRagBundle | None = None


def load_feature_chat_rag() -> FeatureChatRagBundle:
    global feature_chat_rag_bundle
    if feature_chat_rag_bundle is not None:
        return feature_chat_rag_bundle

    if not FEATURE_CHAT_DIR.exists():
        raise RuntimeError(f"Feature chat RAG directory was not found: {FEATURE_CHAT_DIR}")
    module_path = FEATURE_CHAT_DIR / "rag_search.py"
    if not module_path.exists():
        legacy_path = FEATURE_CHAT_DIR / "04_query_qdrant_openai.py"
        module_path = legacy_path if legacy_path.exists() else module_path
    if not module_path.exists():
        raise RuntimeError(f"RAG search module was not found: {module_path}")

    if str(FEATURE_CHAT_DIR) not in sys.path:
        sys.path.insert(0, str(FEATURE_CHAT_DIR))

    import importlib.util
    from rank_bm25 import BM25Okapi
    from rag.embeddings import DEFAULT_HF_EMBEDDING_MODEL, make_embedder
    from rag.qdrant_store import config_from_env, make_client

    spec = importlib.util.spec_from_file_location("runpod_feature_chat_query", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    query_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = query_module
    spec.loader.exec_module(query_module)

    dimensions = os.getenv("RAG_EMBEDDING_DIMENSIONS")
    embedder = make_embedder(
        provider=os.getenv("RAG_EMBEDDING_PROVIDER", "huggingface"),
        model=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_HF_EMBEDDING_MODEL),
        dimensions=int(dimensions) if dimensions else None,
        device=os.getenv("RAG_EMBEDDING_DEVICE", "auto"),
        embedding_backend=os.getenv("RAG_HF_EMBEDDING_BACKEND", "transformers"),
    )
    config = config_from_env(
        os.getenv("QDRANT_URL"),
        os.getenv("QDRANT_API_KEY"),
        os.getenv("QDRANT_COLLECTION"),
        os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true",
    )
    client = make_client(config)
    records = query_module.load_corpus(client, config.collection)
    bm25 = BM25Okapi([query_module.tokenize(query_module.retrieval_text(record)) for record in records])

    feature_chat_rag_bundle = FeatureChatRagBundle(
        query_module=query_module,
        embedder=embedder,
        client=client,
        collection=config.collection,
        records=records,
        bm25=bm25,
    )
    return feature_chat_rag_bundle


def _search_args(module: Any, question: str, *, top_k: int, prefix: str = "RAG") -> Any:
    return module.argparse.Namespace(
        query=question,
        top_k=top_k,
        dense_k=int(os.getenv(f"{prefix}_DENSE_K", str(max(20, top_k * 4)))),
        dense_fetch_k=int(os.getenv(f"{prefix}_DENSE_FETCH_K", str(max(50, top_k * 8)))),
        bm25_k=int(os.getenv(f"{prefix}_BM25_K", str(max(20, top_k * 4)))),
        rrf_k=int(os.getenv("RAG_RRF_K", "60")),
        rrf_top_k=int(os.getenv(f"{prefix}_RRF_TOP_K", str(max(20, top_k * 4)))),
        rerank_top_n=int(os.getenv("RAG_RERANK_TOP_N", str(max(10, top_k * 2)))),
        where_doc_id=os.getenv("RAG_WHERE_DOC_ID") or None,
        where_source_contains=os.getenv("RAG_INTERNAL_DOCUMENT_SOURCE_CONTAINS") if prefix == "RAG" else None,
        exclude_chunk_types=os.getenv("RAG_EXCLUDE_CHUNK_TYPES", "image_caption,chart_caption"),
        skip_rerank=os.getenv("RAG_SKIP_RERANK", "true").lower() == "true",
        cohere_model=os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"),
        max_tokens_per_doc=int(os.getenv("RAG_MAX_TOKENS_PER_DOC", "4096")),
    )


def feature_chat_retrieve(question: str) -> dict[str, Any]:
    bundle = load_feature_chat_rag()
    module = bundle.query_module
    top_k = int(os.getenv("CHAT_RAG_TOP_K", "5"))
    args = _search_args(module, question, top_k=top_k)
    hits = module.retrieve(args, bundle.embedder, bundle.client, bundle.collection, bundle.records, bundle.bm25)
    return {
        "context": module.build_context(hits, int(os.getenv("CHAT_RAG_MAX_CONTEXT_CHARS", "9000"))),
        "sources": module.answer_source_references(hits),
        "hit_count": len(hits),
        "collection": bundle.collection,
    }


def preparation_query(req: PreparationRequest) -> str:
    participant_text = " ".join(
        f"{item.get('name', '')} {item.get('work', '')}" if isinstance(item, dict) else str(item)
        for item in req.participants
    )
    agenda_text = " ".join(str(item) for item in req.agendas)
    return " ".join(
        part.strip()
        for part in [req.title, getattr(req, "project_context", ""), participant_text, agenda_text]
        if part and part.strip()
    )


def preparation_news_query(req: PreparationRequest) -> str:
    agenda_text = " ".join(str(item) for item in req.agendas)
    return " ".join(
        part.strip()
        for part in [req.title, agenda_text]
        if part and part.strip()
    )


def feature_chat_search_hits(question: str, *, top_k: int) -> list[Any]:
    bundle = load_feature_chat_rag()
    module = bundle.query_module
    args = _search_args(module, question, top_k=top_k, prefix="PREPARATION_RAG")
    return module.retrieve(args, bundle.embedder, bundle.client, bundle.collection, bundle.records, bundle.bm25)


def hit_document_type(hit: Any) -> str:
    meta = getattr(hit, "metadata", {}) if isinstance(getattr(hit, "metadata", {}), dict) else {}
    values = [meta.get("source_type"), meta.get("doc_type"), meta.get("document_type"), meta.get("category")]
    text = " ".join(str(value or "").lower() for value in values)
    if "previous" in text or "meeting_minutes" in text or "previous_meeting" in text:
        return "previous_meeting"
    if "internal" in text or "internal_document" in text:
        return "internal_document"
    return ""


def hit_to_preparation_document(hit: Any, category: str) -> dict[str, Any]:
    module = load_feature_chat_rag().query_module
    meta = getattr(hit, "metadata", {}) if isinstance(getattr(hit, "metadata", {}), dict) else {}
    text = module.normalize_context_text(str(getattr(hit, "document", "") or ""))
    return {
        "category": category,
        "source": module.source_label(meta),
        "text": text[: int(os.getenv("PREPARATION_DOC_MAX_CHARS", "1200"))],
        "chunk_id": str(getattr(hit, "chunk_id", "")),
        "rank": int(getattr(hit, "rank", 0) or 0),
        "score": float(getattr(hit, "score", 0.0) or 0.0),
        "metadata": {
            key: meta.get(key)
            for key in [
                "source_type",
                "doc_type",
                "document_type",
                "category",
                "project_id",
                "meeting_id",
                "meeting_topic",
                "meeting_datetime",
                "document_id",
                "viewer_type",
                "viewer_id",
                "storage_key",
                "s3_key",
                "s3_url",
                "file_name",
                "source_filename",
                "page",
                "page_idx",
                "page_no",
                "page_number",
                "page_start",
                "page_end",
            ]
            if meta.get(key) is not None
        },
    }


def select_preparation_documents(req: PreparationRequest) -> dict[str, Any]:
    base_query = preparation_query(req)
    top_k = int(os.getenv("PREPARATION_RAG_TOP_K", "5"))
    fetch_k = int(os.getenv("PREPARATION_RAG_FETCH_K", str(max(20, top_k * 6))))
    previous_hits = feature_chat_search_hits(f"{base_query} 이전 회의 회의록 결정사항 논의내용", top_k=fetch_k) if base_query else []
    internal_hits = feature_chat_search_hits(f"{base_query} 내부 문서 요구사항 정책 참고자료", top_k=fetch_k) if base_query else []

    docs: dict[str, list[dict[str, Any]]] = {"previous_meetings": [], "internal_documents": []}
    seen = {"previous_meetings": set(), "internal_documents": set()}
    for hit, output_key, category in [
        *[(hit, "previous_meetings", "previous_meeting") for hit in previous_hits],
        *[(hit, "internal_documents", "internal_document") for hit in internal_hits],
    ]:
        hit_type = hit_document_type(hit)
        if hit_type and hit_type != category:
            continue
        chunk_id = str(getattr(hit, "chunk_id", ""))
        if chunk_id in seen[output_key] or len(docs[output_key]) >= top_k:
            continue
        seen[output_key].add(chunk_id)
        docs[output_key].append(hit_to_preparation_document(hit, category))

    return {"query": base_query, "news_query": preparation_news_query(req), **docs}
