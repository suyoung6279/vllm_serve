from __future__ import annotations

import os
import time
from typing import Any

import torch
from fastapi import FastAPI, HTTPException

from config import (
    EMBEDDING_MODEL_ID,
    FEATURE_CHAT_DIR,
    LLAMA_N_GPU_LAYERS,
    PRELOAD_TEXT_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    TEXT_BACKEND,
    TEXT_GGUF_FILENAME,
    TEXT_GGUF_REPO,
    TEXT_MODEL_ID,
)
from meeting_ingest import ingest_meeting_minutes
from model_runtime import cleanup_cuda, generate_json, load_text_model
from news import search_preparation_news
from prompts import agenda_messages, chat_messages, minutes_messages, preparation_messages
from retrieval import feature_chat_retrieve, load_feature_chat_rag, select_preparation_documents
from schemas import AgendaRequest, ChatRequest, MinutesRequest, PreparationRequest, TextResponse


app = FastAPI(title="HPM Core LLM/RAG Server", version="1.0.0")


def compact_documents(items: list[dict[str, Any]], *, limit: int = 3, text_chars: int = 700) -> list[dict[str, Any]]:
    compacted = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        compacted.append(
            {
                "source": str(item.get("source") or item.get("title") or "unknown")[:180],
                "text": str(item.get("text") or item.get("content") or "")[:text_chars],
                "category": str(item.get("category") or item.get("source_type") or "")[:80],
                "url": str(item.get("url") or item.get("link") or "")[:500],
                "chunk_id": str(item.get("chunk_id") or "")[:180],
                "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
            }
        )
    return compacted


def preparation_sources(selected_documents: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for category in ["previous_meetings", "internal_documents", "external_news"]:
        for item in selected_documents.get(category, []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("source") or item.get("title") or "unknown")
            url = str(item.get("url") or item.get("link") or "")
            chunk_id = str(item.get("chunk_id") or "")
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            viewer_type = str(metadata.get("viewer_type") or "")
            viewer_id = str(metadata.get("viewer_id") or "")
            if not viewer_type:
                if category == "previous_meetings":
                    viewer_type = "meeting_minutes"
                    viewer_id = str(metadata.get("meeting_id") or "")
                elif category == "internal_documents":
                    viewer_type = "document"
                    viewer_id = str(metadata.get("document_id") or "")
                elif category == "external_news":
                    viewer_type = "external_url"
            viewer: dict[str, Any] = {"type": viewer_type}
            if viewer_id:
                viewer["id"] = viewer_id
            if metadata.get("project_id"):
                viewer["project_id"] = str(metadata.get("project_id"))
            page = metadata.get("page") or metadata.get("page_number") or metadata.get("page_no") or metadata.get("page_idx")
            if page is not None:
                viewer["page"] = page
            if url:
                viewer["url"] = url
            key = (category, label, url or chunk_id)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "label": label,
                    "category": str(item.get("category") or category),
                    "chunk_id": chunk_id,
                    "viewer": viewer,
                    "metadata": metadata,
                }
            )
    return sources


def simple_preparation_result(
    req: PreparationRequest,
    selected_documents: dict[str, Any],
    *,
    error: str = "",
) -> dict[str, Any]:
    agendas = [str(item) for item in req.agendas if str(item).strip()]
    agenda_text = "\n".join(f"- {item}" for item in agendas) or "- 등록된 안건이 없습니다."
    participants = [
        f"- {item.get('name', '미정')} / 역할: {item.get('work', '미정')}"
        for item in req.participants
        if isinstance(item, dict)
    ]
    participant_text = "\n".join(participants) or "- 미정"
    status_text = f"\n\n## 생성 상태\n- 자동 생성 오류: {error}" if error else ""
    text = (
        f"# {req.title or '회의'} 준비자료\n\n"
        f"## 참석자\n{participant_text}\n\n"
        f"## 안건\n{agenda_text}\n\n"
        f"## 준비 체크포인트\n"
        f"- 안건별로 결정해야 할 항목을 사전에 정리합니다.\n"
        f"- 참석자별 확인 자료와 예상 질문을 준비합니다."
        f"{status_text}"
    )
    return {
        "text": text,
        "sources": preparation_sources(selected_documents),
    }


def fallback_preparation_result(
    req: PreparationRequest,
    selected_documents: dict[str, Any],
    *,
    error: str,
) -> dict[str, Any]:
    meeting_datetime = getattr(req, "meeting_datetime", "") or "미정"
    location = getattr(req, "location", "") or "미정"
    agendas = [str(item) for item in req.agendas if str(item).strip()]
    agenda_text = "\n".join(f"- {item}" for item in agendas) or "- 등록된 안건이 없습니다."
    participants = [
        f"- {item.get('name', '미정')} / 직무: {item.get('work', '미정')}"
        for item in req.participants
        if isinstance(item, dict)
    ]
    participant_text = "\n".join(participants) or "- 미정"
    source_count = sum(
        len(selected_documents.get(key, []))
        for key in ["previous_meetings", "internal_documents", "external_news"]
    )
    document = (
        f"# {req.title or '회의'} 준비자료\n\n"
        f"## 회의 정보\n"
        f"- 일시: {meeting_datetime}\n"
        f"- 장소: {location}\n\n"
        f"## 참석자\n{participant_text}\n\n"
        f"## 안건\n{agenda_text}\n\n"
        f"## 준비 체크포인트\n"
        f"- 안건별 목표와 결정해야 할 항목을 사전에 정리합니다.\n"
        f"- 참석자별 사전 확인 자료와 예상 질문을 준비합니다.\n"
        f"- 회의 후 산출물, 담당자, 마감일을 기록할 수 있도록 준비합니다.\n\n"
        f"## 참고자료 상태\n"
        f"- 선택된 참고자료 수: {source_count}\n"
        f"- 자동 생성 오류: {error}"
    )
    return {
        "document": document,
        "sections": [
            {"title": "회의 정보", "content": f"일시: {meeting_datetime}\n장소: {location}", "sources": []},
            {"title": "안건", "content": agenda_text, "sources": []},
            {"title": "준비 체크포인트", "content": "안건별 목표, 결정 항목, 참석자별 사전 확인 자료를 준비합니다.", "sources": []},
        ],
        "selected_documents": selected_documents,
        "source_map": [],
        "fallback_error": error,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "core",
        "text_backend": TEXT_BACKEND,
        "text_model": TEXT_MODEL_ID,
        "text_gguf_repo": TEXT_GGUF_REPO,
        "text_gguf_filename": TEXT_GGUF_FILENAME,
        "torch_cuda": torch.cuda.is_available(),
        "llama_n_gpu_layers": LLAMA_N_GPU_LAYERS,
        "preload_text_model": PRELOAD_TEXT_MODEL,
        "qdrant_collection": QDRANT_COLLECTION,
        "qdrant_url": QDRANT_URL,
        "embedding_model": EMBEDDING_MODEL_ID,
        "feature_chat_dir": str(FEATURE_CHAT_DIR),
    }


@app.on_event("startup")
def preload_models() -> None:
    if PRELOAD_TEXT_MODEL:
        load_text_model()


@app.post("/generate-minutes", response_model=TextResponse)
def generate_minutes(req: MinutesRequest) -> TextResponse:
    started = time.perf_counter()
    try:
        result = generate_json(minutes_messages(req))
        if isinstance(result, dict):
            try:
                result["qdrant_ingest"] = ingest_meeting_minutes(req, result)
            except Exception as ingest_exc:
                result["qdrant_ingest_error"] = str(ingest_exc)
    except Exception as exc:
        cleanup_cuda()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return TextResponse(result=result, elapsed_sec=round(time.perf_counter() - started, 3), model=TEXT_MODEL_ID)


@app.post("/generate-agendas", response_model=TextResponse)
def generate_agendas(req: AgendaRequest) -> TextResponse:
    started = time.perf_counter()
    try:
        result = generate_json(agenda_messages(req), max_new_tokens=1024)
        if req.title.strip() and not result.get("agendas"):
            result = generate_json(agenda_messages(req, retry=True), max_new_tokens=1024)
    except Exception as exc:
        cleanup_cuda()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return TextResponse(result=result, elapsed_sec=round(time.perf_counter() - started, 3), model=TEXT_MODEL_ID)


@app.post("/generate-preparation", response_model=TextResponse)
def generate_preparation(req: PreparationRequest) -> TextResponse:
    started = time.perf_counter()
    selected = {"query": req.title, "news_query": req.title, "previous_meetings": [], "internal_documents": []}
    previous_meetings: list[dict[str, Any]] = []
    internal_documents: list[dict[str, Any]] = []
    external_news: list[dict[str, Any]] = []
    retrieval_error = ""
    news_error = ""
    try:
        try:
            selected = select_preparation_documents(req)
            retrieval_error = ""
        except Exception as retrieval_exc:
            selected = {
                "query": " ".join(part for part in [req.title, req.project_context, " ".join(req.agendas)] if part),
                "news_query": req.title,
                "previous_meetings": [],
                "internal_documents": [],
            }
            retrieval_error = str(retrieval_exc)

        try:
            searched_news = search_preparation_news(selected.get("news_query") or selected["query"])
            news_error = ""
        except Exception as news_exc:
            searched_news = []
            news_error = str(news_exc)

        previous_meetings = compact_documents(selected["previous_meetings"])
        internal_documents = compact_documents(selected["internal_documents"])
        external_news = compact_documents(searched_news)
        selected_documents = {
            "query": selected["query"],
            "previous_meetings": previous_meetings,
            "internal_documents": internal_documents,
            "external_news": external_news,
        }
        try:
            result = generate_json(
                preparation_messages(req, selected_documents),
                max_new_tokens=int(os.getenv("PREPARATION_MAX_NEW_TOKENS", "512")),
            )
        except Exception as generation_exc:
            result = simple_preparation_result(req, selected_documents, error=str(generation_exc))
    except Exception as exc:
        cleanup_cuda()
        selected_documents = {"query": req.title, "previous_meetings": [], "internal_documents": [], "external_news": []}
        result = simple_preparation_result(req, selected_documents, error=str(exc))
        retrieval_error = str(exc)
        news_error = ""

    if isinstance(result, dict):
        result = {
            "text": str(result.get("text") or result.get("document") or ""),
            "sources": preparation_sources(
                {
                    "previous_meetings": previous_meetings,
                    "internal_documents": internal_documents,
                    "external_news": external_news,
                }
            ),
        }
    return TextResponse(result=result, elapsed_sec=round(time.perf_counter() - started, 3), model=TEXT_MODEL_ID)


@app.post("/chat", response_model=TextResponse)
def chat(req: ChatRequest) -> TextResponse:
    started = time.perf_counter()
    context = req.context.strip()
    sources = list(req.sources)
    rag_info: dict[str, Any] | None = None
    if not context:
        try:
            rag_info = feature_chat_retrieve(req.question)
            context = str(rag_info.get("context") or "")
            sources = list(rag_info.get("sources") or [])
        except Exception as exc:
            cleanup_cuda()
            raise HTTPException(status_code=500, detail=f"Feature chat Qdrant search failed: {exc}") from exc
    if not context:
        raise HTTPException(status_code=404, detail="Qdrant search returned 0 context chunks.")

    try:
        result = generate_json(chat_messages(req, context, sources), max_new_tokens=1024)
    except Exception as exc:
        cleanup_cuda()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if isinstance(result, dict) and rag_info is not None:
        result.setdefault("rag_hit_count", rag_info.get("hit_count", 0))
        result.setdefault("rag_collection", rag_info.get("collection"))
    return TextResponse(result=result, elapsed_sec=round(time.perf_counter() - started, 3), model=TEXT_MODEL_ID)
