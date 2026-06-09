from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from config import (
    CHUNK_OVERLAP,
    CHUNK_SCRIPT_PATH,
    CHUNK_SIZE,
    CHUNK_VERSION,
    EMBEDDING_MODEL_ID,
    FEATURE_CHAT_DIR,
    LLAMA_N_GPU_LAYERS,
    OCR_MODEL_ID,
    PARSER_SCRIPT_PATH,
    PRELOAD_OCR_MODEL,
    PRELOAD_TEXT_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    STT_BATCH_SIZE,
    STT_ENABLE_ALIGN,
    STT_ENABLE_DIARIZE,
    STT_MODEL_ID,
    STT_PRELOAD_MODEL,
    TEXT_BACKEND,
    TEXT_GGUF_FILENAME,
    TEXT_GGUF_REPO,
    TEXT_MODEL_ID,
)
from document_ingest import file_sha256, ingest_pdf_chunks
from meeting_ingest import ingest_meeting_minutes
from model_runtime import cleanup_cuda, generate_json, load_ocr_model, load_stt_model, load_text_model, process_vision_info, transcribe_audio_file
from news import search_preparation_news
from prompts import agenda_messages, chat_messages, legacy_preparation_messages, minutes_messages, preparation_messages
from retrieval import feature_chat_retrieve, load_feature_chat_rag, select_preparation_documents
from schemas import AgendaRequest, ChatRequest, MinutesRequest, PreparationRequest, TextResponse


app = FastAPI(title="HPM AI Pipeline Server", version="2.0.0")


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


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "text_backend": TEXT_BACKEND,
        "text_model": TEXT_MODEL_ID,
        "text_gguf_repo": TEXT_GGUF_REPO,
        "text_gguf_filename": TEXT_GGUF_FILENAME,
        "ocr_model": OCR_MODEL_ID,
        "stt_model": STT_MODEL_ID,
        "torch_cuda": torch.cuda.is_available(),
        "llama_n_gpu_layers": LLAMA_N_GPU_LAYERS,
        "preload_text_model": PRELOAD_TEXT_MODEL,
        "preload_ocr_model": PRELOAD_OCR_MODEL,
        "preload_stt_model": STT_PRELOAD_MODEL,
        "qdrant_collection": QDRANT_COLLECTION,
        "qdrant_url": QDRANT_URL,
        "embedding_model": EMBEDDING_MODEL_ID,
        "parser_script": str(PARSER_SCRIPT_PATH),
        "chunk_script": str(CHUNK_SCRIPT_PATH),
        "feature_chat_dir": str(FEATURE_CHAT_DIR),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_version": CHUNK_VERSION,
    }


@app.on_event("startup")
def preload_models() -> None:
    if PRELOAD_TEXT_MODEL:
        load_text_model()
    if PRELOAD_OCR_MODEL:
        load_ocr_model()
    if STT_PRELOAD_MODEL:
        load_stt_model()


@app.post("/documents/ingest", response_model=TextResponse)
async def ingest_documents(
    files: list[UploadFile] = File(...),
    title: str = Form(""),
    meeting_topic: str = Form(""),
    meeting_datetime: str = Form(""),
    document_type: str = Form("internal_document"),
    document_id: str = Form(""),
    project_id: str = Form(""),
    storage_key: str = Form(""),
    s3_key: str = Form(""),
    s3_url: str = Form(""),
) -> TextResponse:
    started = time.perf_counter()
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF file.")

    with tempfile.TemporaryDirectory(prefix="hpm_ingest_") as tmp:
        root = Path(tmp)
        pdf_dir = root / "pdf_data"
        parsed_dir = root / "parsed"
        chunks_dir = root / "chunks"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        saved_files: list[dict[str, Any]] = []
        for index, file in enumerate(files, 1):
            filename = file.filename or f"upload_{index}.pdf"
            if not filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {filename}")
            safe_path = pdf_dir / f"input_{index:03d}.pdf"
            content = await file.read()
            safe_path.write_bytes(content)
            saved_files.append(
                {
                    "original_filename": filename,
                    "safe_filename": safe_path.name,
                    "sha256": file_sha256(safe_path),
                    "bytes": len(content),
                }
            )

        try:
            metadata = {
                "source_type": document_type,
                "doc_type": document_type,
                "document_id": document_id,
                "project_id": project_id,
                "viewer_type": "document",
                "viewer_id": document_id,
                "storage_key": storage_key,
                "s3_key": s3_key,
                "s3_url": s3_url,
                "title": title or meeting_topic or "internal_document",
                "meeting_topic": meeting_topic or title or "미정",
                "meeting_datetime": meeting_datetime or "미정",
                "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            ingest_info = ingest_pdf_chunks(pdf_dir, parsed_dir, chunks_dir, metadata)
        except Exception as exc:
            cleanup_cuda()
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TextResponse(
        result={
            "uploaded_files": saved_files,
            "parser": ingest_info["parser"],
            "chunker": ingest_info["chunker"],
            "chunk_count": len(ingest_info["chunks"]),
            "upserted_points": ingest_info["upserted"],
            "qdrant_collection": QDRANT_COLLECTION,
            "embedding_model": EMBEDDING_MODEL_ID,
        },
        elapsed_sec=round(time.perf_counter() - started, 3),
        model=f"mineru+{EMBEDDING_MODEL_ID}",
    )


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
    try:
        selected: dict[str, Any] = {
            "query": " ".join(
                part.strip()
                for part in [
                    req.title,
                    req.project_context,
                    " ".join(f"{item.get('name', '')} {item.get('work', '')}" for item in req.participants),
                    " ".join(req.agendas),
                ]
                if part and part.strip()
            ),
            "news_query": " ".join(part.strip() for part in [req.title, " ".join(req.agendas)] if part and part.strip()),
            "previous_meetings": [],
            "internal_documents": [],
        }
        retrieval_error = ""
        try:
            selected = select_preparation_documents(req)
        except Exception as exc:
            retrieval_error = f"{type(exc).__name__}: {exc}"

        searched_news = search_preparation_news(selected.get("news_query") or selected["query"])
        previous_meetings = selected.get("previous_meetings", [])
        internal_documents = selected.get("internal_documents", [])
        external_news = searched_news
        selected_documents = {
            "query": selected["query"],
            "previous_meetings": previous_meetings,
            "internal_documents": internal_documents,
            "external_news": external_news,
        }
        result = generate_json(preparation_messages(req, selected_documents))
    except Exception as exc:
        cleanup_cuda()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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


@app.post("/generate-preparation-legacy", response_model=TextResponse)
def generate_preparation_legacy(req: PreparationRequest) -> TextResponse:
    started = time.perf_counter()
    try:
        result = generate_json(legacy_preparation_messages(req))
    except Exception as exc:
        cleanup_cuda()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
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


@app.post("/ocr", response_model=TextResponse)
async def ocr(file: UploadFile = File(...)) -> TextResponse:
    started = time.perf_counter()
    bundle = load_ocr_model()
    processor = bundle.processor
    model = bundle.model

    suffix = os.path.splitext(file.filename or "")[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": tmp_path},
                    {"type": "text", "text": "이미지 또는 문서 페이지의 모든 텍스트를 한국어 원문 중심으로 추출해줘."},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=2048)
        generated = generated[:, inputs.input_ids.shape[-1] :]
        extracted = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    except Exception as exc:
        cleanup_cuda()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return TextResponse(
        result={"text": extracted.strip()},
        elapsed_sec=round(time.perf_counter() - started, 3),
        model=OCR_MODEL_ID,
    )


@app.post("/stt", response_model=TextResponse)
@app.post("/transcribe", response_model=TextResponse)
async def stt(
    file: UploadFile = File(...),
    language: str = Form("ko"),
    align: bool = Form(STT_ENABLE_ALIGN),
    diarize: bool = Form(STT_ENABLE_DIARIZE),
    batch_size: int = Form(STT_BATCH_SIZE),
) -> TextResponse:
    started = time.perf_counter()
    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = transcribe_audio_file(
            tmp_path,
            language=language or None,
            batch_size=batch_size,
            align=align,
            diarize=diarize,
            hf_token=os.getenv("HF_TOKEN"),
        )
    except Exception as exc:
        cleanup_cuda()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return TextResponse(
        result=result,
        elapsed_sec=round(time.perf_counter() - started, 3),
        model=f"whisperx/{STT_MODEL_ID}",
    )
