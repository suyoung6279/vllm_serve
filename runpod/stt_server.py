from __future__ import annotations

import os
import tempfile
import time
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from config import (
    STT_BATCH_SIZE,
    STT_COMPUTE_TYPE,
    STT_DEVICE,
    STT_ENABLE_ALIGN,
    STT_ENABLE_DIARIZE,
    STT_MODEL_ID,
    STT_PRELOAD_MODEL,
)
from model_runtime import cleanup_cuda, load_stt_model, transcribe_audio_file
from schemas import TextResponse


app = FastAPI(title="HPM WhisperX STT Server", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "stt",
        "stt_model": STT_MODEL_ID,
        "stt_device": STT_DEVICE,
        "stt_compute_type": STT_COMPUTE_TYPE,
        "stt_batch_size": STT_BATCH_SIZE,
        "stt_align": STT_ENABLE_ALIGN,
        "stt_diarize": STT_ENABLE_DIARIZE,
        "preload_stt_model": STT_PRELOAD_MODEL,
        "torch_cuda": torch.cuda.is_available(),
    }


@app.on_event("startup")
def preload_models() -> None:
    if STT_PRELOAD_MODEL:
        load_stt_model()


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

    return TextResponse(result=result, elapsed_sec=round(time.perf_counter() - started, 3), model=f"whisperx/{STT_MODEL_ID}")
