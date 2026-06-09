from __future__ import annotations

import os
import tempfile
import time
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from config import LOAD_IN_4BIT, OCR_MODEL_ID, PRELOAD_OCR_MODEL
from model_runtime import cleanup_cuda, load_ocr_model, process_vision_info
from schemas import TextResponse


app = FastAPI(title="HPM OCR Server", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "ocr",
        "ocr_model": OCR_MODEL_ID,
        "load_in_4bit": LOAD_IN_4BIT,
        "preload_ocr_model": PRELOAD_OCR_MODEL,
        "torch_cuda": torch.cuda.is_available(),
    }


@app.on_event("startup")
def preload_models() -> None:
    if PRELOAD_OCR_MODEL:
        load_ocr_model()


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

    return TextResponse(result={"text": extracted.strip()}, elapsed_sec=round(time.perf_counter() - started, 3), model=OCR_MODEL_ID)
