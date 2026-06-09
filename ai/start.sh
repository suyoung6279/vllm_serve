#!/bin/bash
set -e

echo "▶ vLLM 서버 시작 중..."
vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --trust-remote-code \
    --quantization compressed-tensors \
    --served-model-name "$MODEL_NAME" \
    --max-model-len 8192 &

VLLM_PID=$!

echo "⏳ vLLM 준비 대기 중..."
until curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; do
    sleep 5
    echo "  아직 대기 중..."
done
echo "✅ vLLM 준비 완료"

echo "▶ FastAPI 서버 시작 (포트 8501)..."
uvicorn server_vllm:app --host 0.0.0.0 --port 8501

wait $VLLM_PID
