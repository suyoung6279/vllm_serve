# RunPod AI Servers

## Recommended 3-Pod Demo Layout

Use separate pods for stability:

```text
Pod A: Core LLM/RAG
  file: core_server.py
  endpoints: /generate-agendas, /generate-preparation, /generate-minutes, /chat
  install: requirements-core-vllm.txt

Pod B: STT
  file: stt_server.py
  endpoints: /stt, /transcribe
  install: requirements-stt.txt

Pod C: OCR
  file: ocr_server.py
  endpoints: /ocr
  install: requirements-ocr.txt
```

Local Django `.env` should point to each proxy:

```env
RUNPOD_CORE_BASE_URL="https://core-proxy-8501.proxy.runpod.net"
RUNPOD_STT_BASE_URL="https://stt-proxy-8501.proxy.runpod.net"
RUNPOD_OCR_BASE_URL="https://ocr-proxy-8501.proxy.runpod.net"
```

`RUNPOD_BASE_URL` is kept as a fallback only.

## Pod A: Core

Run vLLM and the FastAPI core server as separate processes in the same pod.

For Gemma 4 Unified models, use a vLLM image that includes Gemma 4 support:

```text
vllm/vllm-openai:gemma4
```

Install core dependencies and the isolated vLLM dependencies:

```bash
cd /workspace/final_1team/runpod
pip install -U pip
pip install --no-cache-dir -r requirements-core-vllm.txt
```

Check that the vLLM image dependencies were not replaced:

```bash
python -c "import torch, vllm; print('torch', torch.__version__); print('vllm', vllm.__version__)"
```

Start the vLLM OpenAI-compatible server:

```bash
vllm serve cyankiwi/gemma-4-12B-it-AWQ-INT4 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --trust-remote-code \
  --quantization compressed-tensors \
  --served-model-name cyankiwi/gemma-4-12B-it-AWQ-INT4
```

Start the core API server:

```bash
cd /workspace/final_1team/runpod
export TEXT_BACKEND=vllm
export TEXT_MODEL_ID=cyankiwi/gemma-4-12B-it-AWQ-INT4
export TEXT_VLLM_BASE_URL=http://127.0.0.1:8000/v1
uvicorn core_server:app --host 0.0.0.0 --port 8501
```

## Pod B: STT

```bash
cd /workspace/final_1team/runpod
pip install -U pip
pip install --no-cache-dir -r requirements-stt.txt
apt-get update && apt-get install -y ffmpeg
uvicorn stt_server:app --host 0.0.0.0 --port 8501
```

## Pod C: OCR

```bash
cd /workspace/final_1team/runpod
pip install -U pip
pip install --no-cache-dir -r requirements-ocr.txt
uvicorn ocr_server:app --host 0.0.0.0 --port 8501
```

Each pod should respond to:

```bash
curl http://127.0.0.1:8501/health
```

## Full Combined Server

`model_server.py` still contains the combined server. Use it only when the GPU memory and dependencies are known to be stable.

```bash
pip install --no-cache-dir -r requirements.txt
uvicorn model_server:app --host 0.0.0.0 --port 8501
```
