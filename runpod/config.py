from __future__ import annotations

import os
import importlib.util
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent


def load_env_files() -> None:
    candidates = [
        APP_DIR / ".env",
        ROOT_DIR / ".env",
        Path("/workspace/final_1team/runpod/.env"),
        Path("/workspace/final_1team/.env"),
        Path("/workspace/runpod/.env"),
        Path("/workspace/.env"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_files()

if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") == "1" and importlib.util.find_spec("hf_transfer") is None:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


TEXT_BACKEND = os.getenv("TEXT_BACKEND", "vllm")
TEXT_GGUF_REPO = os.getenv("TEXT_GGUF_REPO", "unsloth/gemma-4-12B-it-qat-GGUF")
TEXT_GGUF_FILENAME = os.getenv("TEXT_GGUF_FILENAME", "*Q4*.gguf")
DEFAULT_TEXT_MODEL_ID = TEXT_GGUF_REPO if TEXT_BACKEND == "llama_cpp" else "cyankiwi/gemma-4-12B-it-AWQ-INT4"
TEXT_MODEL_ID = os.getenv("TEXT_MODEL_ID", DEFAULT_TEXT_MODEL_ID)
TEXT_VLLM_BASE_URL = os.getenv("TEXT_VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
TEXT_VLLM_API_KEY = os.getenv("TEXT_VLLM_API_KEY", "EMPTY")
TEXT_VLLM_TIMEOUT_SEC = float(os.getenv("TEXT_VLLM_TIMEOUT_SEC", "300"))
OCR_MODEL_ID = os.getenv("OCR_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
STT_MODEL_ID = os.getenv("STT_MODEL_ID", "large-v3")
STT_DEVICE = os.getenv("STT_DEVICE", "cuda")
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "float16")
STT_BATCH_SIZE = int(os.getenv("STT_BATCH_SIZE", "16"))
STT_PRELOAD_MODEL = os.getenv("STT_PRELOAD_MODEL", "false").lower() == "true"
STT_ENABLE_ALIGN = os.getenv("STT_ENABLE_ALIGN", "true").lower() == "true"
STT_ENABLE_DIARIZE = os.getenv("STT_ENABLE_DIARIZE", "false").lower() == "true"

LLAMA_N_CTX = int(os.getenv("LLAMA_N_CTX", "32768"))
LLAMA_N_GPU_LAYERS = int(os.getenv("LLAMA_N_GPU_LAYERS", "-1"))
LLAMA_N_THREADS = int(os.getenv("LLAMA_N_THREADS", "8"))
LLAMA_VERBOSE = os.getenv("LLAMA_VERBOSE", "false").lower() == "true"

LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "auto")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))
PRELOAD_TEXT_MODEL = os.getenv("PRELOAD_TEXT_MODEL", "true").lower() == "true"
PRELOAD_OCR_MODEL = os.getenv("PRELOAD_OCR_MODEL", "false").lower() == "true"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mineru_pdf_chunks_ko_sroberta")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "jhgan/ko-sroberta-multitask")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "650"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
CHUNK_VERSION = os.getenv("CHUNK_VERSION", "mineru_v3_650")
CHUNK_ENCODING = os.getenv("CHUNK_ENCODING", "cl100k_base")
CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", "30"))
CHUNK_MAX_TABLE_TOKENS = int(os.getenv("CHUNK_MAX_TABLE_TOKENS", "500"))
CHAT_RAG_TOP_K = int(os.getenv("CHAT_RAG_TOP_K", "5"))
CHAT_RAG_MAX_CONTEXT_CHARS = int(os.getenv("CHAT_RAG_MAX_CONTEXT_CHARS", "6000"))

MINERU_BACKEND = os.getenv("MINERU_BACKEND", "hybrid-auto-engine")
MINERU_METHOD = os.getenv("MINERU_METHOD", "auto")
MINERU_LANG = os.getenv("MINERU_LANG", "korean")
MINERU_FORMULA = os.getenv("MINERU_FORMULA", "true")
MINERU_TABLE = os.getenv("MINERU_TABLE", "true")
MINERU_TIMEOUT_SEC = int(os.getenv("MINERU_TIMEOUT_SEC", "1800"))
PIPELINE_TIMEOUT_SEC = int(os.getenv("PIPELINE_TIMEOUT_SEC", "2400"))

PARSER_SCRIPT_PATH = Path(os.getenv("PARSER_SCRIPT_PATH", str(APP_DIR / "pdf_parser.py")))
CHUNK_SCRIPT_PATH = Path(os.getenv("CHUNK_SCRIPT_PATH", str(APP_DIR / "chunker.py")))


def default_feature_chat_dir() -> Path:
    candidates = [
        APP_DIR / "feature_chat",
        ROOT_DIR / "final_1team-feature-chat",
        ROOT_DIR / "runpod" / "feature_chat",
        Path("/workspace/runpod/feature_chat"),
        Path("/workspace/final_1team/runpod/feature_chat"),
        Path("/workspace/final_1team-feature-chat"),
        Path("/workspace/feature_chat"),
    ]
    for path in candidates:
        if ((path / "rag_search.py").exists() or (path / "04_query_qdrant_openai.py").exists()) and (path / "rag").exists():
            return path
    return candidates[0]


FEATURE_CHAT_DIR = Path(os.getenv("FEATURE_CHAT_DIR", str(default_feature_chat_dir())))
