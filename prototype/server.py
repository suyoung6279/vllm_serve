from __future__ import annotations

import importlib.util
import json
import mimetypes
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT_DIR = Path(__file__).resolve().parent.parent
PROTOTYPE_DIR = Path(__file__).resolve().parent
RAG_DIR = ROOT_DIR / "final_1team-feature-chat"
PARSER_PATH = ROOT_DIR / "runpod" / "pdf_parser.py"
UPLOAD_DIR = PROTOTYPE_DIR / "uploads"
PARSED_DIR = PROTOTYPE_DIR / "parsed"

PORT = int(os.getenv("PROTOTYPE_PORT", "5177"))
RAG_TOP_K = int(os.getenv("PROTOTYPE_RAG_TOP_K", "5"))
PREPARATION_RAG_TOP_K = int(os.getenv("PROTOTYPE_PREPARATION_RAG_TOP_K", "4"))


def load_env() -> None:
    for path in [ROOT_DIR / ".env", ROOT_DIR / "backend" / ".env"]:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()


class RagRetriever:
    def __init__(self) -> None:
        self.loaded = False
        self.query_module = None
        self.embedder = None
        self.client = None
        self.collection = None
        self.records = []
        self.bm25 = None

    def load(self) -> None:
        if self.loaded:
            return
        if str(RAG_DIR) not in sys.path:
            sys.path.insert(0, str(RAG_DIR))

        from rank_bm25 import BM25Okapi
        from rag.embeddings import DEFAULT_HF_EMBEDDING_MODEL, make_embedder
        from rag.qdrant_store import config_from_env, make_client

        module_path = RAG_DIR / "04_query_qdrant_openai.py"
        spec = importlib.util.spec_from_file_location("prototype_query_qdrant", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        dimensions = os.getenv("RAG_EMBEDDING_DIMENSIONS")
        self.query_module = module
        self.embedder = make_embedder(
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
        self.client = make_client(config)
        self.collection = config.collection
        self.records = module.load_corpus(self.client, self.collection)
        self.bm25 = BM25Okapi([module.tokenize(module.retrieval_text(record)) for record in self.records])
        self.loaded = True

    def search_blocks(self, question: str, top_k: int) -> list[dict[str, str]]:
        self.load()
        if self.query_module is None or self.embedder is None or self.client is None or self.bm25 is None:
            raise RuntimeError("RAG retriever is not initialized.")

        args = self.query_module.argparse.Namespace(
            query=question,
            top_k=top_k,
            dense_k=max(20, top_k * 4),
            dense_fetch_k=max(50, top_k * 8),
            bm25_k=max(20, top_k * 4),
            rrf_k=60,
            rrf_top_k=max(20, top_k * 4),
            rerank_top_n=max(10, top_k * 2),
            where_doc_id=None,
            where_source_contains=os.getenv("RAG_INTERNAL_DOCUMENT_SOURCE_CONTAINS") or None,
            exclude_chunk_types=os.getenv("RAG_EXCLUDE_CHUNK_TYPES", "image_caption,chart_caption"),
            skip_rerank=True,
            cohere_model=os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"),
            max_tokens_per_doc=4096,
        )
        hits = self.query_module.retrieve(args, self.embedder, self.client, self.collection, self.records, self.bm25)
        return [self.hit_to_block(hit) for hit in hits[:top_k]]

    def hit_to_block(self, hit: object) -> dict[str, str]:
        if self.query_module is None:
            return {"source": "[회의 주제: 미정][일시: 미정][출처: 미정]", "text": ""}
        meta = getattr(hit, "metadata", {}) or {}
        text = self.query_module.normalize_context_text(getattr(hit, "document", "") or "")
        topic = meta.get("meeting_topic") or meta.get("meeting_title") or meta.get("title") or meta.get("source_title") or "미정"
        meeting_at = meta.get("meeting_at") or meta.get("meeting_datetime") or meta.get("date") or meta.get("created_at") or "미정"
        source = self.query_module.source_label(meta)
        return {
            "source": f"[회의 주제: {topic}][일시: {meeting_at}][출처: {source}]",
            "text": text[:1500],
            "category": "internal_document",
        }


rag_retriever = RagRetriever()


def runpod_base_url() -> str:
    base_url = os.getenv("RUNPOD_BASE_URL", "").strip().rstrip("/")
    minutes_url = os.getenv("RUNPOD_MINUTES_URL", "").strip()
    if base_url:
        return base_url
    if minutes_url:
        return minutes_url.rsplit("/", 1)[0]
    raise RuntimeError("Set RUNPOD_BASE_URL in .env first.")


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload: object) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self.proxy("GET", "/health", None)
            return
        if self.path == "/api/config":
            try:
                self.send_json(
                    200,
                    {
                        "runpod_base_url": runpod_base_url(),
                        "port": PORT,
                        "rag_collection": os.getenv("QDRANT_COLLECTION", ""),
                        "parser": str(PARSER_PATH),
                    },
                )
            except RuntimeError as exc:
                self.send_json(500, {"error": str(exc)})
            return
        if self.path in ["/", "/index.html"]:
            self.send_file(PROTOTYPE_DIR / "index.html")
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/api/documents/parse":
            self.parse_documents()
            return

        allowed = {
            "/api/generate-agendas": "/generate-agendas",
            "/api/generate-preparation": "/generate-preparation",
            "/api/generate-minutes": "/generate-minutes",
            "/api/chat": "/chat",
        }
        target = allowed.get(self.path)
        if target is None:
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        self.proxy("POST", target, body)

    def with_chat_rag(self, body: bytes) -> bytes:
        payload = self.read_json_body(body, {"question": "", "context": "", "history": []})
        question = str(payload.get("question") or "").strip()
        if not question or os.getenv("PROTOTYPE_ENABLE_RAG", "true").lower() != "true":
            return json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            blocks = rag_retriever.search_blocks(question, RAG_TOP_K)
            payload["context"] = "\n\n".join(f"{block['source']}\n{block['text']}" for block in blocks)
            payload["sources"] = [block["source"] for block in blocks]
            payload["rag_hit_count"] = len(blocks)
            if not blocks:
                raise RuntimeError("Qdrant search returned 0 chunks.")
        except Exception as exc:
            raise RuntimeError(f"Qdrant search failed: {exc}") from exc
            payload["context"] = f"[Qdrant 검색 오류]\n{exc}"
            payload["rag_error"] = str(exc)
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def with_preparation_rag(self, body: bytes) -> bytes:
        payload = self.read_json_body(body, {})
        title = str(payload.get("title") or "").strip()
        participants = payload.get("participants") or []
        participant_text = " ".join(
            f"{item.get('name', '')} {item.get('work', '')}" if isinstance(item, dict) else str(item)
            for item in participants
        )
        agendas = payload.get("agendas") or []
        query_base = " ".join([title, participant_text, " ".join(map(str, agendas))]).strip()
        payload.setdefault("previous_meetings", [])
        payload.setdefault("external_documents", [])
        if query_base and os.getenv("PROTOTYPE_ENABLE_RAG", "true").lower() == "true":
            try:
                payload["internal_documents"] = rag_retriever.search_blocks(
                    query_base + " 내부 문서 요구사항 정책 절차 회의 준비",
                    PREPARATION_RAG_TOP_K,
                )
            except Exception as exc:
                payload.setdefault("internal_documents", [])
                payload["rag_error"] = str(exc)
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def read_json_body(self, body: bytes, fallback: dict[str, object]) -> dict[str, object]:
        try:
            value = json.loads(body.decode("utf-8") or "{}")
            return value if isinstance(value, dict) else dict(fallback)
        except Exception:
            return dict(fallback)

    def parse_documents(self) -> None:
        try:
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self.send_json(400, {"error": "multipart/form-data로 PDF 파일을 업로드해야 합니다."})
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            request = Request(
                f"{runpod_base_url()}/documents/ingest",
                data=body,
                headers={
                    "Accept": "application/json",
                    "Content-Type": content_type,
                    "User-Agent": "Mozilla/5.0 Prototype",
                },
                method="POST",
            )
            with urlopen(request, timeout=2400) as response:
                data = response.read()
                status = response.status
                response_type = response.headers.get("Content-Type", "application/json")
            self.send_response(status)
            self.send_header("Content-Type", response_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except HTTPError as exc:
            data = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def multipart_boundary(self) -> str | None:
        content_type = self.headers.get("Content-Type", "")
        if "boundary=" not in content_type:
            return None
        return content_type.split("boundary=", 1)[1].split(";", 1)[0].strip().strip('"')

    def save_multipart_files(self, body: bytes, boundary: str) -> list[Path]:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        delimiter = b"--" + boundary.encode("utf-8")
        saved: list[Path] = []
        for part in body.split(delimiter):
            if not part or part in {b"--\r\n", b"--"} or b"\r\n\r\n" not in part:
                continue
            raw_headers, data = part.strip(b"\r\n").split(b"\r\n\r\n", 1)
            headers = raw_headers.decode("utf-8", errors="replace")
            if 'name="files"' not in headers or "filename=" not in headers:
                continue
            if data.endswith(b"\r\n"):
                data = data[:-2]
            safe_name = f"{int(time.time() * 1000)}_{len(saved) + 1}.pdf"
            dest = UPLOAD_DIR / safe_name
            dest.write_bytes(data)
            saved.append(dest)
        return saved

    def parsed_file_summary(self) -> dict[str, list[str]]:
        return {
            "markdown": sorted(path.name for path in (PARSED_DIR / "markdown").glob("*.md")) if (PARSED_DIR / "markdown").exists() else [],
            "json": sorted(path.name for path in (PARSED_DIR / "json").glob("*.json")) if (PARSED_DIR / "json").exists() else [],
            "reports": sorted(path.name for path in (PARSED_DIR / "reports").glob("*.json")) if (PARSED_DIR / "reports").exists() else [],
        }

    def proxy(self, method: str, target_path: str, body: bytes | None) -> None:
        try:
            request = Request(
                f"{runpod_base_url()}{target_path}",
                data=body,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 Prototype",
                },
                method=method,
            )
            with urlopen(request, timeout=1800) as response:
                data = response.read()
                status = response.status
                content_type = response.headers.get("Content-Type", "application/json")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except HTTPError as exc:
            data = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "text/plain"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (RuntimeError, URLError, TimeoutError) as exc:
            self.send_json(502, {"error": str(exc)})


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Prototype running at http://127.0.0.1:{PORT}")
    print(f"RunPod base URL: {os.getenv('RUNPOD_BASE_URL') or '(not set)'}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
