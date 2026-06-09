from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from config import (
    CHUNK_ENCODING,
    CHUNK_MAX_TABLE_TOKENS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP,
    CHUNK_SCRIPT_PATH,
    CHUNK_SIZE,
    CHUNK_VERSION,
    EMBEDDING_MODEL_ID,
    MINERU_BACKEND,
    MINERU_FORMULA,
    MINERU_LANG,
    MINERU_METHOD,
    MINERU_TABLE,
    MINERU_TIMEOUT_SEC,
    PARSER_SCRIPT_PATH,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from model_runtime import embed_texts, load_embedding_model


def load_script_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def clean_text(text: str) -> str:
    text = re.sub(r"\x00", "", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_parser(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    if not PARSER_SCRIPT_PATH.exists():
        raise RuntimeError(f"Parser script was not found: {PARSER_SCRIPT_PATH}")
    started = time.perf_counter()
    parser_module = load_script_module(PARSER_SCRIPT_PATH, "runpod_parse_pdfs_module")
    args = parser_module.argparse.Namespace(
        input_dir=input_dir,
        output_dir=output_dir,
        work_dir=None,
        backend=MINERU_BACKEND,
        method=MINERU_METHOD,
        lang=MINERU_LANG,
        recursive=True,
        start_index=0,
        limit=None,
        only=None,
        force=True,
        auto_fallback=True,
        keep_work=False,
        keep_raw=False,
        timeout=MINERU_TIMEOUT_SEC,
        cuda="0",
        require_cuda=True,
        formula=MINERU_FORMULA,
        table=MINERU_TABLE,
        api_url=None,
        disable_repeated_margin_clean=False,
        enable_ocr_corrections=True,
        secondary_text=True,
        quality_report=True,
    )

    try:
        parser_module.check_mineru_cli()
        parser_module.check_cuda_requirement(args)
        dirs = parser_module.ensure_output_dirs(output_dir.resolve())
        work_dir = parser_module.select_work_dir(args.work_dir)
        pdfs = parser_module.find_pdfs(
            input_dir.resolve(),
            recursive=args.recursive,
            only=args.only,
            start_index=args.start_index,
            limit=args.limit,
        )
        manifest_path = dirs["manifests"] / "parse_manifest.jsonl"
        registry_path = dirs["manifests"] / "document_registry.jsonl"
        records = []
        success = skipped = failed = 0
        for pdf_path in pdfs:
            record = parser_module.process_pdf(pdf_path, dirs, work_dir, args)
            parser_module.append_jsonl(manifest_path, record)
            registry_record = {
                "doc_id": record.get("doc_id"),
                "source_pdf": record.get("source_pdf"),
                "source_filename": record.get("source_filename"),
                "source_sha256": record.get("source_sha256"),
                "status": record.get("status"),
                "content_list_path": record.get("content_list_path"),
                "markdown_path": record.get("markdown_path"),
                "secondary_text_path": record.get("secondary_text_path"),
                "quality_report_path": record.get("quality_report_path"),
                "created_at": record.get("finished_at"),
            }
            parser_module.append_jsonl(registry_path, registry_record)
            records.append(record)
            status = record.get("status")
            if status == "success":
                success += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
    except Exception as exc:
        raise RuntimeError(f"MinerU parser failed: {exc}") from exc

    return {
        "mode": "direct_import",
        "script": str(PARSER_SCRIPT_PATH),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "pdf_count": len(pdfs),
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "records": records,
    }


def run_chunker(parsed_dir: Path, pdf_dir: Path, chunks_dir: Path) -> dict[str, Any]:
    input_path = parsed_dir / "json" / "all_blocks.jsonl"
    output_path = chunks_dir / "chunks_mineru_v1.jsonl"
    manifest_path = chunks_dir / "chunk_manifest.json"
    if not input_path.exists():
        raise RuntimeError(f"MinerU all_blocks.jsonl was not found: {input_path}")
    if not CHUNK_SCRIPT_PATH.exists():
        raise RuntimeError(f"Chunk script was not found: {CHUNK_SCRIPT_PATH}")
    started = time.perf_counter()
    chunk_module = load_script_module(CHUNK_SCRIPT_PATH, "runpod_make_chunks_module")
    args = chunk_module.argparse.Namespace(
        input=input_path,
        pdf_dir=pdf_dir,
        output=output_path,
        manifest=manifest_path,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        encoding=CHUNK_ENCODING,
        chunk_version=CHUNK_VERSION,
        min_chars=CHUNK_MIN_CHARS,
        max_table_tokens=CHUNK_MAX_TABLE_TOKENS,
        strict=False,
    )
    if args.overlap >= args.chunk_size:
        raise RuntimeError("Chunking failed: overlap must be smaller than chunk size")
    rows = chunk_module.read_jsonl(args.input, args.strict)
    filename_by_sha = chunk_module.pdf_hash_map(args.pdf_dir)
    tokenizer = chunk_module.Tokenizer(args.encoding)
    chunks = chunk_module.build_chunks(rows, tokenizer, args, filename_by_sha)
    chunk_module.write_jsonl(args.output, chunks)
    chunk_module.write_manifest(args.manifest, args, chunks, tokenizer)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return {
        "mode": "direct_import",
        "script": str(CHUNK_SCRIPT_PATH),
        "output": str(output_path),
        "manifest": manifest,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "blocks": len(rows),
        "chunks": len(chunks),
    }


def read_chunks_jsonl(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise RuntimeError(f"Invalid chunk JSON at line {line_no}")
            chunk_id = str(obj.get("chunk_id") or "")
            text = clean_text(str(obj.get("text") or obj.get("document") or ""))
            metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
            if chunk_id and text:
                chunks.append({"chunk_id": chunk_id, "document": text, "metadata": metadata})
    return chunks


def qdrant_point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"hpm-mineru:{chunk_id}"))


def upsert_chunks_to_qdrant(chunks: list[dict[str, Any]], metadata: dict[str, Any]) -> int:
    if not chunks:
        return 0
    from qdrant_client import QdrantClient, models

    bundle = load_embedding_model()
    vectors = embed_texts([chunk["document"] for chunk in chunks])
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        exists = bool(client.collection_exists(QDRANT_COLLECTION))
    except AttributeError:
        exists = QDRANT_COLLECTION in {item.name for item in client.get_collections().collections}
    if not exists:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=bundle.dimensions, distance=models.Distance.COSINE),
        )
    points = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        payload_metadata = {
            **metadata,
            **(chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}),
            "chunk_id": chunk["chunk_id"],
            "parser": "mineru",
        }
        points.append(
            models.PointStruct(
                id=qdrant_point_id(chunk["chunk_id"]),
                vector=vector,
                payload={"chunk_id": chunk["chunk_id"], "document": chunk["document"], "metadata": payload_metadata},
            )
        )
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)


def ingest_pdf_chunks(pdf_dir: Path, parsed_dir: Path, chunks_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    parser_info = run_parser(pdf_dir, parsed_dir)
    chunker_info = run_chunker(parsed_dir, pdf_dir, chunks_dir)
    chunks = read_chunks_jsonl(Path(str(chunker_info["output"])))
    upserted = upsert_chunks_to_qdrant(chunks, metadata)
    return {"parser": parser_info, "chunker": chunker_info, "chunks": chunks, "upserted": upserted}
