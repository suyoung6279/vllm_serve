#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha1, sha256
from pathlib import Path
from typing import Any, Iterable

try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
ARTICLE_RE = re.compile(r"^\s*제\s*(\d{1,4})\s*조(?:의\s*(\d{1,3}))?\s*(?:\(([^)]{1,120})\))?")
HEADING_RE = re.compile(r"^\s*(제\s*\d+\s*(?:장|절)|[0-9]+[.)]\s+.{1,120})")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


class Tokenizer:
    def __init__(self, encoding_name: str) -> None:
        self.encoding_name = encoding_name
        self._encoding = None
        if tiktoken is not None:
            try:
                self._encoding = tiktoken.get_encoding(encoding_name)
            except Exception:
                self._encoding = tiktoken.get_encoding("cl100k_base")
                self.encoding_name = "cl100k_base"

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return max(1, len(text) // 2)

    def split(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        text = normalize_text(text)
        if not text:
            return []
        if self.count(text) <= max_tokens:
            return [text]

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) > 1:
            out: list[str] = []
            current = ""
            for paragraph in paragraphs:
                candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
                if self.count(candidate) <= max_tokens:
                    current = candidate
                    continue
                if current:
                    out.append(current)
                current = paragraph
            if current:
                out.append(current)
            return [piece for chunk in out for piece in self.split(chunk, max_tokens, overlap)]

        if self._encoding is not None:
            tokens = self._encoding.encode(text)
            out = []
            start = 0
            while start < len(tokens):
                end = min(start + max_tokens, len(tokens))
                out.append(self._encoding.decode(tokens[start:end]).strip())
                if end >= len(tokens):
                    break
                start = max(0, end - overlap)
            return [x for x in out if x]

        max_chars = max_tokens * 2
        overlap_chars = overlap * 2
        out = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            out.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(0, end - overlap_chars)
        return [x for x in out if x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create RAG chunks from parsed/json/all_blocks.jsonl.")
    parser.add_argument("--input", type=Path, default=Path("parsed/json/all_blocks.jsonl"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("pdf_data"))
    parser.add_argument("--output", type=Path, default=Path("chunks/chunks_mineru_v1.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("chunks/chunk_manifest.json"))
    parser.add_argument("--chunk-size", type=int, default=650)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--encoding", type=str, default="cl100k_base")
    parser.add_argument("--chunk-version", type=str, default="mineru_v3_650")
    parser.add_argument("--min-chars", type=int, default=30)
    parser.add_argument("--max-table-tokens", type=int, default=500)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = html.unescape(value)
    value = HTML_TAG_RE.sub(" ", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = MULTI_SPACE_RE.sub(" ", value)
    value = MULTI_BLANK_RE.sub("\n\n", value)
    return value.strip()


def read_jsonl(path: Path, strict: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if strict:
                    raise
                print(f"[WARN] malformed JSON skipped at line {line_no}")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            elif strict:
                raise ValueError(f"Line {line_no} is not a JSON object")
    return rows


def pdf_hash_map(pdf_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not pdf_dir.exists():
        return mapping
    for path in sorted(pdf_dir.rglob("*.pdf")):
        h = sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        digest = h.hexdigest()
        mapping[digest] = path.name
        mapping[digest[:16]] = path.name
    return mapping


def page_no(block: dict[str, Any]) -> int | None:
    value = block.get("page_no")
    if value is None and block.get("page_idx") is not None:
        value = int(block["page_idx"]) + 1
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def block_index(block: dict[str, Any]) -> int:
    raw = block.get("raw") if isinstance(block.get("raw"), dict) else {}
    value = raw.get("original_block_index", block.get("original_block_index"))
    try:
        return int(value)
    except Exception:
        return 10**9


def source_name(block: dict[str, Any], filename_by_sha: dict[str, str]) -> str:
    digest = str(block.get("source_sha256") or "")
    if digest and digest in filename_by_sha:
        return filename_by_sha[digest]
    doc_id = str(block.get("doc_id") or "")
    if doc_id.startswith("doc_"):
        short_digest = doc_id.removeprefix("doc_")
        if short_digest in filename_by_sha:
            return filename_by_sha[short_digest]
    return Path(str(block.get("source_filename") or block.get("source_pdf") or "unknown.pdf")).name


def block_text(block: dict[str, Any]) -> str:
    text = normalize_text(block.get("text"))
    if text:
        return text
    raw = block.get("raw") if isinstance(block.get("raw"), dict) else {}
    for key in ("text", "content", "table_body", "table_caption", "image_caption"):
        text = normalize_text(raw.get(key))
        if text:
            return text
    return ""


def article_info(text: str) -> tuple[int | None, int | None, str | None]:
    first = text.splitlines()[0] if text else ""
    match = ARTICLE_RE.match(first)
    if not match:
        return None, None, None
    return int(match.group(1)), int(match.group(2)) if match.group(2) else None, match.group(3)


def heading_from_text(text: str) -> str | None:
    first = text.splitlines()[0] if text else ""
    match = HEADING_RE.match(first)
    return match.group(1).strip() if match else None


def make_prefix(meta: dict[str, Any]) -> str:
    parts = []
    if meta.get("source_filename"):
        parts.append(f"[문서: {meta['source_filename']}]")
    if meta.get("page_start"):
        page = meta["page_start"] if meta.get("page_start") == meta.get("page_end") else f"{meta['page_start']}-{meta['page_end']}"
        parts.append(f"[페이지: {page}]")
    if meta.get("heading"):
        parts.append(f"[제목: {meta['heading']}]")
    if meta.get("article_no"):
        sub = f"의{meta['article_sub_no']}" if meta.get("article_sub_no") else ""
        title = f" {meta['article_title']}" if meta.get("article_title") else ""
        parts.append(f"[조문: 제{meta['article_no']}조{sub}{title}]")
    return "\n".join(parts)


def base_metadata(blocks: list[dict[str, Any]], chunk_type: str, chunk_version: str, filename_by_sha: dict[str, str]) -> dict[str, Any]:
    first = blocks[0]
    pages = [p for p in (page_no(b) for b in blocks) if p is not None]
    articles = [article_info(block_text(b)) for b in blocks]
    article = next((a for a in articles if a[0] is not None), (None, None, None))
    headings = [heading_from_text(block_text(b)) for b in blocks]
    return {
        "doc_id": first.get("doc_id"),
        "source_pdf": first.get("source_pdf"),
        "source_filename": source_name(first, filename_by_sha),
        "source_sha256": first.get("source_sha256"),
        "parser": first.get("parser"),
        "chunk_version": chunk_version,
        "chunk_type": chunk_type,
        "page_start": min(pages) if pages else None,
        "page_end": max(pages) if pages else None,
        "block_types": sorted({str(b.get("block_type") or b.get("type") or "unknown") for b in blocks}),
        "content_categories": sorted({str(b.get("content_category") or "unknown") for b in blocks}),
        "source_block_ids": [str(b.get("block_id") or "") for b in blocks if b.get("block_id")],
        "quality_flags": sorted({str(flag) for b in blocks for flag in (b.get("quality_flags") or [])}),
        "ocr_suspicious": any(bool(b.get("ocr_suspicious")) for b in blocks),
        "article_no": article[0],
        "article_sub_no": article[1],
        "article_title": article[2],
        "heading": next((h for h in headings if h), None),
    }


def emit_group(
    blocks: list[dict[str, Any]],
    chunk_type: str,
    tokenizer: Tokenizer,
    args: argparse.Namespace,
    filename_by_sha: dict[str, str],
) -> list[Chunk]:
    text = "\n\n".join(block_text(b) for b in blocks if block_text(b))
    if len(text) < args.min_chars:
        return []
    meta = base_metadata(blocks, chunk_type, args.chunk_version, filename_by_sha)
    prefix = make_prefix(meta)
    available = max(150, args.chunk_size - tokenizer.count(prefix) - 8)
    pieces = tokenizer.split(text, available, args.overlap)
    chunks: list[Chunk] = []
    for idx, piece in enumerate(pieces, 1):
        final = f"{prefix}\n\n{piece}".strip() if prefix else piece
        meta_i = dict(meta)
        meta_i["chunk_index_in_group"] = idx
        meta_i["chunk_count_in_group"] = len(pieces)
        meta_i["token_count"] = tokenizer.count(final)
        meta_i["char_count"] = len(final)
        chunks.append(Chunk("", final, meta_i))
    return chunks


def group_blocks(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("doc_id") or "unknown_doc")].append(row)
    for blocks in grouped.values():
        blocks.sort(key=lambda b: (page_no(b) or 10**9, block_index(b), str(b.get("block_id") or "")))
    return grouped


def build_chunks(rows: list[dict[str, Any]], tokenizer: Tokenizer, args: argparse.Namespace, filename_by_sha: dict[str, str]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    grouped = group_blocks(rows)
    iterator: Iterable[tuple[str, list[dict[str, Any]]]] = grouped.items()
    if tqdm is not None:
        iterator = tqdm(list(iterator), desc="Chunking documents")

    for _doc_id, blocks in iterator:
        current: list[dict[str, Any]] = []
        current_key: tuple[Any, ...] | None = None

        def flush() -> None:
            nonlocal current, current_key
            if current:
                all_chunks.extend(emit_group(current, str(current_key[0]), tokenizer, args, filename_by_sha))
            current = []
            current_key = None

        for block in blocks:
            text = block_text(block)
            if not text:
                continue
            block_type = str(block.get("block_type") or block.get("type") or "text").lower()
            category = str(block.get("content_category") or "").lower()
            art_no, art_sub, _ = article_info(text)

            if block_type == "table" or category == "table":
                flush()
                all_chunks.extend(emit_group([block], "table", tokenizer, args, filename_by_sha))
                continue
            if block_type in {"image", "chart"} or category == "visual":
                flush()
                continue

            if art_no is not None:
                key = ("article", art_no, art_sub)
            else:
                key = ("section_text", heading_from_text(text), page_no(block))

            if current and current_key != key:
                flush()
            current_key = key
            current.append(block)
            rolling_text = "\n\n".join(block_text(b) for b in current)
            if tokenizer.count(rolling_text) >= args.chunk_size * 1.7:
                flush()
        flush()

    for i, chunk in enumerate(all_chunks, 1):
        doc_id = str(chunk.metadata.get("doc_id") or "unknown_doc")
        chunk.chunk_id = f"{doc_id}::chunk_{i:07d}_{sha1(chunk.text.encode('utf-8')).hexdigest()[:10]}"
        chunk.metadata["chunk_seq_global"] = i
    return all_chunks


def write_jsonl(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps({"chunk_id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata}, ensure_ascii=False) + "\n")


def write_manifest(path: Path, args: argparse.Namespace, chunks: list[Chunk], tokenizer: Tokenizer) -> None:
    by_type = Counter(str(c.metadata.get("chunk_type")) for c in chunks)
    by_doc = Counter(str(c.metadata.get("doc_id")) for c in chunks)
    token_counts = [int(c.metadata.get("token_count") or tokenizer.count(c.text)) for c in chunks]
    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "chunk_version": args.chunk_version,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "encoding": tokenizer.encoding_name,
        "chunk_count": len(chunks),
        "doc_count": len(by_doc),
        "chunk_type_counts": dict(by_type),
        "token_count_min": min(token_counts) if token_counts else 0,
        "token_count_avg": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0,
        "token_count_max": max(token_counts) if token_counts else 0,
        "chunks_per_doc_min": min(by_doc.values()) if by_doc else 0,
        "chunks_per_doc_max": max(by_doc.values()) if by_doc else 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.overlap >= args.chunk_size:
        raise ValueError("--overlap must be smaller than --chunk-size")
    rows = read_jsonl(args.input, args.strict)
    filename_by_sha = pdf_hash_map(args.pdf_dir)
    tokenizer = Tokenizer(args.encoding)
    chunks = build_chunks(rows, tokenizer, args, filename_by_sha)
    write_jsonl(args.output, chunks)
    write_manifest(args.manifest, args, chunks, tokenizer)

    print("[DONE]")
    print(f"  input     : {args.input}")
    print(f"  output    : {args.output}")
    print(f"  manifest  : {args.manifest}")
    print(f"  blocks    : {len(rows)}")
    print(f"  chunks    : {len(chunks)}")
    if chunks:
        counts = [int(c.metadata.get("token_count") or tokenizer.count(c.text)) for c in chunks]
        print(f"  token avg : {round(sum(counts) / len(counts), 2)}")
        print(f"  token max : {max(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
