from __future__ import annotations

import re
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from config import QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL
from model_runtime import embed_texts, load_embedding_model


def clean_text(text: str) -> str:
    text = re.sub(r"\x00", "", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, *, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [text]:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(start + max_chars, len(paragraph))
                chunks.append(paragraph[start:end].strip())
                if end == len(paragraph):
                    break
                start = max(0, end - overlap)
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)
    return chunks


def point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"hpm-meeting-minutes:{chunk_id}"))


def ensure_payload_indexes(client: Any, collection_name: str) -> None:
    from qdrant_client import models

    for field_name in ["metadata.source_type", "metadata.meeting_id"]:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "already exists" not in message and "already has" not in message:
                raise


def base_metadata(req: Any, result: dict[str, Any]) -> dict[str, Any]:
    minutes = result.get("minutes") if isinstance(result.get("minutes"), dict) else {}
    title = getattr(req, "title", "") or minutes.get("meeting_topic") or "미정"
    meeting_datetime = getattr(req, "meeting_datetime", "") or minutes.get("meeting_datetime") or "미정"
    source = f"이전회의록: {title}"
    if meeting_datetime and meeting_datetime != "미정":
        source = f"{source} / {meeting_datetime}"
    return {
        "source_type": "meeting_minutes",
        "doc_type": "meeting_minutes",
        "meeting_id": str(getattr(req, "meeting_id", "") or ""),
        "project_id": str(getattr(req, "project_id", "") or ""),
        "viewer_type": "meeting_minutes",
        "viewer_id": str(getattr(req, "meeting_id", "") or ""),
        "title": title,
        "meeting_topic": title,
        "meeting_datetime": meeting_datetime,
        "location": getattr(req, "location", "") or minutes.get("location") or "미정",
        "source": source,
        "source_filename": source,
        "parser": "meeting_text",
    }


def make_chunks(req: Any, result: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = base_metadata(req, result)
    meeting_id = metadata["meeting_id"] or "unknown"
    chunks: list[dict[str, Any]] = []

    document = result.get("cotent") or result.get("content") or ""
    for index, chunk_text in enumerate(split_text(str(document)), 1):
        chunk_id = f"meeting::{meeting_id}::meeting_document::{index:04d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "document": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_id": chunk_id,
                    "chunk_type": "meeting_document",
                    "chunk_index_in_group": index,
                },
            }
        )
    return chunks


def upsert_meeting_chunks(chunks: list[dict[str, Any]]) -> int:
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

    ensure_payload_indexes(client, QDRANT_COLLECTION)
    meeting_id = str(chunks[0].get("metadata", {}).get("meeting_id") or "")
    if meeting_id:
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source_type",
                            match=models.MatchValue(value="meeting_minutes"),
                        ),
                        models.FieldCondition(
                            key="metadata.meeting_id",
                            match=models.MatchValue(value=meeting_id),
                        ),
                    ]
                )
            ),
        )

    points = [
        models.PointStruct(
            id=point_id(chunk["chunk_id"]),
            vector=vector,
            payload={
                "chunk_id": chunk["chunk_id"],
                "document": chunk["document"],
                "metadata": chunk["metadata"],
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)


def ingest_meeting_minutes(req: Any, result: dict[str, Any]) -> dict[str, Any]:
    chunks = make_chunks(req, result)
    upserted = upsert_meeting_chunks(chunks)
    return {
        "chunk_count": len(chunks),
        "upserted_points": upserted,
        "qdrant_collection": QDRANT_COLLECTION,
        "source_type": "meeting_minutes",
    }
