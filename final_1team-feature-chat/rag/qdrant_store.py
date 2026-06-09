from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client import models


DEFAULT_QDRANT_COLLECTION = "mineru_pdf_chunks_ko_sroberta"


@dataclass
class QdrantConfig:
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection: str = DEFAULT_QDRANT_COLLECTION
    prefer_grpc: bool = False


@dataclass
class PointRecord:
    chunk_id: str
    document: str
    metadata: dict[str, Any]


def qdrant_point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"scenario-pased-v1:{chunk_id}"))


def config_from_env(
    url: str | None = None,
    api_key: str | None = None,
    collection: str | None = None,
    prefer_grpc: bool = False,
) -> QdrantConfig:
    return QdrantConfig(
        url=url or os.getenv("QDRANT_URL") or os.getenv("QDRANT_HOST") or "http://localhost:6333",
        api_key=api_key if api_key is not None else os.getenv("QDRANT_API_KEY"),
        collection=collection or os.getenv("QDRANT_COLLECTION") or DEFAULT_QDRANT_COLLECTION,
        prefer_grpc=prefer_grpc,
    )


def make_client(config: QdrantConfig) -> QdrantClient:
    return QdrantClient(url=config.url, api_key=config.api_key, prefer_grpc=config.prefer_grpc)


def collection_exists(client: QdrantClient, collection: str) -> bool:
    try:
        return bool(client.collection_exists(collection))
    except AttributeError:
        names = {item.name for item in client.get_collections().collections}
        return collection in names


def ensure_collection(client: QdrantClient, collection: str, vector_size: int, reset: bool = False) -> None:
    if reset and collection_exists(client, collection):
        client.delete_collection(collection_name=collection)
        print(f"[INFO] Deleted existing Qdrant collection: {collection}")

    if collection_exists(client, collection):
        return

    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    print(f"[INFO] Created Qdrant collection: {collection} (size={vector_size}, distance=cosine)")


def payload_to_record(payload: dict[str, Any] | None) -> PointRecord:
    payload = payload or {}
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    chunk_id = str(payload.get("chunk_id") or metadata.get("chunk_id") or "")
    document = str(payload.get("document") or "")
    clean_metadata = dict(metadata)
    clean_metadata.setdefault("chunk_id", chunk_id)
    return PointRecord(chunk_id=chunk_id, document=document, metadata=clean_metadata)


def scroll_records(client: QdrantClient, collection: str, batch_size: int = 1000) -> list[PointRecord]:
    records: list[PointRecord] = []
    offset: Any = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            records.append(payload_to_record(getattr(point, "payload", None)))
        if offset is None:
            break
    if not records:
        raise RuntimeError("No documents found in Qdrant collection. Run 03_0_index_qdrant_openai.py first.")
    return records


def count_points(client: QdrantClient, collection: str) -> int:
    result = client.count(collection_name=collection, exact=True)
    return int(result.count)


def existing_chunk_ids(client: QdrantClient, collection: str, chunk_ids: list[str]) -> set[str]:
    if not chunk_ids:
        return set()
    point_ids = [qdrant_point_id(chunk_id) for chunk_id in chunk_ids]
    try:
        points = client.retrieve(collection_name=collection, ids=point_ids, with_payload=True, with_vectors=False)
    except Exception:
        return set()
    existing: set[str] = set()
    for point in points:
        payload = getattr(point, "payload", None) or {}
        chunk_id = payload.get("chunk_id")
        if chunk_id:
            existing.add(str(chunk_id))
    return existing


def make_filter(doc_id: str | None = None) -> models.Filter | None:
    if not doc_id:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.doc_id",
                match=models.MatchValue(value=doc_id),
            )
        ]
    )


def search_points(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    limit: int,
    doc_id: str | None = None,
) -> list[Any]:
    query_filter = make_filter(doc_id)
    if hasattr(client, "query_points"):
        result = client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return list(getattr(result, "points", result))

    return list(
        client.search(
            collection_name=collection,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    )
