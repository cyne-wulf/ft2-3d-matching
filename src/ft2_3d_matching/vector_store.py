from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from .models import Candidate, EnrichedRecord
from .settings import COLLECTION_NAME, VECTOR_STORE

_CLIENTS: dict[str, QdrantClient] = {}


def close_local_clients() -> None:
    for client in list(_CLIENTS.values()):
        client.close()
    _CLIENTS.clear()


atexit.register(close_local_clients)


def local_client(path: Path | None = None) -> QdrantClient:
    store_path = path or Path(os.getenv("FT2_VECTOR_STORE_PATH", str(VECTOR_STORE)))
    store_path.mkdir(parents=True, exist_ok=True)
    key = str(store_path.resolve())
    if key not in _CLIENTS:
        _CLIENTS[key] = QdrantClient(path=key)
    return _CLIENTS[key]


def create_collection(client: QdrantClient, dimension: int, collection: str = COLLECTION_NAME) -> None:
    existing = [item.name for item in client.get_collections().collections]
    if collection in existing:
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )


def index_records(
    records: list[EnrichedRecord],
    vectors: list[list[float]],
    client: QdrantClient | None = None,
    collection: str = COLLECTION_NAME,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records are required before indexing")
    qdrant = client or local_client()
    create_collection(qdrant, len(vectors[0]), collection)
    points = []
    for index, (record, vector) in enumerate(zip(records, vectors, strict=True)):
        payload = {
            "company_id": record.company.company_id,
            "company_name": record.company.name,
            "category_code": record.company.category_code,
            "status": record.company.status,
            "country_code": record.company.country_code,
            "founded_year": record.company.founded_year,
            "funding_total_usd": record.company.funding_total_usd,
            "funding_rounds": record.company.funding_rounds,
            "company_text": record.company_text,
            "founder_text": record.founder_text,
        }
        points.append(models.PointStruct(id=index, vector=vector, payload=payload))
    qdrant.upsert(collection_name=collection, points=points)
    return {"collection": collection, "indexed_count": len(points), "dimension": len(vectors[0])}


def query_records(
    query_vector: list[float],
    limit: int = 5,
    client: QdrantClient | None = None,
    collection: str = COLLECTION_NAME,
) -> list[Candidate]:
    qdrant = client or local_client()
    if hasattr(qdrant, "query_points"):
        result = qdrant.query_points(collection_name=collection, query=query_vector, limit=limit)
        points = result.points
    else:
        points = qdrant.search(collection_name=collection, query_vector=query_vector, limit=limit)

    candidates: list[Candidate] = []
    for point in points:
        payload = point.payload or {}
        candidates.append(
            Candidate(
                company_id=str(payload.get("company_id", "")),
                company_name=str(payload.get("company_name", "")),
                category_code=str(payload.get("category_code", "")),
                status=str(payload.get("status", "")),
                country_code=str(payload.get("country_code", "")),
                founded_year=payload.get("founded_year"),
                funding_total_usd=float(payload.get("funding_total_usd") or 0.0),
                funding_rounds=int(payload.get("funding_rounds") or 0),
                company_text=str(payload.get("company_text", "")),
                founder_text=str(payload.get("founder_text", "")),
                company_thesis_similarity=float(point.score or 0.0),
            )
        )
    return candidates
