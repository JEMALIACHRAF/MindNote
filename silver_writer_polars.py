import os
import json
import uuid
from datetime import datetime

import polars as pl

# ---- S3 paths ----
S3_BUCKET = "projetbigdata0"
SILVER_DOCUMENTS = f"s3://{S3_BUCKET}/silver/documents"
SILVER_CHUNKS = f"s3://{S3_BUCKET}/silver/chunks"
SILVER_EMBEDDINGS = f"s3://{S3_BUCKET}/silver/embeddings"

def _storage_options():
    # s3fs/fsspec utilisent ces variables env automatiquement,
    # mais storage_options explicite marche bien aussi.
    # Si tu préfères, tu peux retourner {} et laisser env gérer.
    return {
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "client_kwargs": {"region_name": os.getenv("AWS_DEFAULT_REGION", "eu-west-1")},
    }

def _today_partition():
    return datetime.utcnow().strftime("%Y-%m-%d")

def _json_dumps_safe(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"

def write_documents_table(
    doc_id: str,
    source_type: str,
    raw_path: str,
    text: str,
    title: str,
    keywords: str,
    summary: str,
    content_hash: str | None = None,
):
    date = _today_partition()
    row = {
        "doc_id": doc_id,
        "source_type": source_type,
        "raw_path": raw_path,
        "text": text,
        "title": title,
        "keywords": keywords,
        "summary": summary,
        "created_at": datetime.utcnow().isoformat(),
        "content_hash": content_hash or "",
    }
    df = pl.DataFrame([row])

    out_path = f"{SILVER_DOCUMENTS}/date={date}/doc_id={doc_id}/part-{uuid.uuid4().hex}.parquet"
    df.write_parquet(out_path, storage_options=_storage_options())
    return out_path

def write_chunks_table(doc_id: str, nodes):
    """
    nodes: liste de TextNode / Node LlamaIndex
          attendu: node.text, node.metadata
    """
    date = _today_partition()

    rows = []
    for idx, node in enumerate(nodes):
        rows.append({
            "chunk_id": uuid.uuid4().hex,
            "doc_id": doc_id,
            "chunk_text": getattr(node, "text", "") or "",
            "chunk_index": idx,
            # metadata -> string JSON (simple, portable parquet)
            "metadata_json": _json_dumps_safe(getattr(node, "metadata", {}) or {}),
            "created_at": datetime.utcnow().isoformat(),
        })

    df = pl.DataFrame(rows)

    out_path = f"{SILVER_CHUNKS}/date={date}/doc_id={doc_id}/part-{uuid.uuid4().hex}.parquet"
    df.write_parquet(out_path, storage_options=_storage_options())
    return out_path

def write_embeddings_table(doc_id: str, chunk_ids: list[str], vectors: list[list[float]], model: str):
    """
    vectors: list[list[float]] (embedding par chunk)
    chunk_ids: mêmes longueur que vectors
    """
    date = _today_partition()
    rows = []
    for chunk_id, vec in zip(chunk_ids, vectors):
        rows.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "vector": vec,  # Parquet supporte list<float>
            "model": model,
            "created_at": datetime.utcnow().isoformat(),
        })

    df = pl.DataFrame(rows)

    out_path = f"{SILVER_EMBEDDINGS}/date={date}/doc_id={doc_id}/part-{uuid.uuid4().hex}.parquet"
    df.write_parquet(out_path, storage_options=_storage_options())
    return out_path

