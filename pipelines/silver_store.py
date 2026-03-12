import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import Any, Iterable

import boto3
import polars as pl


S3_BUCKET = os.getenv("S3_BUCKET", "projetbigdata0")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")

SILVER_DOCUMENTS = f"s3://{S3_BUCKET}/silver/documents"
SILVER_CHUNKS = f"s3://{S3_BUCKET}/silver/chunks"
SILVER_EMBEDDINGS = f"s3://{S3_BUCKET}/silver/embeddings"


def storage_options() -> dict:
    key = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    token = os.getenv("AWS_SESSION_TOKEN")

    opts = {
        "client_kwargs": f'{{"region_name": "{AWS_DEFAULT_REGION}"}}'
    }

    if key and secret:
        opts["key"] = key
        opts["secret"] = secret
    if token:
        opts["token"] = token

    return opts


def s3_client():
    return boto3.client("s3", region_name=AWS_DEFAULT_REGION)


def today_partition() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def json_dumps_safe(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def upload_raw_to_s3(local_path: str, kind: str) -> str:
    filename = os.path.basename(local_path)
    key_name = f"{uuid.uuid4().hex}_{filename}"

    if kind == "audio":
        prefix = "raw/audio/"
    elif kind == "pdf":
        prefix = "raw/pdf/"
    elif kind == "photo":
        prefix = "raw/photo/"
    else:
        prefix = "raw/other/"

    key = prefix + key_name
    s3_client().upload_file(local_path, S3_BUCKET, key)
    return f"s3://{S3_BUCKET}/{key}"


def write_silver_documents_row(
    *,
    doc_id: str,
    source_type: str,
    raw_path: str,
    text: str,
    title: str,
    keywords: str,
    summary: str,
    content_hash: str,
) -> str:
    date = today_partition()

    df = pl.DataFrame([{
        "doc_id": doc_id,
        "source_type": source_type,
        "raw_path": raw_path,
        "text": text,
        "title": title,
        "keywords": keywords,
        "summary": summary,
        "created_at": datetime.utcnow().isoformat(),
        "content_hash": content_hash,
    }])

    out = f"{SILVER_DOCUMENTS}/date={date}/doc_id={doc_id}/part-{uuid.uuid4().hex}.parquet"
    df.write_parquet(out, storage_options=storage_options())
    return out


def write_silver_chunks_and_embeddings(
    *,
    doc_id: str,
    nodes: Iterable,
    embed_model,
    embed_model_name: str,
) -> tuple[str, str]:
    date = today_partition()

    chunk_rows = []
    emb_rows = []

    for idx, node in enumerate(nodes):
        chunk_id = uuid.uuid4().hex

        md = getattr(node, "metadata", None) or {}
        md = dict(md)
        md["doc_id"] = doc_id
        md["chunk_id"] = chunk_id
        node.metadata = md

        text = getattr(node, "text", "") or ""
        vector = embed_model.get_text_embedding(text)

        chunk_rows.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "chunk_text": text,
            "chunk_index": idx,
            "metadata_json": json_dumps_safe(md),
            "created_at": datetime.utcnow().isoformat(),
        })

        emb_rows.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "vector": vector,
            "model": embed_model_name,
            "created_at": datetime.utcnow().isoformat(),
        })

    df_chunks = pl.DataFrame(chunk_rows)
    df_emb = pl.DataFrame(emb_rows)

    out_chunks = f"{SILVER_CHUNKS}/date={date}/doc_id={doc_id}/part-{uuid.uuid4().hex}.parquet"
    out_emb = f"{SILVER_EMBEDDINGS}/date={date}/doc_id={doc_id}/part-{uuid.uuid4().hex}.parquet"

    df_chunks.write_parquet(out_chunks, storage_options=storage_options())
    df_emb.write_parquet(out_emb, storage_options=storage_options())

    return out_chunks, out_emb
