import os
import json
from typing import Optional

import polars as pl
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode

from pipelines.silver_store import S3_BUCKET, storage_options


SILVER_CHUNKS = f"s3://{S3_BUCKET}/silver/chunks"


def load_nodes_from_silver(
    *,
    doc_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[TextNode]:
    if doc_id:
        parquet_glob = f"{SILVER_CHUNKS}/**/doc_id={doc_id}/*.parquet"
    else:
        parquet_glob = f"{SILVER_CHUNKS}/**/*.parquet"

    df = pl.read_parquet(parquet_glob, storage_options=storage_options())

    if limit is not None:
        df = df.head(limit)

    if "chunk_index" in df.columns:
        df = df.sort("chunk_index")

    nodes: list[TextNode] = []

    for row in df.iter_rows(named=True):
        md = {}
        try:
            md = json.loads(row.get("metadata_json") or "{}")
        except Exception:
            md = {}

        nodes.append(
            TextNode(
                text=row.get("chunk_text") or "",
                metadata=md,
            )
        )

    return nodes


def rebuild_index(
    *,
    nodes: list[TextNode],
    llm=None,
    storage_context=None,
):
    return VectorStoreIndex(
        nodes=nodes,
        llm=llm,
        storage_context=storage_context,
    )


def rebuild_index_from_silver(
    *,
    llm=None,
    storage_context=None,
    doc_id: Optional[str] = None,
    limit: Optional[int] = None,
):
    nodes = load_nodes_from_silver(doc_id=doc_id, limit=limit)
    return rebuild_index(nodes=nodes, llm=llm, storage_context=storage_context)
