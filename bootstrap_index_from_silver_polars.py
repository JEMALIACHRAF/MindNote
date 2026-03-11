import os
import json
import polars as pl

from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex

S3_BUCKET = "projetbigdata0"
SILVER_CHUNKS = f"s3://{S3_BUCKET}/silver/chunks"

def _storage_options():
    return {
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "client_kwargs": {"region_name": os.getenv("AWS_DEFAULT_REGION", "eu-west-1")},
    }

def load_nodes_from_silver(limit: int | None = None):
    # lit tous les fichiers parquet sous silver/chunks
    df = pl.read_parquet(f"{SILVER_CHUNKS}/**/*.parquet", storage_options=_storage_options())

    if limit:
        df = df.head(limit)

    nodes = []
    for row in df.iter_rows(named=True):
        md = {}
        try:
            md = json.loads(row.get("metadata_json") or "{}")
        except Exception:
            md = {}

        nodes.append(TextNode(
            text=row.get("chunk_text") or "",
            metadata=md
        ))

    return nodes

def rebuild_index(nodes, llm=None, storage_context=None):
    # Rebuild un index en mémoire (ou avec storage_context si tu veux)
    return VectorStoreIndex(nodes=nodes, llm=llm, storage_context=storage_context)

