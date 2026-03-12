from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable

from llama_index.core import Settings, StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

from pipelines.indexing import load_nodes_from_silver, rebuild_index


default_args = {
    "owner": "ashra",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

DAG_ID = "rebuild_second_brain_index"


def build_runtime():
    os.environ.setdefault("AWS_DEFAULT_REGION", "eu-north-1")

    openai_key = os.getenv("openai_key") or Variable.get("openai_key")
    chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    embedding_model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    llm = LlamaOpenAI(model="gpt-3.5-turbo", api_key=openai_key)
    Settings.llm = llm

    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
    Settings.embed_model = embed_model

    graph_store = SimpleGraphStore()

    chroma_client = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = chroma_client.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        graph_store=graph_store,
    )

    return {
        "llm": llm,
        "storage_context": storage_context,
    }


with DAG(
    dag_id=DAG_ID,
    description="Recharge les chunks depuis Silver et reconstruit l'index",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",  # tous les jours à 02:00
    catchup=False,
    tags=["second-brain", "index", "rebuild"],
) as dag:

    @task
    def read_conf(conf: dict | None = None) -> dict:
        conf = conf or {}
        return {
            "doc_id": conf.get("doc_id"),
            "limit": conf.get("limit"),
        }

    @task
    def load_nodes(payload: dict) -> dict:
        doc_id = payload.get("doc_id")
        limit = payload.get("limit")

        nodes = load_nodes_from_silver(doc_id=doc_id, limit=limit)

        print(f"{len(nodes)} nodes loaded from silver")
        return {
            "doc_id": doc_id,
            "limit": limit,
            "nodes_count": len(nodes),
        }

    @task
    def rebuild(payload: dict) -> dict:
        runtime = build_runtime()

        doc_id = payload.get("doc_id")
        limit = payload.get("limit")

        nodes = load_nodes_from_silver(doc_id=doc_id, limit=limit)

        index = rebuild_index(
            nodes=nodes,
            llm=runtime["llm"],
            storage_context=runtime["storage_context"],
        )

        return {
            "doc_id": doc_id,
            "limit": limit,
            "nodes_count": len(nodes),
            "index_class": index.__class__.__name__,
            "status": "rebuilt",
        }

    @task
    def log_result(payload: dict) -> None:
        print("=" * 60)
        print("INDEX REBUILT")
        print(payload)
        print("=" * 60)

    conf_payload = read_conf()
    loaded = load_nodes(conf_payload)
    rebuilt = rebuild(loaded)
    log_result(rebuilt)
