from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable

from llama_index.core import Settings, StorageContext
from llama_index.core.extractors import SummaryExtractor, TitleExtractor, KeywordExtractor
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

from pipelines import silver_store
from pipelines.audio_pipeline import add_audio_file_db
from pipelines.metadata_service import MetadataService


default_args = {
    "owner": "ashra",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

DAG_ID = "ingest_second_brain_audio"


def save_uploaded_audio_to_stable_path(audio_path: str) -> str:
    base_dir = os.getenv("AUDIO_UPLOAD_DIR", "./data/audio_uploads")
    os.makedirs(base_dir, exist_ok=True)

    stable_path = os.path.join(base_dir, os.path.basename(audio_path))

    try:
        os.replace(audio_path, stable_path)
    except Exception:
        import shutil
        shutil.copy2(audio_path, stable_path)

    return stable_path


def transcribe_audio_openai(audio_path: str, language: str | None = None) -> str:
    from openai import OpenAI

    api_key = os.getenv("openai_key") or Variable.get("openai_key")
    client = OpenAI(api_key=api_key)

    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
        )

    return resp.text or ""


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

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=50),
            TitleExtractor(),
            KeywordExtractor(keywords=5),
            SummaryExtractor(summaries=["self"]),
        ]
    )

    vector_index = VectorStoreIndex(
        nodes=[],
        llm=llm,
        storage_context=storage_context,
    )

    metadata_service = MetadataService(llm)

    return {
        "llm": llm,
        "embed_model": embed_model,
        "embedding_model_name": embedding_model_name,
        "pipeline": pipeline,
        "vector_index": vector_index,
        "metadata_service": metadata_service,
    }


with DAG(
    dag_id=DAG_ID,
    description="Ingestion audio dans Second Brain",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["second-brain", "audio", "rag"],
) as dag:

    @task
    def validate_input(conf: dict | None = None) -> str:
        if not conf:
            raise ValueError("DAG déclenché sans configuration.")
        audio_path = conf.get("audio_path")
        if not audio_path:
            raise ValueError("Le champ 'audio_path' est obligatoire.")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Fichier audio introuvable: {audio_path}")
        return audio_path

    @task
    def ingest_audio(audio_path: str) -> dict:
        runtime = build_runtime()

        result = add_audio_file_db(
            audio_path=audio_path,
            save_uploaded_audio_to_stable_path=save_uploaded_audio_to_stable_path,
            transcribe_audio_openai=transcribe_audio_openai,
            pipeline=runtime["pipeline"],
            vector_index=runtime["vector_index"],
            embed_model=runtime["embed_model"],
            embed_model_name=runtime["embedding_model_name"],
            metadata_service=runtime["metadata_service"],
            silver_store=silver_store,
        )

        return {
            "doc_id": result.doc_id,
            "title": result.title,
            "summary": result.summary,
            "raw_s3_uri": result.raw_s3_uri or "",
            "transcript_length": len(result.transcript or ""),
            "nodes_count": len(result.nodes),
            "stable_path": result.stable_path,
        }

    @task
    def log_result(payload: dict) -> None:
        print("=" * 60)
        print("AUDIO INGESTED")
        print(payload)
        print("=" * 60)

    ap = validate_input()
    out = ingest_audio(ap)
    log_result(out)
