import asyncio
import os
from dataclasses import dataclass

from llama_index.core import Document


@dataclass
class AudioIngestionResult:
    doc_id: str
    raw_s3_uri: str | None
    transcript: str
    title: str
    keywords: str
    summary: str
    nodes: list
    combined_doc: Document
    stable_path: str


def keywords_to_scalar(keywords) -> str:
    if keywords is None:
        return ""
    if isinstance(keywords, str):
        return keywords
    if isinstance(keywords, (list, tuple, set)):
        return ", ".join([str(k).strip() for k in keywords if str(k).strip()])
    return str(keywords)


async def extract_metadata_async(doc: Document, metadata_service) -> tuple[str, str, str]:
    title = metadata_service.generate_title(doc.text)
    keywords = metadata_service.generate_keywords(doc.text)
    summary = metadata_service.generate_summary(doc.text)
    return title, keywords_to_scalar(keywords), summary


def add_audio_file_db(
    *,
    audio_path: str,
    save_uploaded_audio_to_stable_path,
    transcribe_audio_openai,
    pipeline,
    vector_index,
    embed_model,
    embed_model_name: str,
    metadata_service,
    silver_store,
) -> AudioIngestionResult:
    stable_path = save_uploaded_audio_to_stable_path(audio_path)
    doc_id = os.path.basename(stable_path)
    source_type = "audio"

    raw_s3_uri = None
    try:
        raw_s3_uri = silver_store.upload_raw_to_s3(stable_path, "audio")
    except Exception as e:
        print(f"audio upload failed: {e}")

    transcript = transcribe_audio_openai(stable_path)
    if not transcript.strip():
        transcript = "(Transcription vide ou échouée)"

    doc = Document(text=transcript)
    doc.metadata = {
        "file_name": doc_id,
        "source_type": source_type,
        "audio_path": stable_path,
    }
    if raw_s3_uri:
        doc.metadata["raw_s3_uri"] = raw_s3_uri

    try:
        title, keywords, summary = asyncio.run(
            extract_metadata_async(doc, metadata_service)
        )
        doc.metadata["title"] = title
        doc.metadata["keywords"] = keywords
        doc.metadata["summary"] = summary
    except Exception as e:
        print(f"audio metadata extraction failed: {e}")
        doc.metadata.setdefault("title", "Audio Transcript")
        doc.metadata.setdefault("keywords", "")
        doc.metadata.setdefault("summary", "")

    nodes = pipeline.run(documents=[doc])

    for n in nodes:
        md = getattr(n, "metadata", None) or {}
        md = dict(md)
        md["file_name"] = doc_id
        md["source_type"] = source_type
        md["audio_path"] = stable_path
        if raw_s3_uri:
            md["raw_s3_uri"] = raw_s3_uri
        if "keywords" in md:
            md["keywords"] = keywords_to_scalar(md.get("keywords"))
        n.metadata = md

    try:
        content_hash = silver_store.sha256_file(stable_path)
    except Exception:
        content_hash = silver_store.sha256_text(transcript)

    try:
        silver_store.write_silver_documents_row(
            doc_id=doc_id,
            source_type=source_type,
            raw_path=raw_s3_uri or "",
            text=transcript,
            title=doc.metadata.get("title", "Audio Transcript"),
            keywords=doc.metadata.get("keywords", ""),
            summary=doc.metadata.get("summary", ""),
            content_hash=content_hash,
        )
        silver_store.write_silver_chunks_and_embeddings(
            doc_id=doc_id,
            nodes=nodes,
            embed_model=embed_model,
            embed_model_name=embed_model_name,
        )
    except Exception as e:
        print(f"audio silver write failed: {e}")

    vector_index.insert_nodes(nodes)

    return AudioIngestionResult(
        doc_id=doc_id,
        raw_s3_uri=raw_s3_uri,
        transcript=transcript,
        title=doc.metadata.get("title", "Audio Transcript"),
        keywords=doc.metadata.get("keywords", ""),
        summary=doc.metadata.get("summary", ""),
        nodes=nodes,
        combined_doc=doc,
        stable_path=stable_path,
    )
