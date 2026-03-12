import asyncio
import os
from dataclasses import dataclass

from llama_index.core import Document, SimpleDirectoryReader


@dataclass
class IngestionResult:
    doc_id: str
    source_type: str
    raw_s3_uri: str | None
    combined_text: str
    title: str
    keywords: str
    summary: str
    nodes: list
    combined_doc: Document


def keywords_to_scalar(keywords) -> str:
    if keywords is None:
        return ""
    if isinstance(keywords, str):
        return keywords
    if isinstance(keywords, (list, tuple, set)):
        return ", ".join([str(k).strip() for k in keywords if str(k).strip()])
    return str(keywords)


def is_image(filename: str) -> bool:
    import imghdr
    return imghdr.what(filename) is not None


async def extract_metadata_async(doc: Document, metadata_service) -> tuple[str, str, str]:
    title = metadata_service.generate_title(doc.text)
    keywords = metadata_service.generate_keywords(doc.text)
    summary = metadata_service.generate_summary(doc.text)
    return title, keywords_to_scalar(keywords), summary


def infer_source_type(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return "pdf"
    if is_image(filename):
        return "image"
    return "text"


def load_documents_from_file(filename: str, doc_id: str, source_type: str) -> list[Document]:
    if source_type == "image":
        image_content = "Image content extraction to implement"
        doc = Document(text=image_content)
        doc.metadata = {"file_name": doc_id}
        return [doc]

    docs = SimpleDirectoryReader(input_files=[filename]).load_data()
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["file_name"] = doc_id
    return docs


def process_file(
    *,
    filename: str,
    pipeline,
    vector_index,
    embed_model,
    embed_model_name: str,
    metadata_service,
    silver_store,
) -> IngestionResult:
    doc_id = os.path.basename(filename)
    source_type = infer_source_type(filename)

    raw_s3_uri = None
    try:
        if source_type == "pdf":
            raw_s3_uri = silver_store.upload_raw_to_s3(filename, "pdf")
        elif source_type == "image":
            raw_s3_uri = silver_store.upload_raw_to_s3(filename, "photo")
    except Exception as e:
        print(f"raw upload failed: {e}")

    docs = load_documents_from_file(filename, doc_id, source_type)

    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["file_name"] = doc_id
        d.metadata["source_type"] = source_type
        if raw_s3_uri:
            d.metadata["raw_s3_uri"] = raw_s3_uri

    for doc in docs:
        try:
            title, keywords, summary = asyncio.run(
                extract_metadata_async(doc, metadata_service)
            )
            doc.metadata.update({
                "title": title,
                "keywords": keywords,
                "summary": summary,
                "file_name": doc_id,
            })
        except Exception as e:
            print(f"metadata extraction failed: {e}")
            doc.metadata.setdefault("title", "Unknown Title")
            doc.metadata.setdefault("keywords", "")
            doc.metadata.setdefault("summary", "")

    nodes = pipeline.run(documents=docs)

    for n in nodes:
        md = getattr(n, "metadata", None) or {}
        md = dict(md)
        md["file_name"] = str(md.get("file_name") or doc_id)
        md["source_type"] = source_type
        if raw_s3_uri:
            md["raw_s3_uri"] = raw_s3_uri
        if "keywords" in md:
            md["keywords"] = keywords_to_scalar(md.get("keywords"))
        n.metadata = md

    combined_text = "\n\n".join([
        d.text for d in docs if isinstance(d, Document) and d.text
    ])
    first_md = (docs[0].metadata or {}) if docs else {}

    title = first_md.get("title", "Unknown Title")
    keywords = first_md.get("keywords", "")
    summary = first_md.get("summary", "")

    try:
        content_hash = silver_store.sha256_file(filename)
    except Exception:
        content_hash = silver_store.sha256_text(combined_text)

    try:
        silver_store.write_silver_documents_row(
            doc_id=doc_id,
            source_type=source_type,
            raw_path=raw_s3_uri or "",
            text=combined_text,
            title=title,
            keywords=keywords,
            summary=summary,
            content_hash=content_hash,
        )
        silver_store.write_silver_chunks_and_embeddings(
            doc_id=doc_id,
            nodes=nodes,
            embed_model=embed_model,
            embed_model_name=embed_model_name,
        )
    except Exception as e:
        print(f"silver write failed: {e}")

    vector_index.insert_nodes(nodes)

    combined_doc = Document(text=combined_text)
    combined_doc.metadata = {
        "file_name": doc_id,
        "title": title,
        "keywords": keywords,
        "summary": summary,
        "source_type": source_type,
    }
    if raw_s3_uri:
        combined_doc.metadata["raw_s3_uri"] = raw_s3_uri

    return IngestionResult(
        doc_id=doc_id,
        source_type=source_type,
        raw_s3_uri=raw_s3_uri,
        combined_text=combined_text,
        title=title,
        keywords=keywords,
        summary=summary,
        nodes=nodes,
        combined_doc=combined_doc,
    )
