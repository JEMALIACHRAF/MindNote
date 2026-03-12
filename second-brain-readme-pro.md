# 🧠 Second Brain --- RAG Notes System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![LlamaIndex](https://img.shields.io/badge/RAG-LlamaIndex-purple)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green)
![Airflow](https://img.shields.io/badge/Orchestration-Airflow-red)
![AWS](https://img.shields.io/badge/Storage-AWS%20S3-yellow)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

An **AI-powered Second Brain** for managing notes, documents, and audio
using **Retrieval-Augmented Generation (RAG)**.

The system combines:

-   **Gradio** → UI
-   **LlamaIndex** → RAG framework
-   **ChromaDB** → Vector database
-   **OpenAI** → LLM + transcription
-   **AWS S3** → Data lake (silver layer)
-   **Apache Airflow** → Data pipeline orchestration

------------------------------------------------------------------------

# ✨ Features

### 🧠 Chat with your notes

Ask questions about documents using RAG.

### 📄 Document ingestion

Supports:

-   PDF
-   text
-   images
-   audio transcription

### 🧩 Automatic metadata

LLM generates:

-   titles
-   keywords
-   summaries

### 🔎 Semantic search

Vector similarity search powered by **ChromaDB**.

### 🗂️ Topic clustering

Notes grouped automatically using embeddings + clustering.

### ⚙️ Airflow pipelines

Automates ingestion, embeddings, and index rebuild.

------------------------------------------------------------------------

# 🏗 Architecture

``` mermaid
flowchart LR

User -->|Upload| GradioUI

GradioUI --> DocumentPipeline
GradioUI --> AudioPipeline

DocumentPipeline --> MetadataLLM
AudioPipeline --> WhisperTranscription

MetadataLLM --> Chunking
WhisperTranscription --> Chunking

Chunking --> Embeddings

Embeddings --> ChromaDB
Embeddings --> S3Silver

S3Silver --> Airflow

Airflow --> IndexRebuild
IndexRebuild --> ChromaDB

ChromaDB --> RAGQueryEngine
RAGQueryEngine --> GradioUI
```

------------------------------------------------------------------------

# 📊 Data Architecture (Lakehouse style)

``` mermaid
flowchart TB

RawFiles -->|upload| S3Raw

S3Raw --> Processing
Processing --> MetadataExtraction
Processing --> Chunking
Processing --> Embeddings

MetadataExtraction --> SilverDocuments
Chunking --> SilverChunks
Embeddings --> SilverEmbeddings

SilverDocuments --> ChromaDB
SilverChunks --> ChromaDB
SilverEmbeddings --> ChromaDB
```

------------------------------------------------------------------------

# ⚙️ Airflow Pipeline

``` mermaid
flowchart LR

upload_file --> extract_text
extract_text --> generate_metadata
generate_metadata --> chunk_document
chunk_document --> compute_embeddings
compute_embeddings --> write_silver_tables
write_silver_tables --> update_chroma_index
```

------------------------------------------------------------------------

# 📂 Project Structure

    second-brain/
    │
    ├── app.py
    ├── requirements.txt
    ├── README.md
    │
    ├── chroma_db/
    ├── data/
    │
    ├── pipelines/
    │   ├── silver_store.py
    │   ├── ingestion.py
    │   ├── audio_pipeline.py
    │   ├── indexing.py
    │   └── metadata_service.py
    │
    ├── dags/
    │   ├── ingest_document_dag.py
    │   ├── ingest_audio_dag.py
    │   └── rebuild_index_dag.py

------------------------------------------------------------------------

# 🚀 Quick Start

## 1 Install dependencies

``` bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install apache-airflow
```

------------------------------------------------------------------------

# 🔑 Environment Variables

Create a `.env` file:

    openai_key=YOUR_OPENAI_API_KEY

    AWS_ACCESS_KEY_ID=YOUR_KEY
    AWS_SECRET_ACCESS_KEY=YOUR_SECRET
    AWS_DEFAULT_REGION=eu-north-1

    S3_BUCKET=projetbigdata0
    CHROMA_DB_PATH=./chroma_db
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

------------------------------------------------------------------------

# ▶️ Run the App

``` bash
python app.py
```

Open:

    http://localhost:7860

------------------------------------------------------------------------

# ⚡ Run Airflow

Initialize:

``` bash
airflow db init
```

Create user:

``` bash
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  --password admin
```

Start services:

``` bash
airflow scheduler
airflow webserver --port 8080
```

Open:

    http://localhost:8080

------------------------------------------------------------------------

# 🧪 Example DAG Trigger

Document ingestion:

``` json
{
  "file_path": "/path/to/document.pdf"
}
```

Audio ingestion:

``` json
{
  "audio_path": "/path/to/audio.mp3"
}
```

------------------------------------------------------------------------

# 🛠 Tech Stack

  Layer           Technology
  --------------- ----------------------
  UI              Gradio
  RAG Framework   LlamaIndex
  Vector DB       ChromaDB
  LLM             OpenAI
  Storage         AWS S3
  Orchestration   Airflow
  Embeddings      SentenceTransformers
  Processing      Polars

------------------------------------------------------------------------

# 📈 Future Improvements

-   Docker deployment
-   Kubernetes scaling
-   LangChain agents
-   streaming chat
-   multi-user authentication

------------------------------------------------------------------------

# 📜 License

MIT License
