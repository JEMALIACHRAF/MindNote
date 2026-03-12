# 🧠 Second Brain --- Gradio + RAG + Airflow

Application de prise de notes intelligente basée sur : - **Gradio** -
**LlamaIndex** - **ChromaDB** - **OpenAI** - **AWS S3** - **Apache
Airflow**

Cette application permet de : - stocker et indexer des documents -
discuter avec ses notes (RAG) - traiter des fichiers audio - orchestrer
les pipelines de données avec Airflow

------------------------------------------------------------------------

# 📂 Structure du projet

second-brain/ │ ├── app.py ├── requirements.txt ├── .env │ ├──
chroma_db/ ├── data/ │ ├── dags/ │ ├── ingest_document_dag.py │ ├──
ingest_audio_dag.py │ └── rebuild_index_dag.py │ ├── pipelines/ │ ├──
**init**.py │ ├── silver_store.py │ ├── ingestion.py │ ├──
audio_pipeline.py │ ├── indexing.py │ └── metadata_service.py

------------------------------------------------------------------------

# ⚙️ Prérequis

-   Python **3.10+**
-   pip
-   une clé **OpenAI**
-   un compte **AWS S3**
-   Apache **Airflow** (optionnel mais recommandé)

------------------------------------------------------------------------

# 📦 Installation

Créer un environnement virtuel.

## Windows

python -m venv venv venv`\Scripts`{=tex}`\activate`{=tex}

## Linux / macOS

python -m venv venv source venv/bin/activate

Installer les dépendances :

pip install -r requirements.txt pip install apache-airflow

------------------------------------------------------------------------

# 🔑 Configuration

Créer un fichier `.env` :

openai_key=YOUR_OPENAI_API_KEY

AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET AWS_DEFAULT_REGION=eu-north-1

S3_BUCKET=projetbigdata0 CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
AUDIO_UPLOAD_DIR=./data/audio_uploads

------------------------------------------------------------------------

# ▶️ Lancer l'application

python app.py

Puis ouvrir :

http://localhost:7860

------------------------------------------------------------------------

# ⚡ Lancer Airflow

Définir le dossier Airflow.

Linux / macOS

export AIRFLOW_HOME=\$(pwd)/airflow_home

Windows PowerShell

$env:AIRFLOW_HOME = "$(Get-Location)`\airflow`{=tex}\_home"

------------------------------------------------------------------------

Initialiser Airflow

airflow db init

Créer un utilisateur admin

airflow users create --username admin --firstname Admin --lastname User
--role Admin --email admin@example.com --password admin

------------------------------------------------------------------------

Lancer le scheduler

airflow scheduler

Dans un second terminal :

airflow webserver --port 8080

Airflow sera disponible sur :

http://localhost:8080

------------------------------------------------------------------------

# 🚀 Déclencher les DAGs

## Ingestion document

DAG : ingest_second_brain_document

Configuration :

{ "file_path": "/path/to/document.pdf" }

------------------------------------------------------------------------

## Ingestion audio

DAG : ingest_second_brain_audio

Configuration :

{ "audio_path": "/path/to/audio.mp3" }

------------------------------------------------------------------------

## Reconstruction index

DAG : rebuild_second_brain_index

Configuration optionnelle :

{ "doc_id": "document.pdf", "limit": 100 }

------------------------------------------------------------------------

# 🧪 Vérifications

Tester les imports pipelines :

python -c "from pipelines.ingestion import process_file; print('OK')"

Vérifier les DAGs Airflow :

airflow dags list

------------------------------------------------------------------------

# 🛑 Arrêter l'application

CTRL + C dans le terminal.

------------------------------------------------------------------------

# 🧑‍💻 Stack technique

-   Python
-   Gradio
-   LlamaIndex
-   ChromaDB
-   OpenAI API
-   Sentence Transformers
-   Polars
-   AWS S3
-   Apache Airflow
