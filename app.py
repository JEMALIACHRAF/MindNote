"""
Application de prise de notes intelligente avec RAG et clustering
Utilise LlamaIndex, Gradio et divers LLMs pour créer un "Second Brain"
"""

# ============================================================================
#	 IMPORTS - Tous regroupés au début
# ============================================================================

# Standard library
import os
import re
import time
import asyncio
from collections import defaultdict

# Third-party libraries
import gradio as gr
import numpy as np
import chromadb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from yt_dlp import YoutubeDL

# LlamaIndex imports
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    KnowledgeGraphIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.core.node_parser import (
    TokenTextSplitter,
    SentenceSplitter,
    SentenceWindowNodeParser,
)
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.postprocessor import (
    LLMRerank,
    MetadataReplacementPostProcessor,
)
from llama_index.core.schema import TextNode
from llama_index.llms.together import TogetherLLM
from llama_index.llms.openai import OpenAI as llama_openai
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


from pipelines import silver_store
from pipelines.ingestion import process_file as process_file_pipeline
from pipelines.audio_pipeline import add_audio_file_db as add_audio_pipeline
from pipelines.indexing import load_nodes_from_silver, rebuild_index
from pipelines.metadata_service import MetadataService

from dotenv import load_dotenv
load_dotenv()  # charge .env depuis le répertoire courant



# ============================================================================
# CONFIGURATION GLOBALE
# ============================================================================

class Config:
    """Configuration centralisée de l'application"""
    BASE_DIR = "./data"
    AUDIO_CHUNK_DIR = os.path.join(BASE_DIR, "audio_chunks")
    AUDIO_UPLOAD_DIR = os.path.join(BASE_DIR, "audio_uploads")
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  #
    CLUSTERING_MAX_CLUSTERS = 10
    SIMILARITY_THRESHOLD = 0.7
    
    @classmethod
    def ensure_directories(cls):
        """Crée les répertoires nécessaires"""
        os.makedirs(cls.BASE_DIR, exist_ok=True)
        os.makedirs(cls.AUDIO_CHUNK_DIR, exist_ok=True)
        os.makedirs(cls.AUDIO_UPLOAD_DIR, exist_ok=True)



# ============================================================================
# ÉTAT GLOBAL DE L'APPLICATION
# ============================================================================

class AppState:
    """Gestion centralisée de l'état de l'application"""
    
    def __init__(self):
        self.documents = {}
        self.file_nodes = {} 
        self.note_counter = 0
        self.video_counter = 0
        self.chat_history = ""
        self.options = [
            "Strategy case study notes",
            "What's GenAI",
            "Recent news on LLMs",
            "Copyright Infr"
        ]
        
        # Indices
        self.vector_index = None
        self.kg_index = None
        self.chat_engine = None
        self.pipeline = None
        
        # LLM et embedding
        self.llm = None
        self.embedding_model = None
        self.graph_store = None
        self.storage_context = None
    
    def initialize_models(self, llm, docs_path=None):
        """Initialise les modèles et les indices"""
        print("\n" + "="*60)
        print(" INITIALISATION DE L'APPLICATION")
        print("="*60)
        
        print("\n Étape 1/8 - Configuration du LLM...")
        self.llm = llm
        Settings.llm = llm
        print("    LLM configuré")
        
        print("\n Étape 2/8 - Chargement du modèle d'embedding principal...")
        Settings.embed_model = HuggingFaceEmbedding(model_name=Config.EMBEDDING_MODEL)
        print(f"    Modèle principal chargé : {Config.EMBEDDING_MODEL}")
        
        print("\n Étape 3/8 - Chargement du modèle d'embedding pour clustering...")
        self.embedding_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("    Modèle de clustering chargé")
        
        print("\n Étape 4/8 - Initialisation du graph store...")
        self.graph_store = SimpleGraphStore()
        print("    Graph store initialisé")
        
        print("\n Étape 5/8 - Initialisation de ChromaDB...")
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            graph_store=self.graph_store
        )
        print("    ChromaDB initialisé")
        
        print("\n Étape 6/8 - Configuration du pipeline de traitement...")
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=50),
                TitleExtractor(),
                KeywordExtractor(keywords=5),
                SummaryExtractor(summaries=["self"]),
            ]
        )
        print("    Pipeline configuré avec :")
        print("      - SentenceSplitter (taille: 512, overlap: 50)")
        print("      - TitleExtractor")
        print("      - KeywordExtractor (5 mots-clés)")
        print("      - SummaryExtractor")
        
        # Chargement des documents initiaux (optionnel)
        if docs_path and os.path.exists(docs_path):
            print(f"\n Étape 7/8 - Chargement du document initial : {docs_path}")
            docs = SimpleDirectoryReader(input_files=[docs_path]).load_data()
            print(f"    {len(docs)} document(s) chargé(s)")
            
            print("\n    Traitement des documents via le pipeline...")
            nodes = self.pipeline.run(documents=docs)
            print(nodes)
            print(f"    {len(nodes)} nœuds créés")
            
            print("\n Étape 8/8 - Création des indices...")
            print("    Création du Vector Index...")
            self.vector_index = VectorStoreIndex(
                nodes=nodes,
                llm=llm,
                storage_context=self.storage_context,
            )
            print("    Vector Index créé")
            
            self.kg_index = None

            
            print("   Initialisation du chat engine...")
            self.chat_engine = self.vector_index.as_chat_engine(llm=llm, verbose=True)
            print("    Chat engine prêt")
        else:
            print("\n Étape 7/8 - Pas de document initial")
            print("   ℹ  Démarrage sans documents initiaux")
            
            print("\n Étape 8/8 - Création des indices vides...")
            print("    Création du Vector Index vide...")
            self.vector_index = VectorStoreIndex(
                nodes=[],
                llm=llm,
                storage_context=self.storage_context,
            )
            print("    Vector Index créé (vide)")
            
            self.kg_index = None

            
            print("    Initialisation du chat engine...")
            self.chat_engine = self.vector_index.as_chat_engine(llm=llm, verbose=True)
            print("    Chat engine prêt")
        
        print("\n" + "="*60)
        print("INITIALISATION TERMINÉE AVEC SUCCÈS")
        print("="*60 + "\n")


# Instance globale de l'état
app_state = AppState()


# --- AWS runtime config (SAFE) ---
import os
os.environ.setdefault("AWS_DEFAULT_REGION", "eu-north-1")

# Ne mets PAS les clés en dur ici.
# Le script utilisera automatiquement :
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - (optionnel) AWS_SESSION_TOKEN
# si elles existent déjà dans ton environnement.


# ============================================================================
# UTILITAIRES
# ============================================================================
import polars as pl
from llama_index.core import Document

def bootstrap_from_silver_documents(max_docs: int | None = None, load_full_text: bool = True):
    """
    Lit silver/documents depuis S3 (Parquet), reconstruit:
      - app_state.documents (pour Preview/Topics)
      - app_state.options (pour dropdown Search)
    Ne touche PAS à l'index Chroma (qui est déjà persistant local).
    """
    print("\n" + "="*60)
    print("🔄 BOOTSTRAP: chargement depuis Silver/documents (S3)")
    print("="*60)

    docs_glob = f"{SILVER_DOCUMENTS}/**/*.parquet"
    try:
        df = pl.read_parquet(docs_glob, storage_options=_storage_options())
    except Exception as e:
        print(f"⚠️ Impossible de lire {docs_glob}: {e}")
        return

    if df.is_empty():
        print("ℹ️ Aucune donnée dans silver/documents.")
        return

    # Optionnel : garder uniquement le dernier enregistrement par doc_id
    # (si tu append plusieurs fois le même doc_id)
    if "created_at" in df.columns:
        df = df.sort("created_at").group_by("doc_id").tail(1)

    if max_docs is not None:
        df = df.tail(max_docs)

    loaded = 0
    for row in df.iter_rows(named=True):
        doc_id = row.get("doc_id")
        if not doc_id:
            continue

        # Reconstruit Document texte (pour Preview/Topics)
        text = row.get("text") or ""
        if not load_full_text:
            # tu peux limiter si tu veux éviter trop de RAM
            text = text[:20_000]

        doc = Document(text=text)
        doc.metadata = {
            "file_name": doc_id,
            "source_type": row.get("source_type") or "",
            "raw_s3_uri": row.get("raw_path") or "",
            "title": row.get("title") or "Unknown Title",
            "keywords": row.get("keywords") or "",
            "summary": row.get("summary") or "",
            "created_at": row.get("created_at") or "",
            "content_hash": row.get("content_hash") or "",
        }

        # Stocke dans app_state comme avant
        app_state.documents[doc_id] = [doc_id, doc]

        if doc_id not in app_state.options:
            app_state.options.append(doc_id)

        loaded += 1

    print(f" Bootstrap terminé: {loaded} documents rechargés dans app_state.documents")
    print("="*60 + "\n")

import json
from llama_index.core.schema import TextNode

def load_nodes_for_doc_from_silver(doc_id: str):
    """
    Recharge les chunks d'un seul doc depuis silver/chunks (S3) -> TextNode[].
    Met aussi dans app_state.file_nodes[doc_id].
    """
    if not doc_id:
        return None

    chunks_glob = f"{SILVER_CHUNKS}/**/doc_id={doc_id}/*.parquet"
    try:
        df = pl.read_parquet(chunks_glob, storage_options=_storage_options())
    except Exception as e:
        print(f"⚠️ Impossible de lire chunks pour doc_id={doc_id}: {e}")
        return None

    if df.is_empty():
        return None

    # ordre des chunks
    if "chunk_index" in df.columns:
        df = df.sort("chunk_index")

    nodes = []
    for row in df.iter_rows(named=True):
        md = {}
        try:
            md = json.loads(row.get("metadata_json") or "{}")
        except Exception:
            md = {}

        # sécurité
        md["file_name"] = md.get("file_name") or doc_id
        md["doc_id"] = md.get("doc_id") or doc_id

        nodes.append(TextNode(
            text=row.get("chunk_text") or "",
            metadata=md
        ))

    app_state.file_nodes[doc_id] = nodes
    return nodes

def _s3_client():
    import boto3
    # boto3 lit automatiquement AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
    return boto3.client("s3")


def upload_raw_to_s3(local_path: str, kind: str) -> str:
    """Upload un fichier local vers S3, retourne s3://bucket/key"""
    import os
    import uuid

    bucket = "projetbigdata0"

    if kind == "audio":
        prefix = "raw/audio/"
    elif kind == "pdf":
        prefix = "raw/pdf/"
    elif kind == "photo":
        prefix = "raw/photo/"
    else:
        prefix = "raw/other/"

    filename = os.path.basename(local_path)
    key = prefix + f"{uuid.uuid4().hex}_{filename}"

    client = _s3_client()
    client.upload_file(local_path, bucket, key)

    return f"s3://{bucket}/{key}"

import json
import uuid
from datetime import datetime
import polars as pl

S3_BUCKET = "projetbigdata0"
SILVER_DOCUMENTS = f"s3://{S3_BUCKET}/silver/documents"
SILVER_CHUNKS = f"s3://{S3_BUCKET}/silver/chunks"
SILVER_EMBEDDINGS = f"s3://{S3_BUCKET}/silver/embeddings"

def _storage_options():
    # Pour polars + fsspec : client_kwargs doit être une STRING JSON
    region = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")

    key = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    token = os.getenv("AWS_SESSION_TOKEN")  # optionnel

    opts = {
        "client_kwargs": f'{{"region_name": "{region}"}}'
    }

    # Optionnel: si tu veux forcer key/secret au lieu de laisser AWS env auto
    if key and secret:
        opts["key"] = key
        opts["secret"] = secret
    if token:
        opts["token"] = token

    return opts


def _today_partition():
    return datetime.utcnow().strftime("%Y-%m-%d")

def _json_dumps_safe(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"

def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _sha256_text(text: str) -> str:
    import hashlib
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()

def _s3_client():
    import boto3
    return boto3.client("s3")

def upload_raw_to_s3(local_path: str, kind: str) -> str:
    """Upload un fichier local vers S3, retourne s3://bucket/key"""
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
    client = _s3_client()
    client.upload_file(local_path, S3_BUCKET, key)
    return f"s3://{S3_BUCKET}/{key}"

def write_silver_documents_row(
    doc_id: str,
    source_type: str,
    raw_path: str,
    text: str,
    title: str,
    keywords: str,
    summary: str,
    content_hash: str,
):
    date = _today_partition()
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
    df.write_parquet(out, storage_options=_storage_options())
    return out

def write_silver_chunks_and_embeddings(
    doc_id: str,
    nodes,
    embed_model_name: str,
):
    """
    Écrit chunks + embeddings.
    - chunk_id est généré ici et injecté dans metadata du node (md["chunk_id"])
    - embeddings calculés ici (Settings.embed_model.get_text_embedding)
    """
    date = _today_partition()

    chunk_rows = []
    emb_rows = []

    for idx, node in enumerate(nodes):
        chunk_id = uuid.uuid4().hex

        # ensure metadata is dict
        md = getattr(node, "metadata", None) or {}
        md = dict(md)
        md["doc_id"] = doc_id
        md["chunk_id"] = chunk_id
        node.metadata = md

        text = getattr(node, "text", "") or ""
        vector = Settings.embed_model.get_text_embedding(text)

        chunk_rows.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "chunk_text": text,
            "chunk_index": idx,
            "metadata_json": _json_dumps_safe(md),
            "created_at": datetime.utcnow().isoformat(),
        })
        emb_rows.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "vector": vector,  # list[float]
            "model": embed_model_name,
            "created_at": datetime.utcnow().isoformat(),
        })

    df_chunks = pl.DataFrame(chunk_rows)
    df_emb = pl.DataFrame(emb_rows)

    out_chunks = f"{SILVER_CHUNKS}/date={date}/doc_id={doc_id}/part-{uuid.uuid4().hex}.parquet"
    out_emb = f"{SILVER_EMBEDDINGS}/date={date}/doc_id={doc_id}/part-{uuid.uuid4().hex}.parquet"

    df_chunks.write_parquet(out_chunks, storage_options=_storage_options())
    df_emb.write_parquet(out_emb, storage_options=_storage_options())

    return out_chunks, out_emb


def refresh_preview_dropdown():
    """Rafraîchit la liste des notes disponibles pour l’onglet Note Preview."""
    return gr.Dropdown.update(choices=["All Notes"] + list(app_state.documents.keys()))

def is_youtube_url(url: str) -> bool:
    """Vérifie si une URL est un lien YouTube"""
    youtube_pattern = re.compile(
        r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+',
        re.IGNORECASE
    )
    return bool(re.match(youtube_pattern, url))


def is_image(filename: str) -> bool:
    """Vérifie si un fichier est une image"""
    import imghdr
    return imghdr.what(filename) is not None

def save_uploaded_audio_to_stable_path(audio_path: str) -> str:
    """Copie/déplace l'audio uploadé vers un chemin stable pour le conserver."""
    os.makedirs(Config.AUDIO_UPLOAD_DIR, exist_ok=True)
    stable_path = os.path.join(Config.AUDIO_UPLOAD_DIR, os.path.basename(audio_path))
    # move si possible, sinon copy
    try:
        os.replace(audio_path, stable_path)
    except Exception:
        import shutil
        shutil.copy2(audio_path, stable_path)
    return stable_path

def transcribe_audio_openai(audio_path: str, language: str = None) -> str:
    """Transcrit un fichier audio via OpenAI (whisper-1 ou gpt-4o-mini-transcribe)."""
    try:
        from openai import OpenAI
        import openai
        openai.api_key = os.environ.get("openai_key")
        client = OpenAI(api_key=os.environ.get("openai_key"))
    except Exception as e:
        raise RuntimeError(
            "SDK OpenAI manquant ou mal installé. Installe/upgrade: pip install -U openai"
        ) from e

    # Modèle: whisper-1 (classique) ou gpt-4o-mini-transcribe (souvent meilleur)
    model_name = "whisper-1"

    with open(audio_path, "rb") as f:
        # API Transcriptions (file object)
        # ref: OpenAI audio transcription docs
        resp = client.audio.transcriptions.create(
            model=model_name,
            file=f,
            # language="fr",  # optionnel si tu veux forcer
        )
    # resp.text contient le transcript
    return resp.text or ""


def normalize_text(text: str) -> str:
    """Normalise le texte pour la comparaison"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def save_audio_to_stable_path(audio: str) -> str:
    """Déplace un fichier audio vers un chemin stable"""
    if not os.path.exists(Config.AUDIO_CHUNK_DIR):
        os.makedirs(Config.AUDIO_CHUNK_DIR)
    
    stable_audio_path = os.path.join(
        Config.AUDIO_CHUNK_DIR,
        os.path.basename(audio)
    )
    os.rename(audio, stable_audio_path)
    return stable_audio_path

def keywords_to_scalar(keywords) -> str:
    """Convertit keywords (list/str/None) en string compatible Chroma."""
    if keywords is None:
        return ""
    if isinstance(keywords, str):
        return keywords
    if isinstance(keywords, (list, tuple, set)):
        return ", ".join([str(k).strip() for k in keywords if str(k).strip()])
    return str(keywords)

def keywords_to_list(keywords) -> list:
    """Convertit keywords (string 'a, b' ou list) en liste de strings pour affichage/clustering."""
    if keywords is None:
        return []
    if isinstance(keywords, (list, tuple, set)):
        return [str(k).strip() for k in keywords if str(k).strip()]
    if isinstance(keywords, str):
        return [k.strip() for k in keywords.split(",") if k.strip()]
    return [str(keywords).strip()] if str(keywords).strip() else []



def build_query_engine(modality: str, selected_note: str):
    modality_norm = (modality or "").strip().lower()

    if modality_norm != "selected note" or not selected_note:
        return app_state.vector_index.as_query_engine(similarity_top_k=5)

    nodes = app_state.file_nodes.get(selected_note)

    # ✅ NEW: lazy-load depuis Silver si absent
    if not nodes:
        nodes = load_nodes_for_doc_from_silver(selected_note)

    if nodes:
        try:
            temp_index = VectorStoreIndex(nodes=nodes, llm=app_state.llm)
            return temp_index.as_query_engine(similarity_top_k=5)
        except Exception as e:
            print(f"build_query_engine: temp_index failed for '{selected_note}': {e}")

    return app_state.vector_index.as_query_engine(similarity_top_k=5)


# ============================================================================
# GÉNÉRATION DE MÉTADONNÉES
# ============================================================================

def generate_title(text: str) -> str:
    """Génère un titre pour le contenu"""
    prompt = f"Generate a concise and descriptive title for the following content:\n{text[:500]}"
    try:
        response = app_state.llm.complete(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating title: {e}")
        return "Unknown Title"


def generate_keywords(text: str) -> list:
    """Génère des mots-clés à partir du texte"""
    prompt = f"Extract 5-10 keywords from the following content:\n{text[:500]}"
    try:
        resp = app_state.llm.complete(prompt)
        
        # Traite différents formats de réponse
        if resp.text.startswith("Keywords:"):
            lines = resp.text.split("\n")[1:]
            keywords = [line.strip("- ").strip() for line in lines if line.strip()]
        elif "," in resp.text:
            keywords = [kw.strip() for kw in resp.text.split(",")]
        else:
            keywords = resp.text.split()
        
        return list(set(kw for kw in keywords if kw))
    except Exception as e:
        print(f"Error generating keywords: {e}")
        return ["No keywords available"]


def generate_summary(text: str) -> str:
    """Génère un résumé du texte"""
    prompt = f"Summarize the following content in 2-3 sentences:\n{text[:1000]}"
    try:
        response = app_state.llm.complete(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "No summary available."


async def extract_metadata(doc: Document) -> tuple:
    """Extrait les métadonnées d'un document"""
    text_content = doc.text
    
    title = generate_title(text_content)
    keywords = generate_keywords(text_content)
    summary = generate_summary(text_content)
    
    return title, keywords, summary


# ============================================================================
# CLUSTERING ET TOPICS
# ============================================================================

def optimal_num_clusters(embeddings: np.ndarray, max_clusters: int = 10) -> int:
    """Détermine le nombre optimal de clusters"""
    if len(embeddings) < 2:
        raise ValueError("At least two embeddings required for clustering")
    
    best_score = -1
    best_num = 2
    max_clusters = min(max_clusters, len(embeddings))
    
    for k in range(2, max_clusters + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
            score = silhouette_score(embeddings, kmeans.labels_)
            if score > best_score:
                best_score = score
                best_num = k
        except ValueError as e:
            print(f"Skipping k={k}: {e}")
            continue
    
    return best_num


def deduplicate_embeddings(embeddings: list, items: list) -> tuple:
    """Élimine les embeddings dupliqués"""
    unique_embeddings = []
    unique_indices = []
    seen_embeddings = set()
    
    for idx, embedding in enumerate(embeddings):
        emb_tuple = tuple(embedding)
        if emb_tuple not in seen_embeddings:
            seen_embeddings.add(emb_tuple)
            unique_embeddings.append(embedding)
            unique_indices.append(idx)
    
    unique_items = [items[idx] for idx in unique_indices]
    return np.array(unique_embeddings), unique_items


def refine_clusters_with_similarity(
    clustered_notes: dict,
    embeddings: np.ndarray,
    threshold: float = Config.SIMILARITY_THRESHOLD
) -> dict:
    """Affine les clusters basés sur la similarité cosinus"""
    embeddings = np.array(embeddings)
    refined_clusters = defaultdict(list)
    
    for cluster_id, notes in clustered_notes.items():
        note_indices = [
            idx for idx, note in enumerate(notes)
            if isinstance(idx, int) and idx < len(embeddings)
        ]
        
        if not note_indices:
            continue
        
        cluster_embeddings = embeddings[note_indices]
        cluster_centroid = np.mean(cluster_embeddings, axis=0)
        
        for idx in note_indices:
            embedding = embeddings[idx]
            similarity = cosine_similarity(
                embedding.reshape(1, -1),
                cluster_centroid.reshape(1, -1)
            )[0][0]
            
            if similarity >= threshold:
                refined_clusters[cluster_id].append(notes[idx])
            else:
                new_cluster_id = f"refined_{len(refined_clusters)}"
                refined_clusters[new_cluster_id].append(notes[idx])
    
    return refined_clusters


def cluster_notes_and_generate_topics(notes: list) -> dict:
    """Clustering des notes et génération de topics"""
    print("\n" + "="*60)
    print(" CLUSTERING ET GÉNÉRATION DE TOPICS")
    print("="*60)
    
    print(f"\n Nombre de notes à traiter : {len(notes)}")
    
    print("\n Étape 1/7 - Vérification des titres...")
    # Génère les titres manquants
    titles_generated = 0
    for note in notes:
        if not note['title'] or note['title'] in ["Untitled", "Unknown Title"]:
            note['title'] = generate_title(note['content']) if note['content'] else "No Title"
            titles_generated += 1
    print(f"    {titles_generated} titre(s) généré(s)")
    
    print("\n Étape 2/7 - Préparation des métadonnées...")
    # Crée les métadonnées pour l'embedding
    metadata = [
        f"Title: {note['title']}, "
        f"Keywords: {', '.join([str(kw) for kw in note['keywords']])}, "
        f"Summary: {note['summary']}, "
        f"Content: {note['content'][:1000]}"
        for note in notes
    ]
    print(f"    {len(metadata)} métadonnées préparées")
    
    print("\n Étape 3/7 - Génération des embeddings...")
    print("    Calcul des vecteurs d'embedding...")
    # Génère les embeddings
    embeddings = [
        app_state.embedding_model.get_text_embedding(text)
        for text in metadata
    ]
    print(f"    {len(embeddings)} embeddings générés")
    
    print("\nÉtape 4/7 - Déduplication...")
    # Déduplique
    original_count = len(embeddings)
    embeddings, notes = deduplicate_embeddings(embeddings, notes)
    duplicates_removed = original_count - len(embeddings)
    print(f"    {duplicates_removed} doublon(s) supprimé(s)")
    print(f"    {len(notes)} notes uniques")
    
    print("\n Étape 5/7 - Détermination du nombre optimal de clusters...")
    # Détermine le nombre optimal de clusters
    num_clusters = optimal_num_clusters(embeddings, Config.CLUSTERING_MAX_CLUSTERS)
    if num_clusters < 2:
        num_clusters = 1
    print(f"    Nombre optimal de clusters : {num_clusters}")
    
    print("\n Étape 6/7 - Clustering K-means...")
    # Clustering
    print("    Calcul des clusters...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    print("   Clustering terminé")
    
    # Groupe par cluster
    clustered_notes = defaultdict(list)
    for idx, cluster_id in enumerate(clusters):
        clustered_notes[cluster_id].append(notes[idx])
    
    print("\n    Répartition par cluster :")
    for cluster_id, notes_in_cluster in clustered_notes.items():
        print(f"      Cluster {cluster_id}: {len(notes_in_cluster)} note(s)")
    
    print("\n   Affinement avec similarité cosinus...")
    # Affine avec similarité
    refined_clusters = refine_clusters_with_similarity(clustered_notes, embeddings)
    print(f"    {len(refined_clusters)} cluster(s) après affinement")
    
    print("\n Étape 7/7 - Génération des noms de topics...")
    # Génère les noms de topics
    topics = {}
    for i, (cluster_id, cluster_notes) in enumerate(refined_clusters.items(), 1):
        print(f"\n    Topic {i}/{len(refined_clusters)} ({len(cluster_notes)} notes)...")
        representative_notes = "\n".join([
            f"- {note['title']} (Keywords: {', '.join(map(str, note['keywords']))})"
            for note in cluster_notes[:5]
        ])
        
        prompt = (
            "Generate a concise topic name that represents the main theme of these notes:"
            f"\n\n{representative_notes}\n\n"
            "The topic name should be short, meaningful, and capture the essence."
        )
        
        try:
            print("      - Appel au LLM pour générer le nom...")
            response = app_state.llm.complete(prompt)
            topic_name = response.text.strip()
            print(f"       Topic: '{topic_name}'")
        except Exception as e:
            print(f"        Erreur LLM: {e}")
            topic_name = f"Topic {cluster_id + 1}"
            print(f"      ℹ  Utilisation du nom par défaut: '{topic_name}'")
        
        topics[topic_name] = cluster_notes
    
    print("\n" + "="*60)
    print(f" CLUSTERING TERMINÉ - {len(topics)} topic(s) créé(s)")
    print("="*60 + "\n")
    
    return topics


# ============================================================================
# GESTION DES DOCUMENTS
# ============================================================================

def refresh_chat_engine():
    """Actualise le chat engine"""
    try:
        if app_state.vector_index:
            app_state.chat_engine = app_state.vector_index.as_chat_engine(
                similarity_top_k=5
            )
            print("Chat engine refreshed successfully")
    except Exception as e:
        print(f"Error refreshing chat engine: {e}")

def process_file(filename: str) -> str:
    """Traite un fichier (PDF/Image/Texte), upload raw S3, écrit Silver, indexe Chroma.
       - keywords -> string (Chroma compatible)
       - app_state.file_nodes[doc_id] = nodes
       - app_state.documents[doc_id] = combined_doc (preview/topics)
       - Silver: documents + chunks + embeddings (Polars Parquet sur S3)
    """
    print("\n" + "="*60)
    print(f"📁 TRAITEMENT DU FICHIER : {os.path.basename(filename)}")
    print("="*60)

    doc_id = os.path.basename(filename)
    lower = filename.lower()

    # -------- 1) Upload RAW vers S3 ----------
    raw_s3_uri = None
    source_type = "text"
    try:
        if lower.endswith(".pdf"):
            source_type = "pdf"
            raw_s3_uri = upload_raw_to_s3(filename, "pdf")
        elif is_image(filename):
            source_type = "image"
            raw_s3_uri = upload_raw_to_s3(filename, "photo")
        else:
            # texte / autres fichiers: tu peux aussi uploader en raw/other si tu veux
            source_type = "text"
            # raw_s3_uri = upload_raw_to_s3(filename, "other")
    except Exception as e:
        print(f"⚠️ Upload S3 raw failed: {e}")

    # -------- 2) Charger et extraire texte ----------
    if is_image(filename):
        # TODO: remplace par OCR réel plus tard
        image_content = "Image content extraction to implement"
        doc = Document(text=image_content)
        doc.metadata = doc.metadata or {}
        doc.metadata["file_name"] = doc_id
        docs = [doc]
    else:
        docs = SimpleDirectoryReader(input_files=[filename]).load_data()
        for d in docs:
            if isinstance(d, Document):
                d.metadata = d.metadata or {}
                d.metadata["file_name"] = doc_id

    # Inject raw_s3_uri dans metadata doc-level
    for d in docs:
        if isinstance(d, Document):
            if raw_s3_uri:
                d.metadata["raw_s3_uri"] = raw_s3_uri
            d.metadata["source_type"] = source_type

    # -------- 3) Métadonnées (title/keywords/summary) ----------
    for doc in docs:
        if not isinstance(doc, Document):
            continue
        try:
            title, keywords, summary = asyncio.run(extract_metadata(doc))
            doc.metadata.update({
                "title": title,
                "keywords": keywords_to_scalar(keywords),
                "summary": summary,
                "file_name": doc_id,
            })
        except Exception as e:
            print(f"⚠️ Metadata extraction failed: {e}")
            doc.metadata.setdefault("title", "Unknown Title")
            doc.metadata.setdefault("keywords", "")
            doc.metadata.setdefault("summary", "")
            doc.metadata.setdefault("file_name", doc_id)

    # -------- 4) Nodes via pipeline ----------
    print("🧩 Découpage et transformations...")
    nodes = app_state.pipeline.run(documents=docs)
    print(f"✅ {len(nodes)} nœuds créés")

    # Normalise metadata nodes (Chroma flat scalar!)
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

    # -------- 5) Écriture SILVER (documents + chunks + embeddings) ----------
    combined_text = "\n\n".join([d.text for d in docs if isinstance(d, Document) and d.text])
    first_md = (docs[0].metadata or {}) if docs else {}

    title = first_md.get("title", "Unknown Title")
    keywords = first_md.get("keywords", "")
    summary = first_md.get("summary", "")

    # hash du raw si possible sinon hash du texte
    try:
        content_hash = _sha256_file(filename)
    except Exception:
        content_hash = _sha256_text(combined_text)

    try:
        write_silver_documents_row(
            doc_id=doc_id,
            source_type=source_type,
            raw_path=raw_s3_uri or "",
            text=combined_text,
            title=title,
            keywords=keywords,
            summary=summary,
            content_hash=content_hash,
        )
        write_silver_chunks_and_embeddings(
            doc_id=doc_id,
            nodes=nodes,
            embed_model_name=Config.EMBEDDING_MODEL,
        )
        print("✅ Silver écrit (documents/chunks/embeddings)")
    except Exception as e:
        print(f"⚠️ Silver write failed: {e}")

    # -------- 6) Indexation dans Chroma ----------
    print("📌 Insertion dans le Vector Index (Chroma)...")
    app_state.vector_index.insert_nodes(nodes)
    print("✅ Indexation terminée")

    # Stocke nodes par fichier (pour Selected note)
    app_state.file_nodes[doc_id] = nodes

    # -------- 7) Document combiné pour Preview/Topics ----------
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

    app_state.documents[doc_id] = [doc_id, combined_doc]

    if doc_id not in app_state.options:
        app_state.options.append(doc_id)

    refresh_chat_engine()
    print(f"\n✅ Document '{doc_id}' traité et enregistré.")
    print("="*60 + "\n")
    return doc_id


# ============================================================================
# GESTION VIDÉO/AUDIO
# ============================================================================

def download_video(url: str, filename: str = None) -> str:
    """Télécharge une vidéo YouTube"""
    output_path = './data/'
    os.makedirs(output_path, exist_ok=True)
    
    if filename is None:
        filename = "default"
    
    filename_with_extension = f"{filename}_video.mp4"
    options = {
        'format': 'best',
        'outtmpl': os.path.join(output_path, filename_with_extension),
        'quiet': False,
    }
    
    try:
        with YoutubeDL(options) as ydl:
            ydl.download([url])
        print("Download successful!")
        return filename
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None


def preprocess_video(filename: str, filepath: str = None):
    """Prétraite une vidéo (audio + frames)"""
    DATA_DIR = "./data"
    AUDIO_CHUNK_DIR = f"{DATA_DIR}/{filename}_audio_chunk"
    VIDEO_FILE_PATH = filepath or f"{DATA_DIR}/{filename}_video.mp4"
    AUDIO_FILE_PATH = f"{DATA_DIR}/{filename}_audio.mp3"
    FRAMES_DIR = f"{DATA_DIR}/{filename}_frames"
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    extract_audio(VIDEO_FILE_PATH, AUDIO_FILE_PATH)
    split_audio_into_chunks(AUDIO_FILE_PATH, AUDIO_CHUNK_DIR)
    extract_frames(VIDEO_FILE_PATH, FRAMES_DIR)
    
    print("Preprocessing completed")


# ============================================================================
# CALLBACKS GRADIO
# ============================================================================

def add_text(history: list, text: str) -> tuple:
    """Ajoute du texte à l'historique"""
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def add_audio(history: list, audio: str) -> tuple:
    """Traite un fichier audio"""
    try:
        stable_audio_path = save_audio_to_stable_path(audio)
        transcription = transcribe(stable_audio_path)
        history = history + [(transcription, None)]
        return history, None
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return history + [("Error: Could not transcribe audio.", None)], None


def add_file_db(file) -> gr.Dropdown:
    print(f"Processing file: {file.name}")
    doc_name = process_file(file.name)

    if doc_name not in app_state.options:
        app_state.options.append(doc_name)

    return gr.Dropdown.update(choices=app_state.options, value=doc_name)



def add_note(text: str) -> tuple:
    """Ajoute une note"""
    print("\n" + "-"*60)
    print(" AJOUT D'UNE NOUVELLE NOTE")
    print("-"*60)
    
    doc = Document(text=text)
    doc.metadata["file_name"] = f"Note {app_state.note_counter}"
    name = doc.metadata["file_name"]
    app_state.note_counter += 1
    
    print(f"   Nom: {name}")
    print(f"   Taille: {len(text)} caractères")
    
    app_state.documents[name] = [name, doc]
    print("   Note enregistrée")
    
    # Ajoute aux indices
    print("    Ajout aux indices...")
    nodes = app_state.pipeline.run(documents=[doc])
    app_state.vector_index.insert_nodes(nodes)
    print(f"    {len(nodes)} nœud(s) créé(s) et indexé(s)")
    
    app_state.options.append(name)
    refresh_chat_engine()
    
    print(f"    Total de notes: {len(app_state.documents)}")
    print("-"*60 + "\n")
    
    return "", gr.Dropdown.update(choices=app_state.options)

def add_audio_file_db(audio_file) -> gr.Dropdown:
    """
    Upload audio -> conserve local -> upload raw S3 -> transcrit -> Silver -> indexe.
    """
    audio_path = audio_file.name if hasattr(audio_file, "name") else str(audio_file)
    stable_path = save_uploaded_audio_to_stable_path(audio_path)

    doc_id = os.path.basename(stable_path)
    source_type = "audio"

    # -------- 1) Upload RAW audio vers S3 ----------
    raw_s3_uri = None
    try:
        raw_s3_uri = upload_raw_to_s3(stable_path, "audio")
    except Exception as e:
        print(f"⚠️ Upload S3 audio failed: {e}")

    # -------- 2) Transcription ----------
    transcript = transcribe_audio_openai(stable_path)
    if not transcript.strip():
        transcript = "(Transcription vide ou échouée)"

    # -------- 3) Document texte ----------
    doc = Document(text=transcript)
    doc.metadata = doc.metadata or {}
    doc.metadata["file_name"] = doc_id
    doc.metadata["source_type"] = source_type
    doc.metadata["audio_path"] = stable_path
    if raw_s3_uri:
        doc.metadata["raw_s3_uri"] = raw_s3_uri

    # -------- 4) Métadonnées ----------
    try:
        title, keywords, summary = asyncio.run(extract_metadata(doc))
        doc.metadata["title"] = title
        doc.metadata["keywords"] = keywords_to_scalar(keywords)
        doc.metadata["summary"] = summary
    except Exception as e:
        print(f"⚠️ Metadata extraction failed (audio): {e}")
        doc.metadata.setdefault("title", "Audio Transcript")
        doc.metadata.setdefault("keywords", "")
        doc.metadata.setdefault("summary", "")

    # -------- 5) Nodes via pipeline ----------
    nodes = app_state.pipeline.run(documents=[doc])

    # Normalise nodes metadata (Chroma)
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

    # -------- 6) Silver write (documents + chunks + embeddings) ----------
    combined_text = transcript
    content_hash = _sha256_file(stable_path)

    try:
        write_silver_documents_row(
            doc_id=doc_id,
            source_type=source_type,
            raw_path=raw_s3_uri or "",
            text=combined_text,
            title=doc.metadata.get("title", "Audio Transcript"),
            keywords=doc.metadata.get("keywords", ""),
            summary=doc.metadata.get("summary", ""),
            content_hash=content_hash,
        )
        write_silver_chunks_and_embeddings(
            doc_id=doc_id,
            nodes=nodes,
            embed_model_name=Config.EMBEDDING_MODEL,
        )
        print(" Silver écrit (audio documents/chunks/embeddings)")
    except Exception as e:
        print(f" Silver write failed (audio): {e}")

    # -------- 7) Indexation Chroma ----------
    app_state.vector_index.insert_nodes(nodes)

    # Selected note support
    app_state.file_nodes[doc_id] = nodes

    # Preview/Topics
    app_state.documents[doc_id] = [doc_id, doc]

    if doc_id not in app_state.options:
        app_state.options.append(doc_id)

    refresh_chat_engine()
    return gr.Dropdown.update(choices=app_state.options, value=doc_id)



def bot(history: list, modality: str, selected_note: str):
    """Génère une réponse du chatbot (utilise build_query_engine pour filtrer si demandé)."""
    print("\n" + "-"*60)
    print(" REQUÊTE UTILISATEUR")
    print("-"*60)

    user_input = history[-1][0]

    try:
        if isinstance(user_input, str):
            query = user_input
        else:
            query = transcribe(user_input)

        print(f"   Question: {query[:120]}...")
        engine = build_query_engine(modality, selected_note)
        resp = engine.query(query)
        # la forme de resp peut varier selon version -> on convertit en str
        response = str(resp)

        print("    Réponse générée")
    except Exception as e:
        print(f"    Erreur: {e}")
        response = "An error occurred while processing your query."

    app_state.chat_history += f"\nUser: {query}\nAI: {response}"

    history[-1][1] = ""
    for ch in response:
        history[-1][1] += ch
        time.sleep(0.01)
        yield history


def categorize_notes_by_topics() -> dict:
    """Catégorise toutes les notes par topics"""
    print("\n" + "="*60)
    print("  CATÉGORISATION DES NOTES PAR TOPICS")
    print("="*60)
    
    print("\n Collecte des métadonnées...")
    notes_metadata = []
    for note_name, (_, doc) in app_state.documents.items():
        title = doc.metadata.get("title", "Untitled")
        keywords = keywords_to_list(doc.metadata.get("keywords", ""))
        summary = doc.metadata.get("summary", "No summary")
        content = doc.text
        
        notes_metadata.append({
            "id": note_name,
            "title": title,
            "keywords": keywords,
            "summary": summary,
            "content": content
        })
    
    print(f"    {len(notes_metadata)} note(s) collectée(s)\n")
    
    if len(notes_metadata) == 0:
        print("     Aucune note à catégoriser\n")
        return {}
    
    topics = cluster_notes_and_generate_topics(notes_metadata)
    return topics


def display_notes_by_topic(topics: dict) -> str:
    """Affiche les notes par topic en markdown"""
    if not topics:
        return "No topics available"
    
    markdown_content = ""
    for topic, notes in topics.items():
        markdown_content += f"## {topic}\n"
        for note in notes:
            markdown_content += f"- **{note['title']}**: {note['summary'][:100]}...\n"
    
    return markdown_content


def refresh_topics_callback() -> str:
    """Callback pour rafraîchir les topics"""
    topics = categorize_notes_by_topics()
    return display_notes_by_topic(topics)


def update_note_preview(selected_note: str):
    if selected_note in app_state.documents:
        doc_data = app_state.documents[selected_note][1]
        metadata = doc_data.metadata or {}

        title = metadata.get("title", "")
        if not title or title.lower() == "unknown title":
            title = generate_title(doc_data.text)
            metadata["title"] = title

        keywords_raw = metadata.get("keywords", "")
        keywords_list = keywords_to_list(keywords_raw)
        if not keywords_list:
            metadata["keywords"] = keywords_to_scalar(generate_keywords(doc_data.text))
            keywords_list = keywords_to_list(metadata["keywords"])

        summary = metadata.get("summary", "")
        if not summary:
            summary = generate_summary(doc_data.text)
            metadata["summary"] = summary

        preview_content = f"## {selected_note}\n\n"
        preview_content += f"### Title:\n**{title}**\n\n"
        preview_content += f"### Keywords:\n**{', '.join(keywords_list)}**\n\n"
        preview_content += f"### Summary:\n{summary}\n\n"
        preview_content += "### Content:\n\n"
        preview_content += doc_data.text

        return preview_content

    return "No content available"




# ============================================================================
# INTERFACE GRADIO
# ============================================================================

def create_gradio_interface():
    """Crée l'interface Gradio"""
    
    css = """
    /* Ajoutez votre CSS personnalisé ici */
    """
    
    with gr.Blocks(css=css) as demo:
        # Tab: Topics View
        with gr.Tab("Topics View"):
            refresh_topics = gr.Button("Refresh Topics")
            topics_output = gr.Markdown("Notes categorized by topics will appear here.")
            refresh_topics.click(refresh_topics_callback, outputs=topics_output)
        
        # Tab: Second Brain (Chat)
        with gr.Tab("Second Brain"):
            with gr.Row():
                with gr.Column(scale=0.3):
                    search = gr.Dropdown(
                        app_state.options,
                        allow_custom_value=True,
                        label="Search",
                        info="Pick the document you want to retrieve!"
                    )

                    rag_radio = gr.Radio(
                        ["All notes", "Selected note"],
                        value="All notes",
                        label="Search Modality"
                    )

                
                with gr.Column(scale=0.7):
                    chatbot = gr.Chatbot([], elem_id="chatbot")
                    
                    with gr.Row():
                        with gr.Column(scale=0.6):
                            txt = gr.Textbox(
                                show_label=False,
                                placeholder="Enter text and press enter"
                            )
                        with gr.Column(scale=0.3):
                            audio_input = gr.Audio(type="filepath")
                        with gr.Column(scale=0.1):
                            btn = gr.UploadButton("📁")
                    
                    txt_msg = txt.submit(
                        add_text, [chatbot, txt], [chatbot, txt], queue=False
                    ).then(bot, [chatbot, rag_radio, search], chatbot)
                    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt])
                    
                    btn.upload(add_file_db, inputs=[btn], outputs=[search])
                    audio_input.stop_recording(
                        add_audio, [chatbot, audio_input], [chatbot, audio_input]
                    ).then(bot, [chatbot, rag_radio, search], chatbot)
        
        # Tab: Note Taking
        with gr.Tab("Note Taking"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("Add files", open=False):
                        text_file = gr.File(
                            file_types=["text", "application/pdf"],
                            type="file"
                        )
                        text_file.upload(add_file_db, inputs=[text_file], outputs=[search])
                    
                    with gr.Accordion("Add Media", open=False):
                        visual_file = gr.File(
                            file_types=['image', '.mp4', "application/pdf"],
                            type="file"
                        )
                        visual_file.upload(add_file_db, inputs=[visual_file], outputs=[search])
                    

                    with gr.Accordion("Add Audio", open=False):
                        audio_file = gr.File(
                            file_types=["audio"],
                            type="file",
                            label="Upload audio file"
                        )
                        audio_file.upload(add_audio_file_db, inputs=[audio_file], outputs=[search])

                
                with gr.Column(scale=2):
                    text_field = gr.Textbox(
                        label="Note Editor",
                        placeholder="Write your notes here",
                        lines=10
                    )
                    submit = gr.Button("Save note")
                    submit.click(add_note, text_field, [text_field, search])

        # Tab: Note Preview
        with gr.Tab("Note Preview"):
            refresh_preview = gr.Button(" Refresh list")
            preview = gr.Markdown("Please select a note to preview.", visible=True)

            dropdown_preview = gr.Dropdown(
                choices=["All Notes"] + list(app_state.documents.keys()),
                label="Select Note",
                value="All Notes",
            )

            refresh_preview.click(refresh_preview_dropdown, outputs=dropdown_preview)
            dropdown_preview.change(update_note_preview, inputs=dropdown_preview, outputs=preview)


    
    return demo


# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================

def main():
    """Point d'entrée principal de l'application"""
    print("\n" + ""*30)
    print("="*60)
    print("        SECOND BRAIN - Application de Notes Intelligente")
    print("="*60)
    print(""*30 + "\n")
    
    print(" Démarrage de l'application...\n")
    
    # Configuration
    print("  Configuration des répertoires...")
    Config.ensure_directories()
    print("    Répertoires vérifiés/créés\n")
    
    # Initialisation du LLM (à configurer selon vos besoins)
    print(" Initialisation du modèle LLM...")
    print("   Modèle: gpt-3.5-turbo")
    try:
        llm = llama_openai(model="gpt-3.5-turbo", api_key=os.environ.get("openai_key"))

        print("    LLM initialisé\n")
    except Exception as e:
        print(f"    Erreur lors de l'initialisation du LLM: {e}")
        print("     Vérifiez votre clé API dans le fichier .env\n")
        return
    
    # Initialisation de l'état de l'application (sans document initial)
    # Si vous avez un fichier PDF à charger au démarrage, passez son chemin ici :
    # app_state.initialize_models(llm, docs_path="chemin/vers/votre/fichier.pdf")
    app_state.initialize_models(llm, docs_path=None)
    bootstrap_from_silver_documents(load_full_text=True)

    # Création et lancement de l'interface
    print("\n Création de l'interface Gradio...")
    demo = create_gradio_interface()
    print("    Interface créée\n")
    
    print(" Lancement de l'application...")
    print("=" * 60)
    print(" L'application sera accessible sur:")
    print("   - Local:   http://localhost:7860")
    print("   - Réseau:  http://0.0.0.0:7860")
    print("=" * 60)
    print("\n Pour arrêter l'application, appuyez sur Ctrl+C\n")
    
    demo.queue()
    demo.launch(
         server_name="0.0.0.0",
         server_port=7860,
         share=False,
         debug=False
    )


if __name__ == "__main__":
    main()
