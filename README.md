# 🧠 MindNote – Intelligent Notes App

Application de prise de notes intelligente basée sur **RAG (Retrieval-Augmented Generation)** avec **LlamaIndex, ChromaDB et Gradio**.
Elle permet de :

* stocker des notes et documents
* faire de la recherche sémantique
* discuter avec ses notes (chat RAG)
* catégoriser automatiquement les notes par topics
* traiter texte, PDF, images et audio

---

# ⚙️ Prérequis

* Python **3.10+**
* pip
* clé **OpenAI API**
*  accès AWS si tu utilises S3

---

# 📥 Installation

Cloner ou télécharger le projet :

```bash
git clone <repo>
cd second-brain
```

Créer un environnement virtuel :

```bash
python -m venv venv
```

Activer l'environnement :

**Windows**

```bash
venv\Scripts\activate
```

**Mac / Linux**

```bash
source venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

# 🔑 Configuration

Créer un fichier `.env` à la racine du projet :

```
openai_key=YOUR_OPENAI_API_KEY
```

Optionnel (si utilisation AWS S3) :

```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=eu-north-1
```

---

# ▶️ Lancer l'application

Depuis le dossier du projet :

```bash
python app.py
```

L'application démarre un serveur **Gradio**.

---

# 🌐 Accéder à l'application

Ouvrir dans le navigateur :

```
http://localhost:7860
```

---

# 📂 Structure du projet

```
second-brain/
│
├── app.py
├── requirements.txt
├── .env
├── chroma_db/
├── data/
├── bootstrap_index_from_silver_polars.py
├── silver_writer_polars.py
└── venv/
```

---

# 🚀 Fonctionnalités principales

### 🧠 Chat avec vos notes

Interface RAG permettant de poser des questions sur les documents indexés.

### 🗂️ Topics automatiques

Clustering des notes pour générer des catégories automatiquement.

### 📄 Import de documents

Support pour :

* texte
* PDF
* images
* audio
* vidéos YouTube

### 🔎 Recherche sémantique

Recherche vectorielle via **ChromaDB**.

---

# 🛑 Arrêter l'application

Dans le terminal :

```
CTRL + C
```

---

# 🧑‍💻 Stack technique

* **Python**
* **Gradio**
* **LlamaIndex**
* **ChromaDB**
* **Sentence Transformers**
* **OpenAI API**
* **Polars**
* **AWS S3**

---

