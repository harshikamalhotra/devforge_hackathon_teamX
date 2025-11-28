"""
Project configuration file
--------------------------
Centralized configuration for paths, vector DB, graph DB, embeddings, and ingestion.
"""

from pathlib import Path

# -------------------- PROJECT PATHS --------------------
# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data folder for uploaded/sample files
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Vector DB folder (pure Python vector DB storage)
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db"
VECTOR_DB_PATH.mkdir(exist_ok=True)

# Vector DB type: "local" (pure Python) or "chromadb"
VECTOR_DB_TYPE = "chromadb"  # Using ChromaDB as the default vector database
VECTOR_DB_DIR = "vector_db_store"  # Directory for vector DB storage

# -------------------- GRAPH DB CONFIG --------------------
# Memgraph / Graph DB connection settings using gqlalchemy
GRAPH_DB_HOST = "127.0.0.1"
GRAPH_DB_PORT = 7687
GRAPH_DB_USERNAME = ""  # default for local Windows Memgraph
GRAPH_DB_PASSWORD = ""  # default for local Windows Memgraph

# -------------------- EMBEDDING CONFIG --------------------
# Sentence transformer model for embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Maximum text length per embedding chunk (optional, for chunking)
MAX_CHUNK_LENGTH = 500  # number of characters

# -------------------- INGESTION CONFIG --------------------
# Supported file types for ingestion
SUPPORTED_FILE_TYPES = {".txt", ".pdf", ".docx", ".csv"}

# OCR flag (not used currently)
USE_OCR = False

# -------------------- OTHER SETTINGS --------------------
# Placeholder for any future constants or feature toggles
