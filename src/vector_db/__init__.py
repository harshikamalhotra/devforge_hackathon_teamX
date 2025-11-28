from .qdrant_client import LocalVectorDB
from .chromadb_client import ChromaDBClient
from src.utils.config import VECTOR_DB_TYPE, VECTOR_DB_DIR


def get_vector_db(dim: int = 384, db_dir: str = None):
    """
    Factory function to get the configured vector database.
    
    Args:
        dim: Dimension of embeddings (default: 384)
        db_dir: Directory for vector DB storage (defaults to VECTOR_DB_DIR from config)
    
    Returns:
        Vector database instance (LocalVectorDB or ChromaDBClient)
    """
    if db_dir is None:
        db_dir = VECTOR_DB_DIR
    
    if VECTOR_DB_TYPE == "chromadb":
        return ChromaDBClient(dim=dim, db_dir=db_dir)
    else:
        return LocalVectorDB(dim=dim, db_dir=db_dir)


# Export both implementations and factory function
__all__ = ["LocalVectorDB", "ChromaDBClient", "get_vector_db"]