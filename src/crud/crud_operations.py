"""
CRUD operations for documents, vector DB, and graph DB
-------------------------------------------------------
Integrates ingestion, embedding, vector DB, and graph DB
"""

from src.ingestion.ingest_pipeline import IngestionPipeline
from src.embedding.embedder import Embedder
from src.vector_db.qdrant_client import LocalVectorDB     # your pure Python vector DB
from src.graph_db.memgraph_client import MemgraphClient
from src.utils.config import SUPPORTED_FILE_TYPES

class CRUDOperations:
    def __init__(self):
        self.ingestion_pipeline = IngestionPipeline()
        self.embedder = Embedder()
        self.vector_db = LocalVectorDB()  # implement store, retrieve, update, delete
        
        # Try to connect to graph DB, but don't fail if it's not available
        try:
            self.graph_db = MemgraphClient()  # gqlalchemy client
            self.graph_db_available = True
        except ConnectionError:
            self.graph_db = None
            self.graph_db_available = False

    # -------------------- CREATE / ADD DOCUMENT --------------------
    def add_document(self, file_path: str):
        """
        Ingests a document, creates embeddings, stores in vector DB and graph DB.
        """
        ext = file_path.split(".")[-1].lower()
        if f".{ext}" not in SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported file type: {ext}")

        # Ingest
        doc_json = self.ingestion_pipeline.run(file_path)

        # Embed paragraphs
        for para in doc_json["paragraphs"]:
            embedding = self.embedder.encode_text(para["text"])
            para["embedding"] = embedding  # Already a list from encode_text

        # Store in vector DB
        self.vector_db.add_document(doc_id=doc_json["metadata"]["filename"], content=doc_json)

        # Store entities in graph DB (if available)
        if self.graph_db_available and self.graph_db:
            for entity in doc_json.get("entities", []):
                self.graph_db.create_entity_node(entity_id=entity["id"], label=entity.get("label", "Entity"),
                                                 metadata=entity.get("metadata", {}))

            # Optionally, store relationships if any
            for rel in doc_json.get("relationships", []):
                self.graph_db.create_relationship(
                    start_entity_id=rel["start"],
                    end_entity_id=rel["end"],
                    rel_type=rel["type"],
                    metadata=rel.get("metadata", {})
                )

        return doc_json

    # -------------------- READ --------------------
    def get_document(self, doc_id: str):
        """
        Retrieve document from vector DB
        """
        return self.vector_db.get_document(doc_id)

    # -------------------- UPDATE --------------------
    def update_document(self, doc_id: str, new_file_path: str):
        """
        Update a document by re-ingesting and updating vector DB and graph DB
        """
        # Delete old version
        self.delete_document(doc_id)

        # Add new document
        return self.add_document(new_file_path)

    # -------------------- DELETE --------------------
    def delete_document(self, doc_id: str):
        """
        Delete document from vector DB and associated entities/relationships in graph DB
        """
        # Delete from vector DB
        self.vector_db.delete_document(doc_id)

        # Optional: delete entities/relationships from graph DB if linked by doc_id
        # This requires tracking mapping from doc_id -> entities
        # TODO: implement if entity-doc mapping exists

    # -------------------- SEARCH / VECTOR RETRIEVAL --------------------
    def search_similar(self, query_text: str, top_k: int = 5):
        """
        Perform semantic search in vector DB using query text
        """
        query_embedding = self.embedder.encode_text(query_text)
        results = self.vector_db.similarity_search(query_embedding, top_k=top_k)
        return results

    # -------------------- GRAPH QUERY --------------------
    def get_entity_relationships(self, entity_id: str):
        """
        Get all relationships for an entity from graph DB
        """
        if not self.graph_db_available or not self.graph_db:
            return []
        
        query = f"""
        MATCH (n {{id: '{entity_id}'}})-[r]->(m)
        RETURN n.id AS start, type(r) AS rel_type, m.id AS end
        """
        return self.graph_db.run_query(query)
