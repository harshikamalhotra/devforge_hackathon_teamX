import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import uuid


class ChromaDBClient:
    """
    ChromaDB implementation of vector database.
    Provides the same interface as LocalVectorDB for easy switching.
    """
    
    def __init__(self, dim: int = 384, db_dir: str = "vector_db_store", collection_name: str = "documents"):
        """
        Initialize ChromaDB client.
        
        Args:
            dim: Dimension of embeddings (default: 384)
            db_dir: Directory to store ChromaDB data
            collection_name: Name of the ChromaDB collection
        """
        self.dim = dim
        self.db_dir = db_dir
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(db_dir, exist_ok=True)
        
        # Initialize ChromaDB client (persistent mode)
        self.client = chromadb.PersistentClient(
            path=db_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------
    
    def reload(self):
        """
        Reload collection from disk.
        Useful when the database is updated by another process or instance.
        """
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def close(self):
        """
        Close the ChromaDB client and release resources.
        Useful for cleanup, especially on Windows where files may remain locked.
        """
        try:
            # ChromaDB client doesn't have an explicit close method,
            # but we can delete the reference to help with garbage collection
            del self.collection
            del self.client
        except Exception:
            pass
    
    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    
    def upsert_documents(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """
        Upsert documents into ChromaDB.
        
        Args:
            ids: List of document IDs
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
        """
        # ChromaDB expects embeddings as list of lists
        # Ensure embeddings are the right format
        embeddings_list = [list(emb) for emb in embeddings]
        
        # ChromaDB requires all metadata values to be strings, numbers, or booleans
        # Convert complex objects to JSON strings
        import json
        processed_metadatas = []
        for meta in metadatas:
            processed_meta = {}
            for key, value in meta.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    processed_meta[key] = value
                elif isinstance(value, list):
                    # Convert lists to JSON strings for better preservation
                    processed_meta[key] = json.dumps(value)
                elif isinstance(value, dict):
                    # Convert nested dictionaries to JSON strings
                    processed_meta[key] = json.dumps(value)
                else:
                    # Convert other types to strings
                    processed_meta[key] = str(value)
            processed_metadatas.append(processed_meta)
        
        # Upsert to ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=processed_metadatas
        )
    
    def search_vector(self, query_vector: List[float], top_k: int = 5):
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of results with id, score, and payload
        """
        if self.collection.count() == 0:
            return []
        
        # Perform similarity search
        results = self.collection.query(
            query_embeddings=[list(query_vector)],
            n_results=top_k
        )
        
        # Transform ChromaDB results to match LocalVectorDB format
        formatted_results = []
        
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else None
                metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
                
                # Convert distance to similarity score (ChromaDB uses distance, we want similarity)
                # For cosine distance: similarity = 1 - distance
                score = 1 - distance if distance is not None else 0.0
                
                # Process metadata back to original format if needed
                import json
                processed_metadata = {}
                for key, value in metadata.items():
                    # Try to convert back from JSON string representations
                    if isinstance(value, str):
                        # Check if it's a JSON string (list or dict)
                        if (value.startswith('[') and value.endswith(']')) or \
                           (value.startswith('{') and value.endswith('}')):
                            try:
                                processed_metadata[key] = json.loads(value)
                            except (json.JSONDecodeError, ValueError):
                                # Fallback to ast.literal_eval for simple cases
                                try:
                                    import ast
                                    processed_metadata[key] = ast.literal_eval(value)
                                except:
                                    processed_metadata[key] = value
                        else:
                            processed_metadata[key] = value
                    else:
                        processed_metadata[key] = value
                
                formatted_results.append({
                    "id": doc_id,
                    "score": float(score),
                    "payload": processed_metadata
                })
        
        return formatted_results
    
    def delete(self, ids: List[str]):
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        if ids:
            self.collection.delete(ids=ids)
    
    def delete_all(self):
        """Delete all documents from the collection."""
        # Get all IDs and delete them
        all_results = self.collection.get()
        if all_results['ids']:
            self.collection.delete(ids=all_results['ids'])
    
    # -------------------- CRUD Compatibility Methods --------------------
    
    def add_document(self, doc_id: str, content: Dict[str, Any]):
        """
        Add a single document with its content.
        This is a convenience method that extracts paragraphs and creates embeddings.
        Note: This method expects content to already have embeddings in paragraphs.
        For better control, use upsert_documents directly.
        """
        # Extract paragraphs with embeddings
        paragraphs = content.get("paragraphs", [])
        if not paragraphs:
            return
        
        ids = []
        embeddings = []
        metadatas = []
        
        for para in paragraphs:
            para_id = f"{doc_id}_{para.get('id', len(ids))}"
            ids.append(para_id)
            
            # Get embedding from paragraph (should be added by caller)
            embedding = para.get("embedding")
            if embedding is None:
                raise ValueError(f"Paragraph {para_id} missing embedding. Call embedder first.")
            
            embeddings.append(embedding)
            
            # Create metadata - include entity_ids from paragraph if available
            entity_ids = para.get("entity_ids", [])
            metadata = {
                "doc_id": doc_id,
                "paragraph_id": para.get("id"),
                "text": para.get("text", ""),
                "source": content.get("source", doc_id),
                "type": content.get("type", ""),
                "metadata": content.get("metadata", {}),
                "entity_ids": entity_ids  # Store entity IDs for graph retrieval
            }
            metadatas.append(metadata)
        
        self.upsert_documents(ids=ids, embeddings=embeddings, metadatas=metadatas)
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieve all paragraphs for a document by doc_id.
        Returns the document structure with all its paragraphs.
        """
        # Query ChromaDB for all documents with this doc_id
        results = self.collection.get(
            where={"doc_id": doc_id}
        )
        
        if not results['ids']:
            return {}
        
        # Build document structure
        doc_paragraphs = []
        doc_metadata = None
        
        for i, stored_id in enumerate(results['ids']):
            metadata = results['metadatas'][i] if 'metadatas' in results else {}
            
            # Extract paragraph information
            paragraph_id = metadata.get("paragraph_id", "")
            text = metadata.get("text", "")
            
            doc_paragraphs.append({
                "id": paragraph_id,
                "text": text
            })
            
            if doc_metadata is None:
                # Get the nested metadata if available
                nested_meta = metadata.get("metadata", {})
                if isinstance(nested_meta, str):
                    try:
                        import json
                        nested_meta = json.loads(nested_meta)
                    except (json.JSONDecodeError, ValueError):
                        # Fallback to ast.literal_eval for simple cases
                        try:
                            import ast
                            nested_meta = ast.literal_eval(nested_meta)
                        except:
                            nested_meta = {}
                doc_metadata = nested_meta
        
        return {
            "source": doc_id,
            "metadata": doc_metadata or {},
            "paragraphs": doc_paragraphs
        }
    
    def delete_document(self, doc_id: str):
        """
        Delete all paragraphs for a document by doc_id.
        """
        # Delete all documents with this doc_id
        self.collection.delete(
            where={"doc_id": doc_id}
        )
    
    def similarity_search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Alias for search_vector for CRUD compatibility.
        Returns results with 'text' field for easier access.
        """
        results = self.search_vector(query_vector, top_k=top_k)
        
        # Transform results to include text directly
        transformed = []
        for res in results:
            payload = res.get("payload", {})
            # Include all payload fields in metadata for easy access
            transformed.append({
                "id": res.get("id"),
                "score": res.get("score"),
                "text": payload.get("text", ""),
                "doc_id": payload.get("doc_id", ""),
                "paragraph_id": payload.get("paragraph_id", ""),
                "metadata": payload,  # Return entire payload as metadata for compatibility
                "payload": payload  # Also keep payload for direct access
            })
        
        return transformed

