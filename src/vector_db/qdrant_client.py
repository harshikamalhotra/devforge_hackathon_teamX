import os
import json
import numpy as np
from typing import List, Dict, Any


class LocalVectorDB:
    def __init__(self, dim: int = 384, db_dir: str = "vector_db_store"):
        self.dim = dim
        self.db_dir = db_dir

        self.vec_path = os.path.join(db_dir, "vectors.npy")
        self.meta_path = os.path.join(db_dir, "metadata.json")

        os.makedirs(db_dir, exist_ok=True)

        self._load()

    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------

    def _load(self):
        """Load vectors + metadata if available."""
        if os.path.exists(self.vec_path) and os.path.exists(self.meta_path):
            self.vectors = np.load(self.vec_path)
            with open(self.meta_path, "r") as f:
                meta = json.load(f)
                self.ids = meta["ids"]
                self.payloads = meta["payloads"]
        else:
            self.vectors = np.zeros((0, self.dim), dtype=np.float32)
            self.ids = []
            self.payloads = {}

    def _save(self):
        np.save(self.vec_path, self.vectors)
        with open(self.meta_path, "w") as f:
            json.dump({
                "ids": self.ids,
                "payloads": self.payloads
            }, f)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def upsert_documents(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        new_vectors = np.array(embeddings, dtype=np.float32)

        # Append to existing database
        self.vectors = np.vstack([self.vectors, new_vectors])

        for i, doc_id in enumerate(ids):
            self.ids.append(doc_id)
            self.payloads[doc_id] = metadatas[i]

        self._save()

    def search_vector(self, query_vector: List[float], top_k: int = 5):
        if len(self.vectors) == 0:
            return []

        q = np.array(query_vector, dtype=np.float32).reshape(1, -1)

        # Cosine similarity = 1 - cosine distance
        dot_products = np.dot(self.vectors, q.T).reshape(-1)
        norms = (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(q))
        similarities = dot_products / norms

        # Sort top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "id": self.ids[idx],
                "score": float(similarities[idx]),
                "payload": self.payloads.get(self.ids[idx], {})
            })

        return results

    def delete(self, ids: List[str]):
        indices_to_keep = [
            i for i, doc_id in enumerate(self.ids)
            if doc_id not in ids
        ]

        self.vectors = self.vectors[indices_to_keep]
        self.ids = [self.ids[i] for i in indices_to_keep]
        self.payloads = {doc_id: self.payloads[doc_id] for doc_id in self.ids}

        self._save()

    def delete_all(self):
        self.vectors = np.zeros((0, self.dim), dtype=np.float32)
        self.ids = []
        self.payloads = {}
        self._save()

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
            
            # Create metadata
            metadata = {
                "doc_id": doc_id,
                "paragraph_id": para.get("id"),
                "text": para.get("text", ""),
                "source": content.get("source", doc_id),
                "type": content.get("type", ""),
                "metadata": content.get("metadata", {})
            }
            metadatas.append(metadata)
        
        self.upsert_documents(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieve all paragraphs for a document by doc_id.
        Returns the document structure with all its paragraphs.
        """
        # Find all paragraphs for this document
        doc_paragraphs = []
        doc_metadata = None
        
        for stored_id in self.ids:
            payload = self.payloads.get(stored_id, {})
            if payload.get("doc_id") == doc_id:
                doc_paragraphs.append({
                    "id": payload.get("paragraph_id", ""),
                    "text": payload.get("text", "")
                })
                if doc_metadata is None:
                    doc_metadata = payload.get("metadata", {})
        
        if not doc_paragraphs:
            return {}
        
        return {
            "source": doc_id,
            "metadata": doc_metadata or {},
            "paragraphs": doc_paragraphs
        }

    def delete_document(self, doc_id: str):
        """
        Delete all paragraphs for a document by doc_id.
        """
        # Find all IDs for this document
        ids_to_delete = [
            stored_id for stored_id in self.ids
            if self.payloads.get(stored_id, {}).get("doc_id") == doc_id
        ]
        
        if ids_to_delete:
            self.delete(ids_to_delete)

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