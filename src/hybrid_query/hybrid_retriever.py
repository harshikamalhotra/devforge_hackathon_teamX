"""
Hybrid Retriever combining vector similarity + graph relational retrieval.

Based on GraphRAG / HybridRAG architecture:
1. Use vector embeddings to find top-k semantically relevant chunks/documents.
2. Identify entities from those documents.
3. Use graph DB (Cypher) to expand context via relationships.
4. Merge vector + graph results, score & re-rank.
"""

from typing import List, Dict, Any, Tuple
from src.embedding.embedder import Embedder
from src.vector_db.qdrant_client import LocalVectorDB  # or your custom client
from src.graph_db.memgraph_client import MemgraphClient     # using gqlalchemy or other client
from src.utils.config import EMBEDDING_MODEL_NAME

class HybridRetriever:
    def __init__(
        self,
        top_k_vectors: int = 5,
        top_k_final: int = 5
    ):
        self.embedder = Embedder(model_name=EMBEDDING_MODEL_NAME)
        self.vector_db = LocalVectorDB()
        
        # Try to connect to graph DB, but don't fail if it's not available
        try:
            self.graph_db = MemgraphClient()
            self.graph_db_available = True
        except ConnectionError:
            self.graph_db = None
            self.graph_db_available = False
        
        self.top_k_vectors = top_k_vectors
        self.top_k_final = top_k_final

    def retrieve(self, query_text: str) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval for a query. Returns list of results with combined scores.
        Each result includes:
          - vector result (text chunk, metadata, vector_score)
          - optionally related graph context (entities / relationships)
          - total_score (vector_score + graph_score)
        """
        # 1. Vector semantic search
        query_emb = self.embedder.encode_text(query_text)
        vector_results = self.vector_db.similarity_search(query_emb, top_k=self.top_k_vectors)

        # If no graph DB or no entities, just return vector results
        if not vector_results:
            return []

        # 2. Collect entity IDs from vector result payloads
        entity_ids = set()
        for res in vector_results:
            # similarity_search returns results with 'metadata' field containing the payload
            # Check metadata first (which contains the payload data)
            metadata = res.get("metadata", {})
            
            # Also check if there's a direct payload field
            payload = res.get("payload", {})
            
            # Try to get entity_ids from metadata or payload
            eids = metadata.get("entity_ids") or payload.get("entity_ids")
            if isinstance(eids, list) and len(eids) > 0:
                entity_ids.update(eids)

        # 3. Graph-based expansion: fetch related entities/contexts
        graph_contexts = []
        if self.graph_db_available and self.graph_db and entity_ids:
            for eid in entity_ids:
                # Basic 1-hop neighborhood query — modify as needed for deeper traversal
                cypher = (
                    f"MATCH (n {{id: '{eid}'}})-[r]-(m) "
                    "RETURN n.id AS source_id, type(r) AS rel_type, m.id AS related_id"
                )
                try:
                    rels = self.graph_db.run_query(cypher)
                    graph_contexts.extend(rels)
                except Exception as e:
                    # skip on any graph errors
                    continue

        # 4. Combine & score
        combined: List[Dict[str, Any]] = []
        for res in vector_results:
            vec_score = res.get("score", 0.0)
            metadata = res.get("metadata", {})
            payload = res.get("payload", {})
            
            # Get entity_ids from either metadata or payload
            eids = metadata.get("entity_ids") or payload.get("entity_ids", [])
            if not isinstance(eids, list):
                eids = []

            # Graph score: count how many related entities appear in graph_contexts
            graph_score = 0
            related_entities = []
            for ctx in graph_contexts:
                if ctx.get("source_id") in eids or ctx.get("related_id") in eids:
                    graph_score += 1
                    related_entities.append(ctx)

            total_score = vec_score + graph_score  # simple additive ranking

            combined.append({
                "vector_result": res,
                "graph_relations": related_entities,
                "vector_score": vec_score,
                "graph_score": graph_score,
                "total_score": total_score
            })

        # 5. Sort by total_score descending and return top_k_final
        combined_sorted = sorted(combined, key=lambda x: x["total_score"], reverse=True)
        return combined_sorted[: self.top_k_final]
