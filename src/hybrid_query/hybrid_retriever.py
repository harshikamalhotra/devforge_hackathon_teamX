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

        # 4. Combine & score with improved ranking
        combined: List[Dict[str, Any]] = []
        query_words = set(query_text.lower().split())  # Extract query keywords
        
        for res in vector_results:
            vec_score = res.get("score", 0.0)
            metadata = res.get("metadata", {})
            payload = res.get("payload", {})
            text = res.get("text", "").lower()
            
            # Quality filter: Skip very short or very low-scoring results
            if len(text.strip()) < 30 or vec_score < 0.2:
                continue
            
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
            
            # Query keyword boost: Boost score if query keywords appear in text
            keyword_boost = 0.0
            if query_words:
                matching_keywords = sum(1 for word in query_words if word in text)
                keyword_boost = (matching_keywords / len(query_words)) * 0.2  # Max 0.2 boost
            
            # Improved scoring: Weighted combination with keyword boost
            # Vector similarity is most important (0.7), graph adds context (0.2), keywords add precision (0.1)
            weighted_vec = vec_score * 0.7
            weighted_graph = min(graph_score * 0.15, 0.2)  # Cap graph contribution
            weighted_keywords = keyword_boost * 0.1
            
            total_score = weighted_vec + weighted_graph + weighted_keywords

            combined.append({
                "vector_result": res,
                "graph_relations": related_entities,
                "vector_score": vec_score,
                "graph_score": graph_score,
                "keyword_boost": keyword_boost,
                "total_score": total_score
            })

        # 5. Deduplicate results - aggressive deduplication to ensure no duplicates
        seen_ids = set()  # For result IDs
        seen_paragraphs = set()  # For doc_id + paragraph_id combinations
        seen_texts = set()  # For normalized text content
        seen_docs = set()  # Track documents to ensure diversity
        deduplicated = []
        
        for res in combined:
            doc = res["vector_result"]
            
            # Get all identifiers
            result_id = doc.get("id", "")
            doc_id = doc.get("doc_id", "")
            paragraph_id = doc.get("paragraph_id", "")
            text = doc.get("text", "").strip()
            
            # Normalize text for comparison (remove extra whitespace, lowercase for comparison)
            text_normalized = " ".join(text.split()).lower() if text else ""
            
            # Check 1: Result ID (most reliable if present)
            if result_id:
                if result_id in seen_ids:
                    continue
                seen_ids.add(result_id)
            
            # Check 2: Document + Paragraph combination (very reliable)
            if doc_id and paragraph_id:
                para_key = f"{doc_id}||{paragraph_id}"
                if para_key in seen_paragraphs:
                    continue
                seen_paragraphs.add(para_key)
            
            # Check 3: Exact text match (catches duplicates even if IDs differ)
            # Use a hash of first 100 chars + length for faster comparison
            if text_normalized and len(text_normalized) > 5:
                # Create a text signature: first 100 chars + total length
                text_sig = f"{text_normalized[:100]}_{len(text_normalized)}"
                if text_sig in seen_texts:
                    # Also check full text match for certainty
                    if text_normalized in seen_texts:
                        continue
                seen_texts.add(text_normalized)
                seen_texts.add(text_sig)
            
            # Track document for diversity (optional - can be removed if you want same doc multiple times)
            # This ensures we get results from different documents when possible
            if doc_id:
                seen_docs.add(doc_id)
            
            deduplicated.append(res)
        
        # 6. Sort by total_score descending
        combined_sorted = sorted(deduplicated, key=lambda x: x["total_score"], reverse=True)
        
        # 7. Apply diversity filter: Prefer results from different documents
        # This ensures we get diverse perspectives when possible
        final_results = []
        seen_doc_ids = set()
        
        for res in combined_sorted:
            if len(final_results) >= self.top_k_final:
                break
            
            doc = res["vector_result"]
            doc_id = doc.get("doc_id", "")
            
            # If we already have a result from this document and we have multiple results,
            # skip unless this result has a significantly higher score
            if doc_id in seen_doc_ids and len(final_results) > 0:
                # Only add if score is much better (at least 0.1 higher) than existing results
                if final_results and res["total_score"] > final_results[-1]["total_score"] + 0.1:
                    final_results.append(res)
                    seen_doc_ids.add(doc_id)
            else:
                final_results.append(res)
                if doc_id:
                    seen_doc_ids.add(doc_id)
        
        return final_results[: self.top_k_final]
