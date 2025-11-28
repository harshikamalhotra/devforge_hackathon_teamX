"""
Hybrid Retriever combining vector similarity + graph relational retrieval.

Based on GraphRAG / HybridRAG architecture and test case requirements:
1. Use vector embeddings to find top-k semantically relevant chunks/documents.
2. Identify entities from those documents.
3. Use graph DB (Cypher) to expand context via relationships with hop distance.
4. Merge vector + graph results using: final_score = vector_weight * vector_score + graph_weight * graph_score
5. Graph score formula: graph_score = 1 / (1 + hops) where hop=0 → 1.0, hop=1 → 0.5, hop=2 → 0.3333
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from src.embedding.embedder import Embedder
from src.vector_db.qdrant_client import LocalVectorDB  # or your custom client
from src.graph_db.memgraph_client import MemgraphClient     # using gqlalchemy or other client
from src.utils.config import EMBEDDING_MODEL_NAME

class HybridRetriever:
    def __init__(
        self,
        top_k_vectors: int = 10,
        top_k_final: int = 5,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4
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
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        
        # Validate weights sum to 1.0 (or allow flexibility)
        if abs(vector_weight + graph_weight - 1.0) > 0.01:
            # Normalize weights if they don't sum to 1.0
            total = vector_weight + graph_weight
            self.vector_weight = vector_weight / total
            self.graph_weight = graph_weight / total

    def retrieve(
        self, 
        query_text: str,
        vector_weight: Optional[float] = None,
        graph_weight: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval for a query. Returns list of results with combined scores.
        Each result includes:
          - vector result (text chunk, metadata, vector_score)
          - optionally related graph context (entities / relationships)
          - vector_score, graph_score, final_score
          - final_score = vector_weight * vector_score + graph_weight * graph_score
        
        Args:
            query_text: Query string
            vector_weight: Override default vector_weight for this query (default: self.vector_weight)
            graph_weight: Override default graph_weight for this query (default: self.graph_weight)
        """
        # Use provided weights or defaults
        v_weight = vector_weight if vector_weight is not None else self.vector_weight
        g_weight = graph_weight if graph_weight is not None else self.graph_weight
        
        # Normalize weights if needed
        if abs(v_weight + g_weight - 1.0) > 0.01:
            total = v_weight + g_weight
            v_weight = v_weight / total
            g_weight = g_weight / total
        
        # 1. Vector semantic search
        query_emb = self.embedder.encode_text(query_text)
        vector_results = self.vector_db.similarity_search(query_emb, top_k=self.top_k_vectors)

        # If no results, return empty
        if not vector_results:
            return []

        # 2. Collect entity IDs from vector result payloads
        entity_ids = set()
        for res in vector_results:
            metadata = res.get("metadata", {})
            payload = res.get("payload", {})
            eids = metadata.get("entity_ids") or payload.get("entity_ids")
            if isinstance(eids, list) and len(eids) > 0:
                entity_ids.update(eids)

        # 3. Graph-based expansion: compute hop distances from query-relevant entities
        # Find the most relevant entity (from top vector result) as anchor for graph proximity
        anchor_entity_ids = set()
        if vector_results and entity_ids:
            # Use entities from top vector results as anchors
            top_result = vector_results[0]
            top_metadata = top_result.get("metadata", {})
            top_payload = top_result.get("payload", {})
            top_eids = top_metadata.get("entity_ids") or top_payload.get("entity_ids", [])
            if isinstance(top_eids, list) and len(top_eids) > 0:
                anchor_entity_ids.update(top_eids[:3])  # Use top 3 entities as anchors
        
        # Compute graph proximity scores (hop distances) for each result
        graph_proximity_map = {}  # Maps result identifier -> (min_hop, graph_score)
        
        if self.graph_db_available and self.graph_db and anchor_entity_ids:
            # For each result, find minimum hop distance to anchor entities
            for res in vector_results:
                result_id = res.get("id", "")
                metadata = res.get("metadata", {})
                payload = res.get("payload", {})
                result_eids = metadata.get("entity_ids") or payload.get("entity_ids", [])
                
                if not isinstance(result_eids, list) or len(result_eids) == 0:
                    # No entities in this result, graph_score = 0.0
                    graph_proximity_map[result_id] = (float('inf'), 0.0)
                    continue
                
                # Find minimum hop distance from result entities to anchor entities
                min_hop = float('inf')
                
                for result_eid in result_eids:
                    for anchor_eid in anchor_entity_ids:
                        if result_eid == anchor_eid:
                            # Same entity, hop = 0
                            min_hop = 0
                            break
                        else:
                            # Find shortest path between entities
                            hop = self._find_shortest_hop(result_eid, anchor_eid)
                            if hop is not None and hop < min_hop:
                                min_hop = hop
                    
                    if min_hop == 0:
                        break  # Can't get better than 0
                
                # Calculate graph score: graph_score = 1 / (1 + hops)
                if min_hop == float('inf'):
                    graph_score = 0.0
                else:
                    graph_score = 1.0 / (1.0 + min_hop)
                
                graph_proximity_map[result_id] = (min_hop if min_hop != float('inf') else None, graph_score)
        else:
            # No graph DB or no entities, set all graph scores to 0
            for res in vector_results:
                result_id = res.get("id", "")
                graph_proximity_map[result_id] = (None, 0.0)

        # 4. Combine & score using test case formula: final_score = vector_weight * vector_score + graph_weight * graph_score
        combined: List[Dict[str, Any]] = []
        
        for res in vector_results:
            result_id = res.get("id", "")
            vec_score = res.get("score", 0.0)
            metadata = res.get("metadata", {})
            payload = res.get("payload", {})
            text_original = res.get("text", "")
            
            # Quality filter: Skip very short or very low-scoring results
            if len(text_original.strip()) < 10 or vec_score < 0.1:
                continue
            
            # Get graph score from proximity map
            hop_info, graph_score = graph_proximity_map.get(result_id, (None, 0.0))
            
            # Calculate final score using test case formula
            final_score = v_weight * vec_score + g_weight * graph_score
            
            # Get graph relations for display (if available)
            graph_relations = []
            if self.graph_db_available and self.graph_db:
                result_eids = metadata.get("entity_ids") or payload.get("entity_ids", [])
                if isinstance(result_eids, list) and len(result_eids) > 0:
                    # Get relationships for entities in this result
                    for eid in result_eids[:3]:  # Limit to first 3 entities
                        try:
                            cypher = (
                                f"MATCH (n {{id: '{eid}'}})-[r]-(m) "
                                "RETURN n.id AS source_id, type(r) AS rel_type, m.id AS related_id, r.weight AS weight"
                            )
                            rels = self.graph_db.run_query(cypher)
                            graph_relations.extend(rels)
                        except Exception:
                            continue

            combined.append({
                "vector_result": res,
                "graph_relations": graph_relations,
                "vector_score": vec_score,
                "graph_score": graph_score,
                "final_score": final_score,
                "hop": hop_info,
                "vector_weight": v_weight,
                "graph_weight": g_weight
            })

        # 5. Deduplicate results
        seen_ids = set()
        seen_paragraphs = set()
        seen_texts = set()  # Also check for duplicate text content
        deduplicated = []
        
        for res in combined:
            doc = res["vector_result"]
            result_id = doc.get("id", "")
            doc_id = doc.get("doc_id", "")
            paragraph_id = doc.get("paragraph_id", "")
            text = doc.get("text", "").strip()
            
            # Check 1: Result ID
            if result_id and result_id in seen_ids:
                continue
            if result_id:
                seen_ids.add(result_id)
            
            # Check 2: Document + Paragraph combination
            if doc_id and paragraph_id:
                para_key = f"{doc_id}||{paragraph_id}"
                if para_key in seen_paragraphs:
                    continue
                seen_paragraphs.add(para_key)
            
            # Check 3: Exact text match (normalized) - catch duplicates even if IDs differ
            if text and len(text) > 10:  # Only check for substantial text
                text_normalized = " ".join(text.split()).lower()
                if text_normalized in seen_texts:
                    continue
                seen_texts.add(text_normalized)
            
            deduplicated.append(res)
        
        # 6. Sort by final_score descending (as per test case requirements)
        combined_sorted = sorted(deduplicated, key=lambda x: x["final_score"], reverse=True)
        
        # 7. Return top_k_final results
        return combined_sorted[:self.top_k_final]
    
    def _find_shortest_hop(self, start_entity_id: str, end_entity_id: str, max_depth: int = 3) -> Optional[int]:
        """
        Find shortest hop distance between two entities using BFS.
        Returns None if no path found within max_depth.
        """
        if start_entity_id == end_entity_id:
            return 0
        
        if not self.graph_db_available or not self.graph_db:
            return None
        
        try:
            # BFS to find shortest path
            visited = {start_entity_id}
            queue = [(start_entity_id, 0)]
            
            while queue:
                current_id, hop = queue.pop(0)
                
                if hop >= max_depth:
                    continue
                
                # Get neighbors
                cypher = (
                    f"MATCH (n {{id: '{current_id}'}})-[r]-(m) "
                    "RETURN m.id AS neighbor_id"
                )
                neighbors = self.graph_db.run_query(cypher)
                
                for neighbor in neighbors:
                    neighbor_id = neighbor.get("neighbor_id")
                    if neighbor_id == end_entity_id:
                        return hop + 1
                    
                    if neighbor_id and neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, hop + 1))
            
            return None  # No path found
        except Exception:
            return None
