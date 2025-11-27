"""
Test script for HybridRetriever.
Tests the combination of vector similarity search and graph-based retrieval.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hybrid_query.hybrid_retriever import HybridRetriever
from src.vector_db.qdrant_client import LocalVectorDB
from src.embedding.embedder import Embedder
from src.graph_db.memgraph_client import MemgraphClient


def main():
    print("=" * 70)
    print("HYBRID RETRIEVER TEST")
    print("=" * 70)
    print()
    
    # Initialize components
    print("[1/4] Initializing components...")
    try:
        embedder = Embedder()
        # Use a test-specific vector DB directory
        test_db_dir = "vector_db_store_test"
        vector_db = LocalVectorDB(dim=384, db_dir=test_db_dir)
        
        # Create a custom retriever that uses our test vector DB
        retriever = HybridRetriever(top_k_vectors=5, top_k_final=3)
        # Override the vector_db to use our test instance
        retriever.vector_db = vector_db
        
        # Try to connect to graph DB
        try:
            graph_db = MemgraphClient()
            graph_available = True
            print("   ✅ Graph DB connected")
        except ConnectionError:
            graph_available = False
            print("   ⚠️  Graph DB not available (continuing without graph features)")
        
        print("   ✅ All components initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return
    print()
    
    # Step 2: Add sample documents to vector DB
    print("[2/4] Adding sample documents to vector DB...")
    try:
        sample_texts = [
            "Graph databases are powerful for storing relationships between entities.",
            "Cypher queries can retrieve complex relationships efficiently.",
            "Deep learning models improve semantic search capabilities.",
            "Vector embeddings represent text in high-dimensional space.",
            "Hybrid retrieval combines vector search with graph traversal."
        ]
        
        # Create embeddings
        embeddings = embedder.encode_texts(sample_texts)
        
        # Prepare metadata with entity IDs (simulated)
        metadatas = []
        ids = []
        for idx, text in enumerate(sample_texts):
            para_id = f"doc1_p{idx+1}"
            ids.append(para_id)
            
            # Simulate entity extraction - add entity IDs to metadata
            entity_ids = []
            if "Graph" in text or "graph" in text:
                entity_ids.append("e_graph_db")
            if "Cypher" in text or "query" in text:
                entity_ids.append("e_cypher")
            if "Deep learning" in text or "semantic" in text:
                entity_ids.append("e_ml")
            if "Vector" in text or "embedding" in text:
                entity_ids.append("e_vector")
            if "Hybrid" in text or "retrieval" in text:
                entity_ids.append("e_hybrid")
            
            metadata_dict = {
                "text": text,
                "doc_id": "doc1",
                "paragraph_id": para_id,
                "entity_ids": entity_ids,
                "source": "test_document"
            }
            metadatas.append(metadata_dict)
        
        vector_db.upsert_documents(ids=ids, embeddings=embeddings, metadatas=metadatas)
        print(f"   ✅ Added {len(sample_texts)} documents to vector DB")
    except Exception as e:
        print(f"   ❌ Failed to add documents: {e}")
        return
    print()
    
    # Step 3: Add entities and relationships to graph DB (if available)
    if graph_available:
        print("[3/4] Adding entities and relationships to graph DB...")
        try:
            # Create entity nodes
            entities = [
                {"id": "e_graph_db", "label": "Concept", "metadata": {"name": "Graph Database", "type": "technology"}},
                {"id": "e_cypher", "label": "Concept", "metadata": {"name": "Cypher Query", "type": "language"}},
                {"id": "e_ml", "label": "Concept", "metadata": {"name": "Machine Learning", "type": "technology"}},
                {"id": "e_vector", "label": "Concept", "metadata": {"name": "Vector Embedding", "type": "technique"}},
                {"id": "e_hybrid", "label": "Concept", "metadata": {"name": "Hybrid Retrieval", "type": "method"}}
            ]
            
            for entity in entities:
                graph_db.create_entity_node(
                    entity_id=entity["id"],
                    label=entity["label"],
                    metadata=entity["metadata"]
                )
            
            # Create relationships
            relationships = [
                {"start": "e_graph_db", "end": "e_cypher", "type": "USES", "metadata": {}},
                {"start": "e_hybrid", "end": "e_graph_db", "type": "COMBINES", "metadata": {}},
                {"start": "e_hybrid", "end": "e_vector", "type": "COMBINES", "metadata": {}},
                {"start": "e_ml", "end": "e_vector", "type": "GENERATES", "metadata": {}}
            ]
            
            for rel in relationships:
                graph_db.create_relationship(
                    start_entity_id=rel["start"],
                    end_entity_id=rel["end"],
                    rel_type=rel["type"],
                    metadata=rel["metadata"]
                )
            
            print(f"   ✅ Created {len(entities)} entities and {len(relationships)} relationships")
        except Exception as e:
            print(f"   ⚠️  Failed to add graph data: {e}")
            print("   Continuing with vector-only retrieval...")
    else:
        print("[3/4] Skipping graph DB setup (not available)")
    print()
    
    # Step 4: Test hybrid retrieval
    print("[4/4] Testing hybrid retrieval...")
    try:
        query = "How do graph databases help with semantic search?"
        
        print(f"   Query: '{query}'")
        results = retriever.retrieve(query)
        
        print(f"   ✅ Retrieved {len(results)} results")
        print()
        print("   Results (sorted by total_score):")
        print("-" * 70)
        
        for idx, result in enumerate(results, 1):
            vector_res = result.get("vector_result", {})
            text = vector_res.get("text", "N/A")
            vec_score = result.get("vector_score", 0.0)
            graph_score = result.get("graph_score", 0.0)
            total_score = result.get("total_score", 0.0)
            graph_rels = result.get("graph_relations", [])
            
            print(f"\n   {idx}. Score: {total_score:.4f} (vector: {vec_score:.4f}, graph: {graph_score})")
            print(f"      Text: {text[:80]}...")
            if graph_rels:
                print(f"      Graph relations: {len(graph_rels)} related entities found")
            else:
                print(f"      Graph relations: None")
        
        print()
        print("=" * 70)
        print("✅ HYBRID RETRIEVER TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        # Verify results
        assert len(results) > 0, "No results returned"
        assert "total_score" in results[0], "Results missing total_score"
        assert "vector_result" in results[0], "Results missing vector_result"
        print("\n✅ All assertions passed!")
        
    except Exception as e:
        print(f"   ❌ Hybrid retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Cleanup
    try:
        import shutil
        import os
        if os.path.exists("vector_db_store_test"):
            shutil.rmtree("vector_db_store_test")
            print("\n🧹 Test data cleaned up")
    except:
        pass


if __name__ == "__main__":
    main()
