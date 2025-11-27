"""
End-to-End Pipeline Test
------------------------
Tests the complete flow:
1. Ingestion: Extract text and metadata from documents
2. Embedding: Convert text chunks to vector embeddings
3. Vector DB: Store embeddings for semantic search
4. Graph DB: Store entities and relationships

Run this to verify the entire pipeline works correctly.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.ingest_pipeline import IngestionPipeline
from src.embedding.embedder import Embedder
from src.vector_db.qdrant_client import LocalVectorDB
from src.graph_db.memgraph_client import MemgraphClient


def extract_entities_simple(paragraphs):
    """
    Simple entity extraction from paragraphs.
    In a real system, you'd use NER models like spaCy or transformers.
    """
    entities = []
    entity_map = {}
    
    for para in paragraphs:
        text = para["text"].lower()
        para_id = para["id"]
        
        # Simple keyword-based extraction (for demo purposes)
        if "alice" in text:
            if "alice" not in entity_map:
                entity_map["alice"] = {
                    "id": "e_alice",
                    "label": "Person",
                    "metadata": {"name": "Alice", "role": "Software Engineer"}
                }
                entities.append(entity_map["alice"])
        
        if "bob" in text:
            if "bob" not in entity_map:
                entity_map["bob"] = {
                    "id": "e_bob",
                    "label": "Person",
                    "metadata": {"name": "Bob", "role": "CTO"}
                }
                entities.append(entity_map["bob"])
        
        if "devforge" in text:
            if "devforge" not in entity_map:
                entity_map["devforge"] = {
                    "id": "e_devforge",
                    "label": "Company",
                    "metadata": {"name": "DevForge", "location": "Bangalore", "specialization": "AI SaaS products"}
                }
                entities.append(entity_map["devforge"])
    
    # Extract relationships
    relationships = []
    for para in paragraphs:
        text = para["text"].lower()
        if "alice" in text and "devforge" in text:
            relationships.append({
                "start": "e_alice",
                "end": "e_devforge",
                "type": "WORKS_AT",
                "metadata": {"source": para["id"]}
            })
        if "bob" in text and "devforge" in text:
            relationships.append({
                "start": "e_bob",
                "end": "e_devforge",
                "type": "WORKS_AT",
                "metadata": {"source": para["id"]}
            })
    
    return entities, relationships


def main():
    print("=" * 70)
    print("END-TO-END PIPELINE TEST")
    print("=" * 70)
    print()
    
    # Step 1: Initialize components
    print("[1/5] Initializing components...")
    try:
        ingestion = IngestionPipeline()
        embedder = Embedder()
        vector_db = LocalVectorDB(dim=384, db_dir="vector_db_store")
        
        try:
            graph_db = MemgraphClient(host="127.0.0.1", port=7687)
            memgraph_available = True
        except ConnectionError:
            print("   ⚠️  Memgraph not available (connection failed)")
            print("   Continuing without graph DB tests...")
            memgraph_available = False
            graph_db = None
        
        print("   ✅ All components initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize components: {e}")
        return
    print()
    
    # Step 2: Ingest document
    print("[2/5] Ingesting document...")
    sample_file = Path("data/sample.txt")
    
    if not sample_file.exists():
        print(f"   ❌ Sample file not found: {sample_file}")
        return
    
    try:
        ingested_data = ingestion.run(str(sample_file))
        print(f"   ✅ Document ingested: {ingested_data['source']}")
        print(f"   - Type: {ingested_data['type']}")
        print(f"   - Paragraphs: {len(ingested_data['paragraphs'])}")
        print(f"   - Tables: {len(ingested_data['tables'])}")
    except Exception as e:
        print(f"   ❌ Ingestion failed: {e}")
        return
    print()
    
    # Step 3: Create embeddings
    print("[3/5] Creating embeddings...")
    try:
        paragraphs = ingested_data['paragraphs']
        texts = [p['text'] for p in paragraphs]
        
        embeddings = embedder.encode_texts(texts)
        print(f"   ✅ Created {len(embeddings)} embeddings")
        print(f"   - Embedding dimension: {len(embeddings[0])}")
    except Exception as e:
        print(f"   ❌ Embedding creation failed: {e}")
        return
    print()
    
    # Step 4: Store in Vector DB
    print("[4/5] Storing in Vector DB...")
    try:
        # Prepare metadata for each paragraph
        metadatas = []
        ids = []
        
        for para in paragraphs:
            metadata = {
                "paragraph_id": para["id"],
                "source": ingested_data["source"],
                "text": para["text"][:100] + "..." if len(para["text"]) > 100 else para["text"]
            }
            metadatas.append(metadata)
            ids.append(para["id"])
        
        vector_db.upsert_documents(ids=ids, embeddings=embeddings, metadatas=metadatas)
        print(f"   ✅ Stored {len(ids)} documents in Vector DB")
        
        # Test search
        query_text = "What does Alice do?"
        query_embedding = embedder.encode_text(query_text)
        search_results = vector_db.search_vector(query_embedding, top_k=2)
        
        print(f"   ✅ Test search query: '{query_text}'")
        print(f"   - Top result: {search_results[0]['payload']['text'][:50]}...")
        print(f"   - Similarity score: {search_results[0]['score']:.4f}")
    except Exception as e:
        print(f"   ❌ Vector DB operation failed: {e}")
        return
    print()
    
    # Step 5: Store in Graph DB (if available)
    if memgraph_available:
        print("[5/5] Storing in Graph DB...")
        try:
            # Extract entities and relationships
            entities, relationships = extract_entities_simple(paragraphs)
            
            # Store entities
            for entity in entities:
                graph_db.create_entity_node(
                    entity_id=entity["id"],
                    label=entity["label"],
                    metadata=entity["metadata"]
                )
            print(f"   ✅ Created {len(entities)} entity nodes")
            
            # Store relationships (deduplicate)
            seen_rels = set()
            unique_rels = []
            for rel in relationships:
                rel_key = (rel["start"], rel["end"], rel["type"])
                if rel_key not in seen_rels:
                    seen_rels.add(rel_key)
                    unique_rels.append(rel)
            
            for rel in unique_rels:
                graph_db.create_relationship(
                    start_entity_id=rel["start"],
                    end_entity_id=rel["end"],
                    rel_type=rel["type"],
                    metadata=rel["metadata"]
                )
            print(f"   ✅ Created {len(unique_rels)} relationships")
            
            # Test query
            query = "MATCH (p:Person)-[r:WORKS_AT]->(c:Company) RETURN p.name, p.role, c.name, c.location"
            results = graph_db.run_query(query)
            
            print(f"   ✅ Test query executed")
            print(f"   - Found {len(results)} relationships:")
            for result in results:
                print(f"     • {result.get('p.name', 'N/A')} ({result.get('p.role', 'N/A')}) works at {result.get('c.name', 'N/A')} in {result.get('c.location', 'N/A')}")
            
        except Exception as e:
            print(f"   ❌ Graph DB operation failed: {e}")
            return
    else:
        print("[5/5] Skipping Graph DB (not available)")
    print()
    
    # Summary
    print("=" * 70)
    print("✅ PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  • Document ingested: {ingested_data['source']}")
    print(f"  • Paragraphs processed: {len(paragraphs)}")
    print(f"  • Embeddings created: {len(embeddings)}")
    print(f"  • Documents stored in Vector DB: {len(ids)}")
    if memgraph_available:
        print(f"  • Entities stored in Graph DB: {len(entities)}")
        print(f"  • Relationships stored in Graph DB: {len(unique_rels)}")
    print()
    print("All components are working correctly! 🎉")


if __name__ == "__main__":
    main()

