"""
End-to-End Test: Upload → Store → Search
Tests that uploaded documents are properly stored and searchable.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.crud.crud_operations import CRUDOperations
from src.hybrid_query.hybrid_retriever import HybridRetriever
from src.utils.config import DATA_DIR


def main():
    print("=" * 70)
    print("END-TO-END SEARCH TEST")
    print("=" * 70)
    print()
    
    # Initialize
    print("[1/3] Initializing services...")
    crud = CRUDOperations()
    retriever = HybridRetriever(top_k_vectors=5, top_k_final=3)
    print("   ✅ Services initialized")
    print()
    
    # Test with sample.txt
    print("[2/3] Processing sample.txt...")
    sample_file = DATA_DIR / "sample.txt"
    
    if not sample_file.exists():
        print(f"   ❌ Sample file not found: {sample_file}")
        return
    
    try:
        doc_info = crud.add_document(str(sample_file))
        print(f"   ✅ Document processed: {doc_info['metadata']['filename']}")
        print(f"   - Paragraphs: {len(doc_info['paragraphs'])}")
        print(f"   - Entities extracted: {len(doc_info.get('entities', []))}")
        print(f"   - Relationships: {len(doc_info.get('relationships', []))}")
        
        # Check if entities have entity_ids
        paras_with_entities = [p for p in doc_info['paragraphs'] if p.get('entity_ids')]
        print(f"   - Paragraphs with entity_ids: {len(paras_with_entities)}")
    except Exception as e:
        print(f"   ❌ Error processing document: {e}")
        import traceback
        traceback.print_exc()
        return
    print()
    
    # Test search
    print("[3/3] Testing hybrid search...")
    test_queries = [
        "What does Alice do?",
        "Where is DevForge located?",
        "Who works at DevForge?"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        try:
            results = retriever.retrieve(query)
            
            if not results:
                print("   ⚠️  No results found")
            else:
                print(f"   ✅ Found {len(results)} results")
                for idx, res in enumerate(results[:2], 1):  # Show top 2
                    doc = res["vector_result"]
                    text = doc.get("text", "")[:80]
                    vec_score = res.get("vector_score", 0)
                    graph_score = res.get("graph_score", 0)
                    total_score = res.get("total_score", 0)
                    
                    print(f"      {idx}. Score: {total_score:.3f} (vec: {vec_score:.3f}, graph: {graph_score})")
                    print(f"         Text: {text}...")
                    if graph_score > 0:
                        print(f"         ✅ Graph relationships found!")
        except Exception as e:
            print(f"   ❌ Search error: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 70)
    print("✅ END-TO-END TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

