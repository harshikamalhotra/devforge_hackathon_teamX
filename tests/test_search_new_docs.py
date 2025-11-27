"""Test searching for newly uploaded documents"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.crud.crud_operations import CRUDOperations
from src.hybrid_query.hybrid_retriever import HybridRetriever

def main():
    print("=" * 70)
    print("TESTING SEARCH FOR NEW DOCUMENTS")
    print("=" * 70)
    print()
    
    # Initialize
    crud = CRUDOperations()
    retriever = HybridRetriever(top_k_vectors=10, top_k_final=5)
    
    # Ensure retriever uses the same vector DB instance
    retriever.vector_db = crud.vector_db
    
    print(f"Vector DB has {len(crud.vector_db.ids)} documents stored")
    print()
    
    # Check what documents are stored
    doc_sources = set()
    for doc_id in crud.vector_db.ids:
        payload = crud.vector_db.payloads.get(doc_id, {})
        source = payload.get("source", "unknown")
        doc_sources.add(source)
    
    print("Documents in vector DB:")
    for source in sorted(doc_sources):
        count = sum(1 for doc_id in crud.vector_db.ids 
                   if crud.vector_db.payloads.get(doc_id, {}).get("source") == source)
        print(f"  - {source}: {count} paragraphs")
    print()
    
    # Test search
    queries = [
        "What is the main topic?",
        "What are the key points?",
        "Tell me about the content"
    ]
    
    for query in queries:
        print(f"Query: '{query}'")
        results = retriever.retrieve(query)
        
        if results:
            print(f"  Found {len(results)} results")
            for idx, res in enumerate(results[:3], 1):
                doc = res["vector_result"]
                source = doc.get("metadata", {}).get("source") or doc.get("source", "unknown")
                text = doc.get("text", "")[:80]
                score = res.get("total_score", 0)
                print(f"    {idx}. [{source}] Score: {score:.3f}")
                print(f"       {text}...")
        else:
            print("  No results found")
        print()

if __name__ == "__main__":
    main()

