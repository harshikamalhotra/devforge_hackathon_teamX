"""
Test script to upload PDF, ingest it, and test hybrid search
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.crud.crud_operations import CRUDOperations
from src.hybrid_query.hybrid_retriever import HybridRetriever
from src.utils.config import DATA_DIR

def main():
    print("=" * 70)
    print("PDF UPLOAD AND HYBRID SEARCH TEST")
    print("=" * 70)
    
    # Initialize components
    print("\n[1/4] Initializing components...")
    crud = CRUDOperations()
    retriever = HybridRetriever(top_k_vectors=10, top_k_final=5)
    retriever.vector_db = crud.vector_db
    print("   ✅ Components initialized")
    
    # Find PDF file
    print("\n[2/4] Finding PDF file...")
    pdf_file = DATA_DIR / "Devfolio Hackathon Problem Statement Vector+Graph Native Database— Test Cases (1).pdf"
    
    if not pdf_file.exists():
        print(f"   ❌ PDF file not found: {pdf_file}")
        return False
    
    print(f"   ✅ Found PDF: {pdf_file.name}")
    
    # Ingest PDF
    print("\n[3/4] Ingesting PDF document...")
    try:
        doc_info = crud.add_document(str(pdf_file))
        print(f"   ✅ Document ingested successfully!")
        print(f"   📊 Paragraphs: {len(doc_info.get('paragraphs', []))}")
        print(f"   📊 Entities: {len(doc_info.get('entities', []))}")
        print(f"   📊 Relationships: {len(doc_info.get('relationships', []))}")
        
        # Show sample paragraph
        if doc_info.get('paragraphs'):
            sample_para = doc_info['paragraphs'][0]
            print(f"\n   Sample paragraph text (first 200 chars):")
            print(f"   {sample_para.get('text', '')[:200]}...")
            
    except Exception as e:
        print(f"   ❌ Error ingesting PDF: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Update retriever to use latest vector DB
    retriever.vector_db = crud.vector_db
    
    # Test hybrid search with various queries
    print("\n[4/4] Testing hybrid search...")
    test_queries = [
        "redis caching",
        "hybrid search",
        "graph traversal",
        "vector similarity",
        "test cases"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        print("-" * 70)
        
        try:
            # Test with default weights (0.6, 0.4)
            results = retriever.retrieve(query, vector_weight=0.6, graph_weight=0.4)
            
            if not results:
                print(f"   ⚠️  No results found")
                continue
            
            print(f"   ✅ Found {len(results)} results")
            
            for idx, res in enumerate(results, 1):
                doc = res.get("vector_result", {})
                text = doc.get("text", "")
                vector_score = res.get("vector_score", 0.0)
                graph_score = res.get("graph_score", 0.0)
                final_score = res.get("final_score", 0.0)
                hop = res.get("hop")
                
                # Check if text is truncated
                text_length = len(text)
                is_truncated = text_length < 50  # Very short might indicate truncation
                
                print(f"\n   Result {idx}:")
                print(f"      Vector Score: {vector_score:.6f}")
                print(f"      Graph Score: {graph_score:.6f} (hop: {hop})")
                print(f"      Final Score: {final_score:.6f}")
                print(f"      Text Length: {text_length} chars")
                
                if is_truncated:
                    print(f"      ⚠️  WARNING: Text might be truncated!")
                
                # Show text preview
                print(f"      Text Preview: {text[:150]}...")
                
                # Verify formula
                v_weight = res.get("vector_weight", 0.6)
                g_weight = res.get("graph_weight", 0.4)
                expected_final = v_weight * vector_score + g_weight * graph_score
                
                if abs(final_score - expected_final) > 0.0001:
                    print(f"      ❌ ERROR: Final score mismatch!")
                    print(f"         Expected: {expected_final:.6f}, Got: {final_score:.6f}")
                else:
                    print(f"      ✅ Formula verified: {v_weight:.2f}×{vector_score:.6f} + {g_weight:.2f}×{graph_score:.6f} = {final_score:.6f}")
                
        except Exception as e:
            print(f"   ❌ Error during search: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETED")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


