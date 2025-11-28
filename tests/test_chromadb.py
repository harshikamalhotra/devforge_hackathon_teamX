"""
Test script for ChromaDB implementation.
Tests all functionality to ensure it matches LocalVectorDB interface.
"""

import sys
import os
import shutil
from pathlib import Path
import uuid
import random

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_db.chromadb_client import ChromaDBClient
from src.embedding.embedder import Embedder


def cleanup_test_db(db_dir: str, db_instance=None):
    """Remove test database directory if it exists."""
    # Close the database instance first to release file locks
    if db_instance is not None:
        try:
            db_instance.close()
        except Exception:
            pass
    
    # Try multiple times on Windows (files may take time to release)
    import time
    max_attempts = 3
    for attempt in range(max_attempts):
        if os.path.exists(db_dir):
            try:
                shutil.rmtree(db_dir)
                print(f"   🧹 Cleaned up test database: {db_dir}")
                return
            except Exception as e:
                if attempt < max_attempts - 1:
                    time.sleep(0.5)  # Wait a bit before retrying
                else:
                    print(f"   ⚠️  Could not clean up {db_dir}: {e} (this is OK on Windows)")


def test_basic_operations():
    """Test basic upsert and search operations."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Operations (upsert, search)")
    print("=" * 70)
    
    test_db_dir = "vector_db_store_chromadb_test"
    cleanup_test_db(test_db_dir)
    
    try:
        # Initialize ChromaDB
        db = ChromaDBClient(dim=384, db_dir=test_db_dir)
        print("   ✅ ChromaDB initialized")
        
        # Create test data
        ids = [str(uuid.uuid4()) for _ in range(5)]
        embeddings = [[random.random() for _ in range(384)] for _ in range(5)]
        metas = [{"text": f"Document {i}", "index": i} for i in range(5)]
        
        # Test upsert
        print("   📝 Inserting documents...")
        db.upsert_documents(ids, embeddings, metas)
        print(f"   ✅ Inserted {len(ids)} documents")
        
        # Test search - query should match first document best
        print("   🔍 Searching with first document as query...")
        query = embeddings[0]
        results = db.search_vector(query, top_k=3)
        
        print(f"   ✅ Found {len(results)} results")
        assert len(results) > 0, "Should return at least one result"
        assert results[0]["id"] == ids[0], "First result should be the query document itself"
        print(f"   ✅ Top result is correct: {results[0]['id']}")
        print(f"      Score: {results[0]['score']:.4f}")
        
        # Test similarity_search alias
        print("   🔍 Testing similarity_search method...")
        sim_results = db.similarity_search(query, top_k=2)
        assert len(sim_results) > 0, "similarity_search should return results"
        assert "text" in sim_results[0], "similarity_search should include text field"
        print(f"   ✅ similarity_search works correctly")
        
        cleanup_test_db(test_db_dir, db)
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_db(test_db_dir, db if 'db' in locals() else None)
        return False


def test_crud_operations():
    """Test CRUD operations (add_document, get_document, delete_document)."""
    print("\n" + "=" * 70)
    print("TEST 2: CRUD Operations")
    print("=" * 70)
    
    test_db_dir = "vector_db_store_chromadb_test_crud"
    cleanup_test_db(test_db_dir)
    
    try:
        embedder = Embedder()
        db = ChromaDBClient(dim=384, db_dir=test_db_dir)
        print("   ✅ ChromaDB initialized")
        
        # Create a document with paragraphs
        doc_id = "test_doc_1"
        paragraphs = [
            {"id": "p1", "text": "This is the first paragraph about machine learning."},
            {"id": "p2", "text": "This is the second paragraph about neural networks."},
            {"id": "p3", "text": "This is the third paragraph about deep learning."}
        ]
        
        # Add embeddings to paragraphs
        for para in paragraphs:
            para["embedding"] = embedder.encode_text(para["text"])
            para["entity_ids"] = ["e_ml", "e_ai"]  # Simulated entity IDs
        
        content = {
            "source": doc_id,
            "type": "test",
            "metadata": {"author": "test_user", "version": 1},
            "paragraphs": paragraphs
        }
        
        # Test add_document
        print("   📝 Adding document with paragraphs...")
        db.add_document(doc_id, content)
        print(f"   ✅ Added document: {doc_id}")
        
        # Test get_document
        print("   📖 Retrieving document...")
        retrieved = db.get_document(doc_id)
        assert retrieved != {}, "Document should be retrieved"
        assert "paragraphs" in retrieved, "Retrieved document should have paragraphs"
        assert len(retrieved["paragraphs"]) == 3, "Should retrieve all 3 paragraphs"
        print(f"   ✅ Retrieved document with {len(retrieved['paragraphs'])} paragraphs")
        
        # Verify paragraph content
        for i, para in enumerate(retrieved["paragraphs"]):
            assert para["id"] == paragraphs[i]["id"], f"Paragraph {i} ID should match"
            assert para["text"] == paragraphs[i]["text"], f"Paragraph {i} text should match"
        print("   ✅ Paragraph content matches")
        
        # Test delete_document
        print("   🗑️  Deleting document...")
        db.delete_document(doc_id)
        retrieved_after_delete = db.get_document(doc_id)
        assert retrieved_after_delete == {}, "Document should be deleted"
        print("   ✅ Document deleted successfully")
        
        cleanup_test_db(test_db_dir, db)
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_db(test_db_dir, db if 'db' in locals() else None)
        return False


def test_delete_operations():
    """Test delete operations."""
    print("\n" + "=" * 70)
    print("TEST 3: Delete Operations")
    print("=" * 70)
    
    test_db_dir = "vector_db_store_chromadb_test_delete"
    cleanup_test_db(test_db_dir)
    
    try:
        db = ChromaDBClient(dim=384, db_dir=test_db_dir)
        print("   ✅ ChromaDB initialized")
        
        # Insert multiple documents
        ids = [f"doc_{i}" for i in range(5)]
        embeddings = [[random.random() for _ in range(384)] for _ in range(5)]
        metas = [{"text": f"Document {i}"} for i in range(5)]
        
        db.upsert_documents(ids, embeddings, metas)
        print(f"   ✅ Inserted {len(ids)} documents")
        
        # Test delete specific IDs
        print("   🗑️  Deleting specific documents...")
        ids_to_delete = [ids[0], ids[2]]
        db.delete(ids_to_delete)
        
        # Verify deletion
        query = embeddings[1]  # Use embedding from non-deleted doc
        results = db.search_vector(query, top_k=10)
        remaining_ids = [r["id"] for r in results]
        
        assert ids[0] not in remaining_ids, "First document should be deleted"
        assert ids[2] not in remaining_ids, "Third document should be deleted"
        assert ids[1] in remaining_ids, "Second document should still exist"
        print(f"   ✅ Specific documents deleted correctly")
        print(f"      Remaining documents: {len(remaining_ids)}")
        
        # Test delete_all
        print("   🗑️  Deleting all documents...")
        db.delete_all()
        results_after = db.search_vector(query, top_k=10)
        assert len(results_after) == 0, "All documents should be deleted"
        print("   ✅ All documents deleted")
        
        cleanup_test_db(test_db_dir, db)
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_db(test_db_dir, db if 'db' in locals() else None)
        return False


def test_metadata_handling():
    """Test metadata serialization and deserialization."""
    print("\n" + "=" * 70)
    print("TEST 4: Metadata Handling")
    print("=" * 70)
    
    test_db_dir = "vector_db_store_chromadb_test_metadata"
    cleanup_test_db(test_db_dir)
    
    try:
        db = ChromaDBClient(dim=384, db_dir=test_db_dir)
        print("   ✅ ChromaDB initialized")
        
        # Test with complex metadata (lists, nested dicts)
        ids = ["doc1", "doc2"]
        embeddings = [[random.random() for _ in range(384)] for _ in range(2)]
        metas = [
            {
                "text": "Document 1",
                "entity_ids": ["e1", "e2", "e3"],
                "metadata": {"author": "user1", "tags": ["tag1", "tag2"]},
                "nested": {"level1": {"level2": "value"}}
            },
            {
                "text": "Document 2",
                "entity_ids": ["e4", "e5"],
                "metadata": {"author": "user2"}
            }
        ]
        
        print("   📝 Inserting documents with complex metadata...")
        db.upsert_documents(ids, embeddings, metas)
        print("   ✅ Documents inserted")
        
        # Search and verify metadata is preserved
        print("   🔍 Searching and verifying metadata...")
        results = db.search_vector(embeddings[0], top_k=2)
        
        assert len(results) > 0, "Should return results"
        result_meta = results[0]["payload"]
        
        # Check that metadata fields are accessible
        assert "text" in result_meta, "Text should be in metadata"
        assert "entity_ids" in result_meta, "entity_ids should be in metadata"
        
        # Verify entity_ids can be parsed back
        entity_ids = result_meta["entity_ids"]
        if isinstance(entity_ids, str):
            import json
            entity_ids = json.loads(entity_ids)
        assert isinstance(entity_ids, list), "entity_ids should be a list"
        print(f"   ✅ Metadata preserved correctly")
        print(f"      Entity IDs: {entity_ids}")
        
        cleanup_test_db(test_db_dir, db)
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_db(test_db_dir, db if 'db' in locals() else None)
        return False


def test_reload():
    """Test reload functionality."""
    print("\n" + "=" * 70)
    print("TEST 5: Reload Functionality")
    print("=" * 70)
    
    test_db_dir = "vector_db_store_chromadb_test_reload"
    cleanup_test_db(test_db_dir)
    
    try:
        # Create first instance and add data
        db1 = ChromaDBClient(dim=384, db_dir=test_db_dir)
        ids = ["doc1", "doc2"]
        embeddings = [[random.random() for _ in range(384)] for _ in range(2)]
        metas = [{"text": f"Document {i}"} for i in range(2)]
        db1.upsert_documents(ids, embeddings, metas)
        print("   ✅ Created first instance and added data")
        
        # Create second instance (should see the data)
        db2 = ChromaDBClient(dim=384, db_dir=test_db_dir)
        results = db2.search_vector(embeddings[0], top_k=2)
        assert len(results) > 0, "Second instance should see existing data"
        print("   ✅ Second instance can see existing data")
        
        # Test reload
        db2.reload()
        results_after_reload = db2.search_vector(embeddings[0], top_k=2)
        assert len(results_after_reload) > 0, "After reload should still see data"
        print("   ✅ Reload works correctly")
        
        # Close both instances
        db1.close()
        db2.close()
        cleanup_test_db(test_db_dir)
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        # Try to close instances if they exist
        try:
            if 'db1' in locals():
                db1.close()
            if 'db2' in locals():
                db2.close()
        except:
            pass
        cleanup_test_db(test_db_dir)
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("CHROMADB IMPLEMENTATION TEST SUITE")
    print("=" * 70)
    
    # Check if chromadb is installed
    try:
        import chromadb
        print(f"   ✅ ChromaDB version: {chromadb.__version__}")
    except ImportError:
        print("   ❌ ChromaDB not installed. Please run: pip install chromadb")
        return
    
    tests = [
        test_basic_operations,
        test_crud_operations,
        test_delete_operations,
        test_metadata_handling,
        test_reload
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"   Tests passed: {passed}/{total}")
    
    if passed == total:
        print("   ✅ All tests passed!")
        return 0
    else:
        print("   ❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

