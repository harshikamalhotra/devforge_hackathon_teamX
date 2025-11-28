"""
Test to verify that the frontend uses the configured vector DB backend.
This ensures that when VECTOR_DB_TYPE is set to "chromadb", the frontend uses ChromaDB.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.crud.crud_operations import CRUDOperations
from src.utils.config import VECTOR_DB_TYPE
from src.vector_db import LocalVectorDB, ChromaDBClient


def test_vector_db_integration():
    """Test that CRUDOperations uses the configured vector DB."""
    print("=" * 70)
    print("FRONTEND VECTOR DB INTEGRATION TEST")
    print("=" * 70)
    print()
    
    print(f"Current VECTOR_DB_TYPE configuration: {VECTOR_DB_TYPE}")
    print()
    
    # Initialize CRUDOperations (same as frontend does)
    print("Initializing CRUDOperations (as frontend does)...")
    crud = CRUDOperations()
    
    # Check which type of vector DB was created
    vector_db = crud.vector_db
    db_type = type(vector_db).__name__
    
    print(f"✅ CRUDOperations initialized")
    print(f"   Vector DB type: {db_type}")
    print()
    
    # Verify it matches the configuration
    if VECTOR_DB_TYPE == "chromadb":
        assert isinstance(vector_db, ChromaDBClient), \
            f"Expected ChromaDBClient but got {db_type}"
        print("✅ Correctly using ChromaDB backend")
        print(f"   Database directory: {vector_db.db_dir}")
        print(f"   Collection name: {vector_db.collection_name}")
    else:
        assert isinstance(vector_db, LocalVectorDB), \
            f"Expected LocalVectorDB but got {db_type}"
        print("✅ Correctly using LocalVectorDB backend")
        print(f"   Database directory: {vector_db.db_dir}")
    
    print()
    print("=" * 70)
    print("INTEGRATION TEST PASSED")
    print("=" * 70)
    print()
    print("💡 To switch to ChromaDB, edit src/utils/config.py:")
    print('   Change: VECTOR_DB_TYPE = "local"')
    print('   To:     VECTOR_DB_TYPE = "chromadb"')
    print()
    print("💡 The frontend will automatically use the configured backend!")
    print("   No code changes needed in the frontend.")
    
    return True


if __name__ == "__main__":
    try:
        test_vector_db_integration()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

