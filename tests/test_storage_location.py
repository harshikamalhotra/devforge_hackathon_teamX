"""Check where vector DB data is stored"""

from pathlib import Path
import os

def main():
    store_dir = Path("vector_db_store")
    
    print("Vector DB Storage Location:")
    print("=" * 70)
    print(f"Directory: {store_dir.absolute()}")
    print(f"Exists: {store_dir.exists()}")
    print()
    
    if store_dir.exists():
        vec_file = store_dir / "vectors.npy"
        meta_file = store_dir / "metadata.json"
        
        if vec_file.exists():
            size_kb = os.path.getsize(vec_file) / 1024
            print(f"✅ vectors.npy: {size_kb:.2f} KB")
        else:
            print("❌ vectors.npy: Not found")
        
        if meta_file.exists():
            size_kb = os.path.getsize(meta_file) / 1024
            print(f"✅ metadata.json: {size_kb:.2f} KB")
        else:
            print("❌ metadata.json: Not found")
    
    print()
    print("Graph DB Storage:")
    print("=" * 70)
    print("Memgraph stores data in Docker container volumes")
    print("Check with: docker exec memgraph cypher-shell 'MATCH (n) RETURN count(n)'")

if __name__ == "__main__":
    main()

