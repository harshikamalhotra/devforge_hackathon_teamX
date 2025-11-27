"""Test if entities are stored in graph DB"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.crud.crud_operations import CRUDOperations
from src.graph_db.memgraph_client import MemgraphClient

def main():
    print("Checking graph DB entities...")
    
    try:
        graph = MemgraphClient()
        results = graph.run_query("MATCH (n) RETURN n.id as id, labels(n) as labels LIMIT 10")
        print(f"Entities in graph DB: {len(results)}")
        for r in results[:10]:
            print(f"  - ID: {r.get('id')}, Labels: {r.get('labels')}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nProcessing sample.txt to add entities...")
    crud = CRUDOperations()
    doc_info = crud.add_document("data/sample.txt")
    
    print(f"Extracted entities: {len(doc_info.get('entities', []))}")
    for entity in doc_info.get('entities', [])[:5]:
        print(f"  - {entity['id']}: {entity['label']}")
    
    print("\nChecking graph DB again...")
    try:
        graph = MemgraphClient()
        results = graph.run_query("MATCH (n) RETURN n.id as id, labels(n) as labels LIMIT 10")
        print(f"Entities in graph DB: {len(results)}")
        for r in results[:10]:
            print(f"  - ID: {r.get('id')}, Labels: {r.get('labels')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

