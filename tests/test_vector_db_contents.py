"""Check what's stored in vector DB"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_db.qdrant_client import LocalVectorDB

def main():
    db = LocalVectorDB()
    
    print(f"Total documents in vector DB: {len(db.ids)}")
    print("\n" + "="*70)
    print("Sample documents:")
    print("="*70)
    
    # Group by doc_id
    doc_groups = {}
    for doc_id in db.ids:
        payload = db.payloads.get(doc_id, {})
        doc_id_key = payload.get("doc_id", "unknown")
        source = payload.get("source", "unknown")
        
        if doc_id_key not in doc_groups:
            doc_groups[doc_id_key] = {
                "source": source,
                "count": 0,
                "sample_text": payload.get("text", "")[:100]
            }
        doc_groups[doc_id_key]["count"] += 1
    
    for doc_id, info in doc_groups.items():
        print(f"\nDocument: {doc_id}")
        print(f"  Source: {info['source']}")
        print(f"  Paragraphs: {info['count']}")
        print(f"  Sample text: {info['sample_text']}...")

if __name__ == "__main__":
    main()

