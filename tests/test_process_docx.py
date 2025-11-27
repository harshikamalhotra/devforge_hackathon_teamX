"""Test processing .docx files"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.crud.crud_operations import CRUDOperations

def main():
    crud = CRUDOperations()
    
    # Test processing blog 1.docx
    docx_file = Path("data/blog 1.docx")
    if docx_file.exists():
        print(f"Processing: {docx_file.name}")
        try:
            doc = crud.add_document(str(docx_file))
            print(f"✅ Processed: {doc['metadata']['filename']}")
            print(f"   Paragraphs: {len(doc['paragraphs'])}")
            print(f"   Entities: {len(doc.get('entities', []))}")
            print(f"   Relationships: {len(doc.get('relationships', []))}")
            print(f"\n   First paragraph preview: {doc['paragraphs'][0]['text'][:100]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"File not found: {docx_file}")

if __name__ == "__main__":
    main()

