"""
Test script for CRUD operations.
Covers: add, get, search, update, delete document
"""

import os
from src.crud.crud_operations import CRUDOperations
from src.utils.config import DATA_DIR

def main():
    crud = CRUDOperations()

    # -------------------- Setup: create sample file --------------------
    sample_file_path = DATA_DIR / "sample_crud.txt"
    sample_text = """This is a test document for CRUD operations.
It has multiple paragraphs.

Second paragraph for embedding and testing."""
    
    with open(sample_file_path, "w", encoding="utf-8") as f:
        f.write(sample_text)

    doc_id = sample_file_path.name

    # -------------------- CREATE / ADD --------------------
    print("Adding document...")
    doc_json = crud.add_document(str(sample_file_path))
    print("Document added:", doc_json["metadata"]["filename"])

    # -------------------- READ / GET --------------------
    print("\nFetching document...")
    fetched_doc = crud.get_document(doc_id)
    print("Fetched document metadata:", fetched_doc.get("metadata", {}))

    # -------------------- SEARCH --------------------
    print("\nSearching similar paragraphs...")
    query_text = "test document"
    search_results = crud.search_similar(query_text, top_k=2)
    for idx, res in enumerate(search_results):
        print(f"{idx+1}. Paragraph: {res['text'][:50]}..., Score: {res.get('score', 'N/A')}")

    # -------------------- UPDATE --------------------
    print("\nUpdating document with new content...")
    new_file_path = DATA_DIR / "sample_crud_updated.txt"
    new_text = """This is an updated document for CRUD testing.
New content added."""
    
    with open(new_file_path, "w", encoding="utf-8") as f:
        f.write(new_text)

    updated_doc = crud.update_document(doc_id, str(new_file_path))
    print("Document updated. New metadata:", updated_doc["metadata"])

    # -------------------- DELETE --------------------
    print("\nDeleting document...")
    crud.delete_document(doc_id)
    print("Document deleted from vector DB (and optionally from graph DB).")

    # -------------------- Cleanup --------------------
    os.remove(sample_file_path)
    os.remove(new_file_path)
    print("\nTest files cleaned up.")


if __name__ == "__main__":
    main()
