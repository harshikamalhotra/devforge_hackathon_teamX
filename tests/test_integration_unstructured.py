"""
Integration test for Unstructured Data Processor with Ingestion Pipeline
Tests the complete flow: Process → Save → Ingest → Index
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_processor.unstructured_processor import UnstructuredDataProcessor
from src.crud.crud_operations import CRUDOperations
from src.utils.config import DATA_DIR


def test_html_to_ingestion():
    """Test complete flow: HTML → Process → Save → Ingest."""
    print("=" * 70)
    print("TESTING HTML TO INGESTION INTEGRATION")
    print("=" * 70)
    
    processor = UnstructuredDataProcessor()
    crud = CRUDOperations()
    
    # Sample HTML content
    html_content = """
    <html>
    <body>
        <h1>Integration Test Document</h1>
        <h2>Section 1: Introduction</h2>
        <p>This is a test document for integration testing.</p>
        <p>It contains multiple paragraphs and structured content.</p>
        
        <h2>Section 2: Features</h2>
        <ul>
            <li>Feature A: HTML processing</li>
            <li>Feature B: Text extraction</li>
            <li>Feature C: Structure preservation</li>
        </ul>
        
        <h2>Section 3: Conclusion</h2>
        <p>This document tests the complete integration pipeline.</p>
    </body>
    </html>
    """
    
    try:
        # Step 1: Process and save
        print("[1/3] Processing HTML and saving...")
        result = processor.process_and_save(
            content=html_content,
            content_type="html",
            filename="integration_test_html",
            output_dir=DATA_DIR
        )
        
        if not result["success"]:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            return False
        
        print(f"   ✅ File saved: {result['filename']}")
        print(f"   ✅ Content length: {result['content_length']} characters")
        print(f"   ✅ Paragraphs: {result['paragraphs']}")
        
        # Step 2: Ingest the file
        print("\n[2/3] Ingesting document...")
        doc_info = crud.add_document(result["file_path"])
        
        print(f"   ✅ Document ingested")
        print(f"   ✅ Paragraphs extracted: {len(doc_info['paragraphs'])}")
        print(f"   ✅ Entities found: {len(doc_info.get('entities', []))}")
        print(f"   ✅ Relationships found: {len(doc_info.get('relationships', []))}")
        
        # Step 3: Verify content
        print("\n[3/3] Verifying content...")
        if len(doc_info['paragraphs']) > 0:
            print(f"   ✅ Content successfully indexed")
            print(f"   ✅ First paragraph preview: {doc_info['paragraphs'][0]['text'][:80]}...")
            return True
        else:
            print(f"   ❌ No paragraphs extracted")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_markdown_to_ingestion():
    """Test complete flow: Markdown → Process → Save → Ingest."""
    print("\n" + "=" * 70)
    print("TESTING MARKDOWN TO INGESTION INTEGRATION")
    print("=" * 70)
    
    processor = UnstructuredDataProcessor()
    crud = CRUDOperations()
    
    # Sample Markdown content
    markdown_content = """
    # Research Document
    
    ## Abstract
    This is a research document for testing the integration pipeline.
    
    ## Methodology
    We used the following approach:
    - Data collection
    - Analysis
    - Interpretation
    
    ## Results
    The results show significant findings in the field.
    
    ## Conclusion
    This research demonstrates the effectiveness of the system.
    """
    
    try:
        # Step 1: Process and save
        print("[1/3] Processing Markdown and saving...")
        result = processor.process_and_save(
            content=markdown_content,
            content_type="markdown",
            filename="integration_test_markdown",
            output_dir=DATA_DIR
        )
        
        if not result["success"]:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            return False
        
        print(f"   ✅ File saved: {result['filename']}")
        
        # Step 2: Ingest the file
        print("\n[2/3] Ingesting document...")
        doc_info = crud.add_document(result["file_path"])
        
        print(f"   ✅ Document ingested")
        print(f"   ✅ Paragraphs extracted: {len(doc_info['paragraphs'])}")
        
        # Step 3: Verify content
        print("\n[3/3] Verifying content...")
        if len(doc_info['paragraphs']) > 0:
            print(f"   ✅ Content successfully indexed")
            return True
        else:
            print(f"   ❌ No paragraphs extracted")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plain_text_to_ingestion():
    """Test complete flow: Plain Text → Process → Save → Ingest."""
    print("\n" + "=" * 70)
    print("TESTING PLAIN TEXT TO INGESTION INTEGRATION")
    print("=" * 70)
    
    processor = UnstructuredDataProcessor()
    crud = CRUDOperations()
    
    # Sample plain text content
    text_content = """
    Project Documentation
    
    Project Name: Integration Test Project
    Date: 2024-01-15
    
    Overview:
    This project tests the integration of unstructured data processing
    with the ingestion pipeline.
    
    Components:
    1. Data Processor
    2. Ingestion Pipeline
    3. Vector Database
    4. Graph Database
    
    Status: Testing in progress
    """
    
    try:
        # Step 1: Process and save
        print("[1/3] Processing Plain Text and saving...")
        result = processor.process_and_save(
            content=text_content,
            content_type="text",
            filename="integration_test_text",
            output_dir=DATA_DIR
        )
        
        if not result["success"]:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            return False
        
        print(f"   ✅ File saved: {result['filename']}")
        
        # Step 2: Ingest the file
        print("\n[2/3] Ingesting document...")
        doc_info = crud.add_document(result["file_path"])
        
        print(f"   ✅ Document ingested")
        print(f"   ✅ Paragraphs extracted: {len(doc_info['paragraphs'])}")
        
        # Step 3: Verify content
        print("\n[3/3] Verifying content...")
        if len(doc_info['paragraphs']) > 0:
            print(f"   ✅ Content successfully indexed")
            return True
        else:
            print(f"   ❌ No paragraphs extracted")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("UNSTRUCTURED DATA PROCESSOR INTEGRATION TEST SUITE")
    print("=" * 70 + "\n")
    
    results = []
    
    # Run integration tests
    results.append(("HTML to Ingestion", test_html_to_ingestion()))
    results.append(("Markdown to Ingestion", test_markdown_to_ingestion()))
    results.append(("Plain Text to Ingestion", test_plain_text_to_ingestion()))
    
    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

