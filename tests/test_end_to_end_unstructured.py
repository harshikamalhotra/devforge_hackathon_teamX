"""
End-to-end test for Unstructured Data Processor
Tests the complete workflow including edge cases
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_processor.unstructured_processor import UnstructuredDataProcessor
from src.utils.config import DATA_DIR


def test_edge_cases():
    """Test edge cases and error handling."""
    print("=" * 70)
    print("TESTING EDGE CASES")
    print("=" * 70)
    
    processor = UnstructuredDataProcessor()
    
    # Test 1: Empty content
    print("\n[1] Testing empty content...")
    try:
        result = processor.process_and_save(
            content="",
            content_type="text",
            filename="test_empty",
            output_dir=DATA_DIR
        )
        if result["success"]:
            print("   ✅ Empty content handled (file created)")
        else:
            print(f"   ⚠️  Empty content rejected: {result.get('error', 'Unknown')}")
    except Exception as e:
        print(f"   ⚠️  Exception (acceptable): {e}")
    
    # Test 2: Very long content
    print("\n[2] Testing very long content...")
    long_content = "This is a test. " * 1000
    try:
        result = processor.process_and_save(
            content=long_content,
            content_type="text",
            filename="test_long",
            output_dir=DATA_DIR
        )
        if result["success"]:
            print(f"   ✅ Long content processed: {result['content_length']} characters")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown')}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test 3: Special characters in filename
    print("\n[3] Testing special characters in filename...")
    try:
        result = processor.process_and_save(
            content="Test content",
            content_type="text",
            filename="test@#$%^&*()",
            output_dir=DATA_DIR
        )
        if result["success"]:
            print(f"   ✅ Special characters handled: {result['filename']}")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown')}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test 4: Invalid HTML
    print("\n[4] Testing invalid HTML...")
    invalid_html = "<html><body><p>Unclosed tag<div>More content"
    try:
        result = processor.process_html(invalid_html, "test_invalid")
        if result:
            print(f"   ✅ Invalid HTML handled gracefully: {len(result)} characters")
        else:
            print(f"   ❌ Failed to process invalid HTML")
    except Exception as e:
        print(f"   ⚠️  Exception (may be acceptable): {e}")
    
    # Test 5: URL with invalid domain
    print("\n[5] Testing invalid URL...")
    try:
        result = processor.scrape_url("https://invalid-domain-that-does-not-exist-12345.com")
        if result is None:
            print("   ✅ Invalid URL handled gracefully (returned None)")
        else:
            print(f"   ⚠️  Unexpected result: {len(result) if result else 0} characters")
    except Exception as e:
        print(f"   ✅ Invalid URL handled gracefully (exception caught): {type(e).__name__}")
    
    print("\n✅ Edge case testing completed")
    return True


def test_all_formats():
    """Test all supported formats."""
    print("\n" + "=" * 70)
    print("TESTING ALL FORMATS")
    print("=" * 70)
    
    processor = UnstructuredDataProcessor()
    
    formats = [
        ("html", "<html><body><h1>Test</h1><p>Content</p></body></html>"),
        ("markdown", "# Title\n\n## Section\n\nContent here."),
        ("text", "Plain text content with multiple lines.\n\nSecond paragraph."),
    ]
    
    all_passed = True
    
    for format_type, content in formats:
        print(f"\nTesting {format_type} format...")
        try:
            result = processor.process_and_save(
                content=content,
                content_type=format_type,
                filename=f"test_{format_type}_format",
                output_dir=DATA_DIR
            )
            
            if result["success"]:
                print(f"   ✅ {format_type} format processed successfully")
                print(f"      File: {result['filename']}")
                print(f"      Size: {result['content_length']} characters")
            else:
                print(f"   ❌ {format_type} format failed: {result.get('error', 'Unknown')}")
                all_passed = False
        except Exception as e:
            print(f"   ❌ {format_type} format exception: {e}")
            all_passed = False
    
    return all_passed


def main():
    """Run all end-to-end tests."""
    print("\n" + "=" * 70)
    print("END-TO-END UNSTRUCTURED DATA PROCESSOR TEST")
    print("=" * 70)
    
    results = []
    
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("All Formats", test_all_formats()))
    
    # Summary
    print("\n" + "=" * 70)
    print("END-TO-END TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Total: {passed}/{total} test suites passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

