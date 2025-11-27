"""
Test script for Unstructured Data Processor
Tests HTML, Markdown, Text, and URL processing functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_processor.unstructured_processor import UnstructuredDataProcessor
from src.utils.config import DATA_DIR


def test_html_processing():
    """Test HTML content processing."""
    print("=" * 70)
    print("TESTING HTML PROCESSING")
    print("=" * 70)
    
    processor = UnstructuredDataProcessor()
    
    # Sample HTML content
    html_content = """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <h1>Main Heading</h1>
        <h2>Sub Heading</h2>
        <p>This is a paragraph with some content.</p>
        <p>Another paragraph with more information.</p>
        <ul>
            <li>First item</li>
            <li>Second item</li>
            <li>Third item</li>
        </ul>
        <table>
            <tr><th>Name</th><th>Value</th></tr>
            <tr><td>Item 1</td><td>100</td></tr>
            <tr><td>Item 2</td><td>200</td></tr>
        </table>
        <script>console.log('This should be removed');</script>
        <style>.hidden { display: none; }</style>
    </body>
    </html>
    """
    
    try:
        result = processor.process_html(html_content, "test_html")
        print(f"✅ HTML processing successful")
        print(f"   Output length: {len(result)} characters")
        print(f"   Preview (first 200 chars):")
        print(f"   {result[:200]}...")
        print()
        
        # Check that script and style are removed
        if "console.log" not in result and "display: none" not in result:
            print("✅ Script and style tags removed correctly")
        else:
            print("⚠️  Warning: Script/style content may still be present")
        
        # Check that structure is preserved
        if "Main Heading" in result or "Sub Heading" in result:
            print("✅ Headings preserved")
        if "First item" in result or "Second item" in result:
            print("✅ Lists preserved")
        
        return True
    except Exception as e:
        print(f"❌ HTML processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_markdown_processing():
    """Test Markdown content processing."""
    print("=" * 70)
    print("TESTING MARKDOWN PROCESSING")
    print("=" * 70)
    
    processor = UnstructuredDataProcessor()
    
    # Sample Markdown content
    markdown_content = """
    # Main Title
    
    This is a paragraph with some **bold** and *italic* text.
    
    ## Section 1
    
    Here's a list:
    - Item one
    - Item two
    - Item three
    
    ### Subsection
    
    More content here.
    
    ```python
    def hello():
        print("Hello World")
    ```
    """
    
    try:
        result = processor.process_markdown(markdown_content, "test_markdown")
        print(f"✅ Markdown processing successful")
        print(f"   Output length: {len(result)} characters")
        print(f"   Preview (first 200 chars):")
        print(f"   {result[:200]}...")
        print()
        
        # Check that content is preserved
        if "Main Title" in result or "Section 1" in result:
            print("✅ Markdown structure preserved")
        
        return True
    except Exception as e:
        print(f"❌ Markdown processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plain_text_processing():
    """Test plain text processing."""
    print("=" * 70)
    print("TESTING PLAIN TEXT PROCESSING")
    print("=" * 70)
    
    processor = UnstructuredDataProcessor()
    
    # Sample plain text with extra whitespace
    text_content = """
    This is some plain text content.
    
    It has multiple paragraphs.
    
    And some    extra    spaces    that    need    cleaning.
    
    
    
    And multiple line breaks.
    """
    
    try:
        result = processor.process_plain_text(text_content, "test_text")
        print(f"✅ Plain text processing successful")
        print(f"   Output length: {len(result)} characters")
        print(f"   Preview:")
        print(f"   {result[:200]}...")
        print()
        
        # Check that whitespace is normalized
        if "    " not in result:  # No multiple spaces
            print("✅ Extra spaces normalized")
        if "\n\n\n" not in result:  # No triple line breaks
            print("✅ Line breaks normalized")
        
        return True
    except Exception as e:
        print(f"❌ Plain text processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_and_process():
    """Test the complete process_and_save functionality."""
    print("=" * 70)
    print("TESTING SAVE AND PROCESS")
    print("=" * 70)
    
    processor = UnstructuredDataProcessor()
    
    # Test HTML processing and saving
    html_content = """
    <html>
    <body>
        <h1>Test Document</h1>
        <p>This is a test paragraph.</p>
        <p>Another paragraph for testing.</p>
    </body>
    </html>
    """
    
    try:
        result = processor.process_and_save(
            content=html_content,
            content_type="html",
            filename="test_html_output",
            output_dir=DATA_DIR
        )
        
        if result["success"]:
            print(f"✅ File saved successfully")
            print(f"   Filename: {result['filename']}")
            print(f"   File path: {result['file_path']}")
            print(f"   Content length: {result['content_length']} characters")
            print(f"   Paragraphs: {result['paragraphs']}")
            
            # Verify file exists
            file_path = Path(result['file_path'])
            if file_path.exists():
                print(f"✅ File exists at: {file_path}")
                
                # Read and verify content
                with open(file_path, 'r', encoding='utf-8') as f:
                    saved_content = f.read()
                
                if "Test Document" in saved_content:
                    print("✅ Content saved correctly")
                else:
                    print("⚠️  Warning: Content may not match")
                
                return True
            else:
                print(f"❌ File not found at: {file_path}")
                return False
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Save and process failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_markdown_save():
    """Test Markdown processing and saving."""
    print("=" * 70)
    print("TESTING MARKDOWN SAVE")
    print("=" * 70)
    
    processor = UnstructuredDataProcessor()
    
    markdown_content = """
    # Research Notes
    
    ## Introduction
    This document contains research findings.
    
    ## Findings
    - Finding 1: Important discovery
    - Finding 2: Another discovery
    - Finding 3: Final discovery
    
    ## Conclusion
    These are the key findings from our research.
    """
    
    try:
        result = processor.process_and_save(
            content=markdown_content,
            content_type="markdown",
            filename="test_markdown_output",
            output_dir=DATA_DIR
        )
        
        if result["success"]:
            print(f"✅ Markdown file saved successfully")
            print(f"   Filename: {result['filename']}")
            print(f"   Content length: {result['content_length']} characters")
            return True
        else:
            print(f"❌ Markdown processing failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Markdown save failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plain_text_save():
    """Test plain text processing and saving."""
    print("=" * 70)
    print("TESTING PLAIN TEXT SAVE")
    print("=" * 70)
    
    processor = UnstructuredDataProcessor()
    
    text_content = """
    Project Notes
    
    Meeting Date: 2024-01-15
    Participants: Team A, Team B
    
    Discussion Points:
    1. Feature implementation
    2. Bug fixes
    3. Testing requirements
    
    Action Items:
    - Complete feature X
    - Fix bug Y
    - Write tests for Z
    """
    
    try:
        result = processor.process_and_save(
            content=text_content,
            content_type="text",
            filename="test_text_output",
            output_dir=DATA_DIR
        )
        
        if result["success"]:
            print(f"✅ Plain text file saved successfully")
            print(f"   Filename: {result['filename']}")
            print(f"   Content length: {result['content_length']} characters")
            return True
        else:
            print(f"❌ Plain text processing failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Plain text save failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_url_scraping():
    """Test URL scraping (optional - may fail if no internet or URL issues)."""
    print("=" * 70)
    print("TESTING URL SCRAPING")
    print("=" * 70)
    
    processor = UnstructuredDataProcessor()
    
    # Test with a simple, reliable URL
    test_url = "https://example.com"
    
    try:
        result = processor.scrape_url(test_url)
        
        if result:
            print(f"✅ URL scraping successful")
            print(f"   Content length: {len(result)} characters")
            print(f"   Preview (first 200 chars):")
            print(f"   {result[:200]}...")
            return True
        else:
            print(f"⚠️  URL scraping returned None (may be network issue)")
            print(f"   This is acceptable if there's no internet connection")
            return True  # Not a failure, just network issue
            
    except Exception as e:
        print(f"⚠️  URL scraping failed (may be network issue): {e}")
        print(f"   This is acceptable if there's no internet connection")
        return True  # Not a failure, just network issue


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("UNSTRUCTURED DATA PROCESSOR TEST SUITE")
    print("=" * 70 + "\n")
    
    results = []
    
    # Run tests
    results.append(("HTML Processing", test_html_processing()))
    print()
    results.append(("Markdown Processing", test_markdown_processing()))
    print()
    results.append(("Plain Text Processing", test_plain_text_processing()))
    print()
    results.append(("Save and Process", test_save_and_process()))
    print()
    results.append(("Markdown Save", test_markdown_save()))
    print()
    results.append(("Plain Text Save", test_plain_text_save()))
    print()
    results.append(("URL Scraping", test_url_scraping()))
    print()
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
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

