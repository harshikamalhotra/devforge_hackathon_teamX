"""
Test script for processing HTML snippet
Processes the provided HTML and extracts meaningful text
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_processor.unstructured_processor import UnstructuredDataProcessor
from src.utils.config import DATA_DIR


def test_html_snippet(html_content: str):
    """
    Test processing of HTML snippet and display meaningful text.
    
    Args:
        html_content: The HTML content to process
    """
    print("=" * 80)
    print("PROCESSING HTML SNIPPET - EXTRACTING MEANINGFUL TEXT")
    print("=" * 80)
    print()
    
    processor = UnstructuredDataProcessor()
    
    try:
        # Process the HTML
        print("[1/2] Processing HTML content...")
        processed_text = processor.process_html(html_content, source_name="html_snippet")
        
        print(f"[2/2] Extraction complete!")
        print()
        print("=" * 80)
        print("EXTRACTED MEANINGFUL TEXT")
        print("=" * 80)
        print()
        print(processed_text)
        print()
        print("=" * 80)
        print("STATISTICS")
        print("=" * 80)
        print(f"Original HTML length: {len(html_content)} characters")
        print(f"Extracted text length: {len(processed_text)} characters")
        print(f"Compression ratio: {len(processed_text) / len(html_content) * 100:.2f}%")
        
        # Count paragraphs
        paragraphs = [p.strip() for p in processed_text.split('\n\n') if p.strip()]
        print(f"Number of paragraphs: {len(paragraphs)}")
        
        # Count headings
        headings = [line for line in processed_text.split('\n') if line.strip().startswith('#')]
        print(f"Number of headings: {len(headings)}")
        
        # Count list items
        list_items = [line for line in processed_text.split('\n') if line.strip().startswith('-')]
        print(f"Number of list items: {len(list_items)}")
        
        print()
        print("=" * 80)
        print("SAVING TO FILE")
        print("=" * 80)
        
        # Save to file
        output_path = DATA_DIR / "html_snippet_extracted.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_text)
        
        print(f"✓ Extracted text saved to: {output_path}")
        print()
        
        return processed_text
        
    except Exception as e:
        print(f"❌ Error processing HTML: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Try to read HTML from a file first
    html_file_path = DATA_DIR / "test_html_snippet.html"
    
    html_snippet = None
    
    # Check if HTML file exists
    if html_file_path.exists():
        print(f"Reading HTML from: {html_file_path}")
        html_snippet = html_file_path.read_text(encoding='utf-8')
    elif len(sys.argv) > 1:
        # Try to read from command line argument (file path or direct HTML)
        arg_path = Path(sys.argv[1])
        if arg_path.exists():
            html_snippet = arg_path.read_text(encoding='utf-8')
        else:
            # Use as direct HTML string (for small snippets)
            html_snippet = sys.argv[1]
    else:
        # Default: use a sample HTML snippet for testing
        html_snippet = """<body dir="ltr" class="render-mode-BIGPIPE nav-v2 ember-application payment-failure-global-alert-lix-enabled-class icons-loaded boot-complete" data-t-link-to-event-attached="true">
        <div class="main-content">
            <h1>Sample Document Title</h1>
            <p>This is a sample paragraph to test HTML processing.</p>
            <h2>Section 1</h2>
            <p>Another paragraph with meaningful content.</p>
            <ul>
                <li>First item</li>
                <li>Second item</li>
            </ul>
        </div>
        </body>"""
        print("⚠️  No HTML file found. Using sample HTML.")
        print(f"💡 To test your HTML snippet, save it to: {html_file_path}")
        print()
    
    # Run the test
    result = test_html_snippet(html_snippet)
    
    if result:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")
        sys.exit(1)

