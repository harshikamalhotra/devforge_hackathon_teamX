"""Test .docx text extraction"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.ingest_pipeline import IngestionPipeline

def main():
    pipeline = IngestionPipeline()
    
    docx_file = Path("data/blog 1.docx")
    if docx_file.exists():
        print(f"Processing: {docx_file.name}")
        try:
            result = pipeline.run(str(docx_file))
            print(f"Type: {result['type']}")
            print(f"Paragraphs: {len(result['paragraphs'])}")
            print(f"Raw text length: {len(result.get('raw_text', ''))}")
            
            # Try to get text from Docling directly
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import ConversionStatus
            
            converter = DocumentConverter()
            conv_result = converter.convert(str(docx_file))
            
            if conv_result.status == ConversionStatus.SUCCESS:
                doc = conv_result.document
                print(f"\nDocling document attributes: {dir(doc)}")
                print(f"Has 'text' attribute: {hasattr(doc, 'text')}")
                if hasattr(doc, 'text'):
                    text = getattr(doc, 'text', '')
                    print(f"Text length: {len(text)}")
                    print(f"Text preview: {text[:200]}...")
                
                # Check for other text attributes
                for attr in ['content', 'body', 'main_text', 'full_text']:
                    if hasattr(doc, attr):
                        val = getattr(doc, attr, '')
                        if val:
                            print(f"Found '{attr}': {len(str(val))} chars")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

