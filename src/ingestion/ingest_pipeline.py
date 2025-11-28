"""
Ingestion Pipeline using Docling
--------------------------------
Extracts:
- Metadata
- Paragraphs
- Tables (if present)
- Placeholder for entity extraction

Supported formats: txt, pdf, docx, csv
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import ConversionStatus


class IngestionPipeline:
    SUPPORTED_EXT = {".txt", ".pdf", ".docx", ".csv"}

    def __init__(self):
        """
        Initialize the Docling document converter
        """
        # Use default pipeline options from the installed Docling version.
        # The constructor signature may change across versions, so we avoid
        # passing deprecated/removed keyword arguments like `pipeline_options`.
        self.converter = DocumentConverter()

    def run(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single document and return structured JSON.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXT:
            raise ValueError(f"Unsupported file extension: {ext}")

        # For plain text files, bypass Docling and read directly.
        if ext == ".txt":
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            tables = []
        else:
            # Convert document using Docling for supported rich formats
            result = self.converter.convert(str(path))
            if result.status != ConversionStatus.SUCCESS:
                raise RuntimeError(f"Docling conversion failed for {file_path}")

            doc = result.document

            # Extract raw text - Docling uses export_to_text() method
            try:
                raw_text = doc.export_to_text() if hasattr(doc, "export_to_text") else ""
            except Exception:
                # Fallback: try to get text from body or other attributes
                raw_text = getattr(doc, "text", "") or getattr(doc, "body", "") or ""
                # If body is an object, try to convert it
                if hasattr(raw_text, "export_to_text"):
                    raw_text = raw_text.export_to_text()
                elif not isinstance(raw_text, str):
                    raw_text = str(raw_text) if raw_text else ""

            # Extract tables if available
            tables = getattr(doc, "tables", [])

        # Split text into paragraphs
        paragraphs = self._split_into_paragraphs(raw_text)

        # Extract entities (simple keyword-based extraction)
        entities, relationships = self._extract_entities_simple(paragraphs)

        # Metadata
        metadata = {
            "filename": path.name,
            "filesize": path.stat().st_size,
            "extension": path.suffix
        }

        return {
            "source": path.name,
            "type": ext.replace(".", ""),
            "metadata": metadata,
            "paragraphs": paragraphs,
            "tables": tables,
            "entities": entities,
            "relationships": relationships,
        }

    # -------------------- Internal Methods --------------------
    def _split_into_paragraphs(self, raw_text: str) -> List[Dict[str, Any]]:
        """
        Split text into paragraphs. Uses multiple strategies to ensure full text is captured:
        1. Split by double newlines (\n\n) - standard paragraph breaks
        2. If that yields very few paragraphs, also try single newlines
        3. Merge very short fragments with previous paragraph to avoid truncation
        """
        paragraphs = []
        
        # First, try splitting by double newlines (standard paragraph breaks)
        blocks = raw_text.split("\n\n")
        
        # If we get very few blocks and text is long, also try single newlines
        if len(blocks) < 3 and len(raw_text) > 500:
            # Try splitting by single newlines and then grouping
            lines = raw_text.split("\n")
            blocks = []
            current_block = []
            for line in lines:
                line = line.strip()
                if line:
                    current_block.append(line)
                else:
                    if current_block:
                        blocks.append(" ".join(current_block))
                        current_block = []
            if current_block:
                blocks.append(" ".join(current_block))
        
        # Process blocks and merge very short ones to avoid truncation
        idx = 0
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            # If block is very short (< 50 chars) and we have a previous paragraph, merge it
            if len(block) < 50 and paragraphs:
                # Merge with previous paragraph to avoid truncation
                paragraphs[-1]["text"] += " " + block
            else:
                # Normalize whitespace but preserve full content
                # Replace multiple spaces with single space, but keep newlines within paragraph
                normalized = " ".join(block.split())
                if normalized:  # Only add if not empty after normalization
                    paragraphs.append({
                        "id": f"p{idx+1}",
                        "text": normalized
                    })
                    idx += 1
        
        # If we still have no paragraphs but have text, create one paragraph with all text
        if not paragraphs and raw_text.strip():
            normalized = " ".join(raw_text.split())
            if normalized:
                paragraphs.append({
                    "id": "p1",
                    "text": normalized
                })
        
        return paragraphs

    def _extract_entities_simple(self, paragraphs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Simple entity extraction from paragraphs.
        Uses keyword-based extraction for common entities.
        In production, use NER models like spaCy or transformers.
        """
        entities = []
        relationships = []
        entity_map = {}
        
        # Common entity patterns (can be expanded)
        person_keywords = ["alice", "bob", "john", "mary", "david", "sarah", "engineer", "developer", "manager", "director", "ceo", "cto"]
        company_keywords = ["company", "corporation", "inc", "ltd", "llc", "organization", "firm", "enterprise"]
        location_keywords = ["bangalore", "mumbai", "delhi", "new york", "london", "san francisco", "city", "location"]
        tech_keywords = ["ai", "machine learning", "deep learning", "neural network", "algorithm", "database", "graph", "vector"]
        
        for para in paragraphs:
            text_lower = para["text"].lower()
            para_entities = []
            
            # Extract persons (check for person keywords first, then capitalized names)
            words = para["text"].split()
            for word in words:
                word_clean = word.strip(".,!?;:").lower()
                # Check if it's a known person keyword
                if word_clean in person_keywords:
                    entity_id = f"e_{word_clean}"
                    if entity_id not in entity_map:
                        entity_map[entity_id] = {
                            "id": entity_id,
                            "label": "Person",
                            "metadata": {"name": word.strip(".,!?;:").capitalize()}
                        }
                        entities.append(entity_map[entity_id])
                    para_entities.append(entity_id)
                # Check for capitalized names (but skip if it's a company keyword)
                elif (word[0].isupper() and len(word) > 3 and 
                      word_clean not in ["the", "this", "that", "there", "company", "corporation"] and
                      not any(ck in word_clean for ck in company_keywords)):
                    # Only add if not already added as company
                    entity_id = f"e_{word_clean}"
                    if entity_id not in entity_map:
                        entity_map[entity_id] = {
                            "id": entity_id,
                            "label": "Person",
                            "metadata": {"name": word.strip(".,!?;:")}
                        }
                        entities.append(entity_map[entity_id])
                    if entity_id not in para_entities:
                        para_entities.append(entity_id)
            
            # Extract companies/organizations (check for capitalized company names)
            # Look for capitalized words that might be company names
            words = para["text"].split()
            for i, word in enumerate(words):
                word_clean = word.strip(".,!?;:").lower()
                # Check if next word is a company keyword
                if i < len(words) - 1:
                    next_word = words[i+1].strip(".,!?;:").lower()
                    if next_word in company_keywords or any(ck in next_word for ck in company_keywords):
                        company_name = word.strip(".,!?;:")
                        entity_id = f"e_{company_name.lower().replace(' ', '_')}"
                        if entity_id not in entity_map:
                            entity_map[entity_id] = {
                                "id": entity_id,
                                "label": "Company",
                                "metadata": {"name": company_name}
                            }
                            entities.append(entity_map[entity_id])
                        para_entities.append(entity_id)
                # Also check if word itself contains company indicators
                elif any(ck in word_clean for ck in company_keywords):
                    if i > 0:
                        company_name = words[i-1].strip(".,!?;:")
                        entity_id = f"e_{company_name.lower().replace(' ', '_')}"
                        if entity_id not in entity_map:
                            entity_map[entity_id] = {
                                "id": entity_id,
                                "label": "Company",
                                "metadata": {"name": company_name}
                            }
                            entities.append(entity_map[entity_id])
                        para_entities.append(entity_id)
            
            # Extract locations
            for keyword in location_keywords:
                if keyword in text_lower:
                    entity_id = f"e_{keyword}"
                    if entity_id not in entity_map:
                        entity_map[entity_id] = {
                            "id": entity_id,
                            "label": "Location",
                            "metadata": {"name": keyword.capitalize()}
                        }
                        entities.append(entity_map[entity_id])
                    para_entities.append(entity_id)
            
            # Extract tech concepts
            for keyword in tech_keywords:
                if keyword in text_lower:
                    entity_id = f"e_{keyword.replace(' ', '_')}"
                    if entity_id not in entity_map:
                        entity_map[entity_id] = {
                            "id": entity_id,
                            "label": "Concept",
                            "metadata": {"name": keyword, "type": "technology"}
                        }
                        entities.append(entity_map[entity_id])
                    para_entities.append(entity_id)
            
            # Store entity_ids in paragraph for later use
            para["entity_ids"] = para_entities
            
            # Create relationships (simple: if person and company in same paragraph, create WORKS_AT)
            person_entities = [e for e in para_entities if entity_map.get(e, {}).get("label") == "Person"]
            company_entities = [e for e in para_entities if entity_map.get(e, {}).get("label") == "Company"]
            
            for person_id in person_entities:
                for company_id in company_entities:
                    relationships.append({
                        "start": person_id,
                        "end": company_id,
                        "type": "WORKS_AT",
                        "metadata": {"source": para["id"]}
                    })
        
        return entities, relationships
