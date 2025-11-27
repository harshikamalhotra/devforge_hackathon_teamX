"""
Unstructured Data Processor
----------------------------
Converts various unstructured data formats (HTML, Wiki, Markdown, URLs) 
into clean structured text files.
"""

import re
import html
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

# Try to import markdown, fallback to basic processing if not available
try:
    import markdown as md_lib
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    md_lib = None


class UnstructuredDataProcessor:
    """Process unstructured data into structured text format."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def process_html(self, html_content: str, source_name: str = "html_content") -> str:
        """
        Process HTML content and extract clean text.
        Enhanced to handle nested divs and social media profiles (LinkedIn, etc.).
        
        Args:
            html_content: Raw HTML string (outer HTML)
            source_name: Name for the source document
            
        Returns:
            Clean structured text
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements (scripts, styles, navigation, ads, etc.)
            unwanted_tags = ["script", "style", "meta", "link", "noscript", "iframe", "svg", "canvas", "title"]
            for tag in unwanted_tags:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Focus on body content - if body exists, work with it; otherwise use whole soup
            main_content = soup.find('body') or soup.find('main') or soup
            
            # Remove navigation, header, footer, and ad-related elements
            # But preserve body/main elements themselves
            unwanted_classes = [
                "nav", "navigation", "navbar", "header", "footer", 
                "sidebar", "menu", "ad", "advertisement", "ads", "banner",
                "cookie", "popup", "modal", "overlay", "tooltip"
            ]
            for class_name in unwanted_classes:
                for element in soup.find_all(class_=re.compile(class_name, re.I)):
                    # Don't remove body or main elements themselves
                    if element.name not in ['body', 'main', 'html']:
                        element.decompose()
            
            # Remove elements with data attributes that suggest non-content
            # But preserve body/main elements
            for element in soup.find_all(attrs={"data-testid": re.compile("ad|banner|nav|footer|header", re.I)}):
                if element.name not in ['body', 'main', 'html']:
                    element.decompose()
            
            # Also remove specific navigation, header, footer tags (but not body/main)
            for tag_name in ['nav', 'header', 'footer']:
                for element in soup.find_all(tag_name):
                    if element.name not in ['body', 'main', 'html']:
                        element.decompose()
            
            # Extract text with structure preservation
            text_parts = []
            seen_texts = set()  # To avoid duplicates
            processed_elements = set()  # Track processed elements
            
            # First, process h1 headings (main titles) - prioritize these and ensure they're at the start
            h1_headings = main_content.find_all('h1')
            for heading in h1_headings:
                level = 1
                text = re.sub(r'\s+', ' ', heading.get_text(strip=True))
                if text and text not in seen_texts and len(text) > 1:
                    # Insert at the beginning for main titles
                    text_parts.insert(0, f"# {text}")
                    seen_texts.add(text)
                    processed_elements.add(heading)
                    
                    # Try to capture related content in the same container (like profile title, location)
                    parent = heading.parent
                    if parent:
                        # Look for sibling divs with additional info (but not nested too deep)
                        for sibling in parent.children:
                            if hasattr(sibling, 'name') and sibling.name in ['div', 'span']:
                                sibling_text = re.sub(r'\s+', ' ', sibling.get_text(separator=' ', strip=True))
                                if sibling_text and len(sibling_text) > 5 and len(sibling_text) < 200 and sibling_text not in seen_texts:
                                    # Check it's not navigation
                                    sibling_classes = ' '.join(sibling.get('class', [])).lower()
                                    if not any(nav_term in sibling_classes for nav_term in ['nav', 'menu', 'header', 'footer', 'ad', 'banner']):
                                        # Add as subtitle/info after h1
                                        text_parts.insert(1, sibling_text)
                                        seen_texts.add(sibling_text)
                                        processed_elements.add(sibling)
            
            # Process other headings (h2-h6)
            for heading in main_content.find_all(['h2', 'h3', 'h4', 'h5', 'h6']):
                if heading in processed_elements:
                    continue
                level = int(heading.name[1])
                text = re.sub(r'\s+', ' ', heading.get_text(strip=True))
                if text and text not in seen_texts and len(text) > 1:
                    text_parts.append(f"{'#' * level} {text}")
                    seen_texts.add(text)
                    processed_elements.add(heading)
                    
                    # Try to capture related content that follows this heading
                    # Look for next siblings that are part of the same section
                    next_elem = heading.next_sibling
                    section_items = []
                    while next_elem and len(section_items) < 5:  # Limit to avoid going too far
                        if hasattr(next_elem, 'name'):
                            # Stop at next heading of same or higher level
                            if next_elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                                next_level = int(next_elem.name[1])
                                if next_level <= level:
                                    break
                            
                            # Capture divs with structured info (like experience items)
                            if next_elem.name == 'div':
                                div_text = re.sub(r'\s+', ' ', next_elem.get_text(separator=' ', strip=True))
                                if div_text and len(div_text) > 10 and div_text not in seen_texts:
                                    div_classes = ' '.join(next_elem.get('class', [])).lower()
                                    if not any(nav_term in div_classes for nav_term in ['nav', 'menu', 'header', 'footer', 'ad', 'banner']):
                                        section_items.append(div_text)
                                        seen_texts.add(div_text)
                                        processed_elements.add(next_elem)
                        next_elem = next_elem.next_sibling if hasattr(next_elem, 'next_sibling') else None
                    
                    if section_items:
                        text_parts.extend(section_items)
            
            # Process paragraphs (search within main content) - skip if already processed
            for para in main_content.find_all('p'):
                if para in processed_elements or any(para in p for p in processed_elements if hasattr(p, '__iter__')):
                    continue
                text = re.sub(r'\s+', ' ', para.get_text(separator=' ', strip=True))
                if text and len(text) > 10 and text not in seen_texts:
                    text_parts.append(text)
                    seen_texts.add(text)
                    processed_elements.add(para)
            
            # Process lists (search within main content)
            for list_elem in main_content.find_all(['ul', 'ol']):
                list_items = []
                for li in list_elem.find_all('li', recursive=False):
                    item_text = re.sub(r'\s+', ' ', li.get_text(separator=' ', strip=True))
                    if item_text and item_text not in seen_texts and len(item_text) > 3:
                        list_items.append(f"- {item_text}")
                        seen_texts.add(item_text)
                if list_items:
                    text_parts.append("\n".join(list_items))
            
            # Process tables (search within main content)
            for table in main_content.find_all('table'):
                rows = []
                for tr in table.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if cells and any(cell for cell in cells):
                        row_text = " | ".join(cells)
                        if row_text not in seen_texts:
                            rows.append(row_text)
                            seen_texts.add(row_text)
                if rows:
                    text_parts.append("\n".join(rows))
            
            # Enhanced: Process meaningful divs (for social media profiles, nested content)
            # Look for divs with substantial text content that aren't already captured
            for div in main_content.find_all('div'):
                # Skip if already processed
                if div in processed_elements:
                    continue
                # Skip if div contains only other block elements (already processed)
                if div.find(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'table']):
                    continue
                
                # Get text from div and clean it
                div_text = re.sub(r'\s+', ' ', div.get_text(separator=' ', strip=True))
                
                # Check if it's substantial content (not just navigation, buttons, etc.)
                if (div_text and 
                    len(div_text) > 20 and  # Minimum length
                    div_text not in seen_texts and
                    # Filter out common non-content patterns
                    not re.match(r'^(Home|About|Contact|Login|Sign|Menu|Search|Follow|Share|Like|Comment|Subscribe|Cookie|Accept|Decline|Close|×|←|→|↑|↓)$', div_text, re.I) and
                    # Check if it looks like meaningful content (has some words)
                    len(div_text.split()) > 3):
                    
                    # Check if parent is not a navigation/header/footer
                    parent_classes = ' '.join(div.parent.get('class', [])) if div.parent else ''
                    if not any(nav_term in parent_classes.lower() for nav_term in ['nav', 'header', 'footer', 'menu', 'sidebar']):
                        text_parts.append(div_text)
                        seen_texts.add(div_text)
                        processed_elements.add(div)
            
            # Process span elements with substantial content (for inline content in divs)
            for span in main_content.find_all('span'):
                span_text = re.sub(r'\s+', ' ', span.get_text(separator=' ', strip=True))
                if (span_text and 
                    len(span_text) > 30 and  # Longer threshold for spans
                    span_text not in seen_texts and
                    len(span_text.split()) > 5 and
                    # Not a button or link text
                    not span.find_parent(['button', 'a']) and
                    # Not in navigation
                    not any(nav_term in ' '.join(span.parent.get('class', [])).lower() for nav_term in ['nav', 'menu', 'header', 'footer'])):
                    text_parts.append(span_text)
                    seen_texts.add(span_text)
            
            # If no structured content found, get all text (fallback)
            if not text_parts:
                text = main_content.get_text(separator='\n', strip=True)
                # Clean up excessive whitespace
                text = re.sub(r'\n{3,}', '\n\n', text)
                # Remove very short lines that are likely noise
                lines = [line for line in text.split('\n') if len(line.strip()) > 5]
                text = '\n'.join(lines)
                return text
            
            # Combine all parts and apply advanced cleaning
            result = "\n\n".join(text_parts)
            
            # Advanced text cleaning and formatting
            result = self._clean_and_format_text(result)
            
            return result.strip()
            
        except Exception as e:
            # Fallback: basic HTML tag removal
            text = re.sub(r'<[^>]+>', '', html_content)
            text = html.unescape(text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Remove very short lines
            lines = [line for line in text.split('\n') if len(line.strip()) > 3]
            text = '\n'.join(lines)
            return text.strip()
    
    def _clean_and_format_text(self, text: str) -> str:
        """
        Advanced text cleaning and formatting for professional output.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Clean, well-formatted text
        """
        # Remove excessive whitespace and normalize
        # First, normalize all whitespace within lines
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty lines (we'll add them back strategically)
            if not line.strip():
                continue
            
            # Clean the line: remove excessive spaces, normalize
            cleaned_line = re.sub(r'[ \t]+', ' ', line.strip())
            
            # Skip very short lines that are likely artifacts
            if len(cleaned_line) < 2:
                continue
            
            # Skip lines that are just punctuation or symbols
            if re.match(r'^[^\w\s]+$', cleaned_line):
                continue
            
            # Remove excessive leading/trailing whitespace from content
            cleaned_line = cleaned_line.strip()
            
            # Create a normalized version for duplicate detection (lowercase, no extra spaces)
            normalized = re.sub(r'\s+', ' ', cleaned_line.lower())
            
            # Skip duplicate consecutive lines
            if cleaned_lines and cleaned_line == cleaned_lines[-1].strip():
                continue
            
            # Skip if we've seen very similar content (fuzzy duplicate detection)
            # Check if this line is substantially similar to any previous line
            is_duplicate = False
            for prev_line in cleaned_lines:
                prev_normalized = re.sub(r'\s+', ' ', prev_line.lower())
                # If one is a substantial substring of the other (80% match)
                if len(normalized) > 30 and len(prev_normalized) > 30:
                    shorter = min(len(normalized), len(prev_normalized))
                    longer = max(len(normalized), len(prev_normalized))
                    if shorter / longer > 0.8:  # 80% similarity
                        # Check if they're very similar
                        if normalized[:int(shorter*0.8)] in prev_normalized or prev_normalized[:int(shorter*0.8)] in normalized:
                            is_duplicate = True
                            break
                elif normalized == prev_normalized:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            cleaned_lines.append(cleaned_line)
        
        # Join lines with proper spacing
        result = '\n'.join(cleaned_lines)
        
        # Normalize multiple newlines (max 2 consecutive)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        # Clean up spacing around headings
        result = re.sub(r'\n(#{1,6}\s+[^\n]+)\n+', r'\n\1\n\n', result)
        
        # Remove excessive spaces between words (but preserve single spaces)
        result = re.sub(r' {2,}', ' ', result)
        
        # Clean up list formatting
        result = re.sub(r'\n(- [^\n]+)\n+(- [^\n]+)', r'\n\1\n\2', result)
        
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in result.split('\n')]
        result = '\n'.join(lines)
        
        # Final normalization: ensure proper paragraph spacing
        # Headings should have content after them, not just empty lines
        result = re.sub(r'(#{1,6}\s+[^\n]+)\n\n\n+', r'\1\n\n', result)
        
        # Remove leading/trailing newlines
        result = result.strip()
        
        return result
    
    def process_markdown(self, markdown_content: str, source_name: str = "markdown_content") -> str:
        """
        Process Markdown/Wiki content.
        
        Args:
            markdown_content: Markdown or Wiki formatted text
            source_name: Name for the source document
            
        Returns:
            Clean structured text (Markdown converted to plain text with structure)
        """
        try:
            # Convert markdown to HTML first, then extract text
            if MARKDOWN_AVAILABLE and md_lib:
                html_content = md_lib.markdown(markdown_content)
            else:
                # Fallback: treat as plain text with markdown structure
                html_content = markdown_content
                # Simple markdown to HTML conversion for basic elements
                html_content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
                html_content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
                html_content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
                html_content = re.sub(r'^\* (.+)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
                html_content = re.sub(r'^- (.+)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract text while preserving structure
            text_parts = []
            
            # Process headings
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                level = int(heading.name[1])
                text = heading.get_text(strip=True)
                if text:
                    text_parts.append(f"\n{'#' * level} {text}\n")
            
            # Process paragraphs
            for para in soup.find_all('p'):
                text = para.get_text(separator=' ', strip=True)
                if text and len(text) > 5:  # Only substantial paragraphs
                    text_parts.append(text)
            
            # Process lists
            for list_elem in soup.find_all(['ul', 'ol']):
                list_items = []
                for li in list_elem.find_all('li', recursive=False):
                    item_text = li.get_text(separator=' ', strip=True)
                    if item_text:
                        list_items.append(f"- {item_text}")
                if list_items:
                    text_parts.append("\n".join(list_items))
            
            # Process code blocks (preserve as-is)
            for code in soup.find_all(['code', 'pre']):
                code_text = code.get_text()
                if code_text:
                    text_parts.append(f"\n```\n{code_text}\n```\n")
            
            # If no structured content found, try to extract all text and split by double newlines
            if not text_parts:
                all_text = soup.get_text(separator='\n', strip=True)
                # Split by double newlines to preserve paragraph structure
                paragraphs = [p.strip() for p in all_text.split('\n\n') if p.strip()]
                text_parts.extend(paragraphs)
            
            result = "\n\n".join(text_parts)
            # Clean up excessive whitespace
            result = re.sub(r'\n{3,}', '\n\n', result)
            
            # If conversion didn't work well, return original markdown with minimal processing
            if len(result.strip()) < len(markdown_content.strip()) * 0.3:
                # Just clean up the original markdown
                cleaned = re.sub(r'\n{3,}', '\n\n', markdown_content.strip())
                return cleaned
            
            return result.strip()
            
        except Exception as e:
            # Fallback: return original markdown
            return markdown_content.strip()
    
    def process_plain_text(self, text_content: str, source_name: str = "text_content") -> str:
        """
        Process plain text content.
        
        Args:
            text_content: Plain text
            source_name: Name for the source document
            
        Returns:
            Clean structured text
        """
        # Clean up the text
        text = text_content.strip()
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove excessive spaces
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
    
    def scrape_url(self, url: str) -> Optional[str]:
        """
        Scrape content from a URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Clean structured text or None if failed
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "meta", "link", "nav", "footer", "header"]):
                script.decompose()
            
            # Try to find main content area
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|main|article', re.I))
            
            if main_content:
                html_content = str(main_content)
            else:
                html_content = str(soup)
            
            # Process the HTML
            return self.process_html(html_content, source_name=url)
            
        except Exception as e:
            return None
    
    def process_html_file(self, html_file_path: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Process an HTML file and convert it to structured text.
        
        Args:
            html_file_path: Path to the HTML file
            output_dir: Directory to save the processed text file (default: same as HTML file)
            
        Returns:
            Dictionary with processing results including path to the processed .txt file
        """
        try:
            if not html_file_path.exists():
                return {
                    "success": False,
                    "error": f"HTML file not found: {html_file_path}",
                    "file_path": None
                }
            
            # Read HTML content
            html_content = html_file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Process HTML
            processed_text = self.process_html(html_content, source_name=html_file_path.stem)
            
            # Determine output path
            if output_dir is None:
                output_dir = html_file_path.parent
            
            # Create output filename (same name, .txt extension)
            output_filename = html_file_path.stem + '.txt'
            output_path = output_dir / output_filename
            
            # Save processed text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            
            return {
                "success": True,
                "file_path": str(output_path),
                "filename": output_filename,
                "original_file": str(html_file_path),
                "content_length": len(processed_text),
                "paragraphs": len([p for p in processed_text.split('\n\n') if p.strip()])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": None
            }
    
    def process_and_save(
        self, 
        content: str, 
        content_type: str, 
        filename: str,
        output_dir: Path,
        url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process content and save as structured text file.
        
        Args:
            content: The content to process
            content_type: Type of content ('html', 'markdown', 'text', 'url')
            filename: Output filename (without extension)
            output_dir: Directory to save the file
            url: Optional URL if content_type is 'url'
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Process based on type
            if content_type == 'html':
                processed_text = self.process_html(content, source_name=filename)
            elif content_type == 'markdown':
                processed_text = self.process_markdown(content, source_name=filename)
            elif content_type == 'url':
                processed_text = self.scrape_url(content)
                if processed_text is None:
                    raise ValueError(f"Failed to scrape URL: {content}")
            else:  # 'text' or default
                processed_text = self.process_plain_text(content, source_name=filename)
            
            # Ensure filename is safe
            safe_filename = re.sub(r'[^\w\s-]', '', filename).strip()
            safe_filename = re.sub(r'[-\s]+', '_', safe_filename)
            if not safe_filename:
                safe_filename = f"processed_{content_type}"
            
            # Add .txt extension if not present
            if not safe_filename.endswith('.txt'):
                safe_filename += '.txt'
            
            # Save to file
            output_path = output_dir / safe_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            
            return {
                "success": True,
                "file_path": str(output_path),
                "filename": safe_filename,
                "content_length": len(processed_text),
                "paragraphs": len([p for p in processed_text.split('\n\n') if p.strip()])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": None
            }

