import streamlit as st
import sys
import pathlib
ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.crud.crud_operations import CRUDOperations
from src.hybrid_query.hybrid_retriever import HybridRetriever
from src.utils.config import DATA_DIR
# Import vector DB config with fallback for Streamlit caching issues
try:
    from src.utils.config import VECTOR_DB_TYPE, VECTOR_DB_DIR
except ImportError:
    # Fallback if config hasn't been reloaded
    import importlib
    import src.utils.config as config_module
    importlib.reload(config_module)
    VECTOR_DB_TYPE = getattr(config_module, 'VECTOR_DB_TYPE', 'local')
    VECTOR_DB_DIR = getattr(config_module, 'VECTOR_DB_DIR', 'vector_db_store')
from src.ingestion.ingest_pipeline import IngestionPipeline
from src.data_processor.unstructured_processor import UnstructuredDataProcessor
from src.graph_db.graph_loader import GraphLoader
import tempfile
import os
import importlib
import sys

# Force reload graph_loader module to ensure latest version (for development)
if 'src.graph_db.graph_loader' in sys.modules:
    importlib.reload(sys.modules['src.graph_db.graph_loader'])
    from src.graph_db.graph_loader import GraphLoader

# Initialize services - use cache but allow clearing
@st.cache_resource
def init_services():
    crud = CRUDOperations()
    retriever = HybridRetriever(top_k_vectors=10, top_k_final=3)
    return crud, retriever

# Initialize services
crud, retriever = init_services()

# GraphLoader doesn't need caching - create fresh instance each time
# This ensures we get the latest version with all methods

# Helper function to get vector DB info
def get_vector_db_info():
    """Get information about the active vector DB backend."""
    db_type = VECTOR_DB_TYPE
    db_dir = VECTOR_DB_DIR
    
    if db_type == "chromadb":
        storage_info = f"ChromaDB collection in `{db_dir}/`"
        db_name = "ChromaDB"
    else:
        storage_info = f"`{db_dir}/vectors.npy` and `{db_dir}/metadata.json`"
        db_name = "LocalVectorDB (Pure Python)"
    
    return db_name, storage_info, db_type

# Display vector DB backend info in sidebar
with st.sidebar:
    st.markdown("### 🔧 System Configuration")
    db_name, storage_info, db_type = get_vector_db_info()
    st.info(f"**Vector DB:** {db_name}")
    if db_type == "chromadb":
        st.success("✨ Using ChromaDB backend")
    else:
        st.info("📦 Using LocalVectorDB (Pure Python)")

# Add button to clear cache and reload (for development)
if st.sidebar.button("🔄 Reload Services (Clear Cache)"):
    st.cache_resource.clear()
    st.rerun()

st.title("🧠 Vector + Graph Hybrid RAG | DevForge Hackathon")
st.write("Upload files → Ingest → Embed → Store → Hybrid Query")

# ---------------------------------------------
# Upload section
# ---------------------------------------------
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "csv", "docx", "html"])

if uploaded_file:
    file_path = DATA_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"📄 Uploaded: {uploaded_file.name}")
    
    # Check if uploaded file is HTML
    is_html_file = uploaded_file.name.lower().endswith('.html') or uploaded_file.name.lower().endswith('.htm')
    processed_html_path = None
    
    if is_html_file:
        # Automatically process HTML file
        try:
            with st.spinner("🔄 Processing HTML: Extracting meaningful text and removing tags..."):
                processor = UnstructuredDataProcessor()
                result = processor.process_html_file(file_path, output_dir=DATA_DIR)
                
                if result["success"]:
                    processed_html_path = Path(result["file_path"])
                    content_length = result.get('content_length', 0)
                    paragraphs = result.get('paragraphs', 0)
                    
                    if content_length == 0 or paragraphs == 0:
                        st.warning(f"⚠️ HTML processed but extracted minimal content: {content_length} characters, {paragraphs} paragraphs")
                        st.info("💡 The HTML file may contain mostly scripts, styles, or empty content. Check the preview below.")
                    else:
                        st.success(f"✅ HTML processed successfully!")
                        st.info(f"📝 Extracted {content_length} characters, {paragraphs} paragraphs")
                    
                    st.info(f"💾 Processed text saved as: `{result['filename']}`")
                    
                    # Show preview of processed content
                    with st.expander("📄 Preview Processed Content"):
                        if processed_html_path.exists():
                            preview_text = processed_html_path.read_text(encoding='utf-8')
                            if preview_text.strip():
                                st.text_area(
                                    "Processed HTML Content Preview", 
                                    preview_text[:1000] + ("..." if len(preview_text) > 1000 else ""), 
                                    height=200, 
                                    disabled=True, 
                                    label_visibility="collapsed"
                                )
                            else:
                                st.warning("The processed file is empty. The HTML may not contain extractable text content.")
                        else:
                            st.error(f"Processed file not found: {processed_html_path}")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    st.error(f"❌ HTML processing failed: {error_msg}")
                    st.info("💡 Tip: Make sure your HTML file contains readable text content, not just scripts or styles.")
        except Exception as e:
            st.error(f"❌ Error processing HTML: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Determine which file to use for indexing
    file_to_index = processed_html_path if processed_html_path else file_path
    
    if st.button("Process & Index Document"):
        try:
            with st.spinner("Processing document..."):
                doc_info = crud.add_document(str(file_to_index))
            st.success(f"✅ Document successfully indexed in Vector+Graph DB!")
            st.json(doc_info["metadata"])
            st.info(f"📊 Indexed: {len(doc_info['paragraphs'])} paragraphs, {len(doc_info.get('entities', []))} entities, {len(doc_info.get('relationships', []))} relationships")
            
            # Force retriever to reload vector DB by reinitializing
            # Note: This is a workaround - in production, use a proper cache invalidation
            retriever.vector_db = crud.vector_db
            
            # Show where data is stored (dynamic based on vector DB type)
            _, storage_info, _ = get_vector_db_info()
            st.info(f"💾 Data stored in: {storage_info}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# ---------------------------------------------
# Unstructured Data Processing section
# ---------------------------------------------
st.subheader("📝 Process Unstructured Data")
st.write("Convert HTML, Wiki, Markdown, URLs, or plain text into structured format")

# Initialize processor
processor = UnstructuredDataProcessor()

# Content type selector
content_type = st.selectbox(
    "Select Content Type",
    ["html", "markdown", "text", "url"],
    help="Choose the format of your input data"
)

# Input method selector
input_method = st.radio(
    "Input Method",
    ["Paste Content", "Enter URL"],
    horizontal=True
)

content = None
url_input = None

if input_method == "Paste Content":
    if content_type == "html":
        content = st.text_area(
            "Paste HTML Content (Outer HTML)",
            height=200,
            help="Paste the HTML content you want to convert"
        )
    elif content_type == "markdown":
        content = st.text_area(
            "Paste Markdown/Wiki Content",
            height=200,
            help="Paste Markdown or Wiki formatted content"
        )
    else:  # text
        content = st.text_area(
            "Paste Plain Text Content",
            height=200,
            help="Paste plain text, notes, or scraped data"
        )
else:  # URL
    url_input = st.text_input(
        "Enter URL to Scrape",
        placeholder="https://example.com/article",
        help="Enter a URL to scrape and convert to structured text"
    )
    if url_input:
        content = url_input

# Filename input
filename = st.text_input(
    "Output Filename (without extension)",
    value=f"processed_{content_type}",
    help="Name for the output text file"
)

# Process button
if st.button("🔄 Convert & Process", type="primary"):
    if not content or not content.strip():
        st.warning("Please provide content to process!")
    elif not filename or not filename.strip():
        st.warning("Please provide a filename!")
    else:
        try:
            with st.spinner("Processing and converting content..."):
                # Process and save
                if input_method == "Enter URL":
                    result = processor.process_and_save(
                        content=url_input,
                        content_type="url",
                        filename=filename,
                        output_dir=DATA_DIR
                    )
                else:
                    result = processor.process_and_save(
                        content=content,
                        content_type=content_type,
                        filename=filename,
                        output_dir=DATA_DIR
                    )
                
                if result["success"]:
                    st.success(f"✅ Content converted and saved as: `{result['filename']}`")
                    
                    # Get content length and paragraphs with proper defaults
                    content_length = result.get('content_length', 0)
                    paragraphs = result.get('paragraphs', 0)
                    
                    # Debug: Show what we got
                    if content_length == 0 or paragraphs == 0:
                        st.warning(f"⚠️ Warning: Extracted minimal content - {content_length} characters, {paragraphs} paragraphs")
                        st.info("💡 This might indicate the HTML contains mostly scripts, styles, or empty content.")
                        st.info("💡 Check the preview below to see what was extracted.")
                    else:
                        st.info(f"📊 Processed: {content_length} characters, {paragraphs} paragraphs")
                    
                    # Show preview
                    with st.expander("📄 Preview Converted Content"):
                        with open(result["file_path"], "r", encoding="utf-8") as f:
                            preview_text = f.read()
                        st.text_area("Preview Content", preview_text[:1000] + ("..." if len(preview_text) > 1000 else ""), 
                                   height=200, disabled=True, label_visibility="collapsed")
                    
                    # Auto-process option
                    if st.button("🚀 Auto-Process & Index This File"):
                        try:
                            with st.spinner("Processing and indexing document..."):
                                doc_info = crud.add_document(result["file_path"])
                            st.success(f"✅ Document successfully indexed in Vector+Graph DB!")
                            st.json(doc_info["metadata"])
                            st.info(f"📊 Indexed: {len(doc_info['paragraphs'])} paragraphs, {len(doc_info.get('entities', []))} entities, {len(doc_info.get('relationships', []))} relationships")
                            
                            # Force retriever to reload vector DB
                            retriever.vector_db = crud.vector_db
                            
                            # Show where data is stored (dynamic based on vector DB type)
                            _, storage_info, _ = get_vector_db_info()
                            st.info(f"💾 Data stored in: {storage_info}")
                        except Exception as e:
                            st.error(f"❌ Error during indexing: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                    
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# ---------------------------------------------
# Hybrid Search section
# ---------------------------------------------
st.subheader("🔍 Hybrid Search (Vector + Graph)")
query = st.text_input("Enter your question or text query")

# Add weight configuration sliders
col1, col2 = st.columns(2)
with col1:
    vector_weight = st.slider(
        "Vector Weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6, 
        step=0.1,
        help="Weight for vector similarity score (0.0 = ignore vector, 1.0 = vector only)"
    )
with col2:
    graph_weight = st.slider(
        "Graph Weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.4, 
        step=0.1,
        help="Weight for graph proximity score (0.0 = ignore graph, 1.0 = graph only)"
    )

# Auto-normalize weights if they don't sum to 1.0
if abs(vector_weight + graph_weight - 1.0) > 0.01:
    total = vector_weight + graph_weight
    if total > 0:
        vector_weight = vector_weight / total
        graph_weight = graph_weight / total
        st.info(f"⚠️ Weights normalized to: Vector={vector_weight:.2f}, Graph={graph_weight:.2f}")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter some text before searching!")
    else:
        # Ensure retriever uses the latest vector DB instance
        retriever.vector_db = crud.vector_db
        
        with st.spinner("Running hybrid retrieval..."):
            results = retriever.retrieve(query, vector_weight=vector_weight, graph_weight=graph_weight)

        if not results:
            st.info("No matches found.")
        else:
            st.success(f"Found {len(results)} results")
            
            # Graph Visualization Section
            st.subheader("📊 Graph Visualization")
            st.info("💡 This graph shows how the search results are connected. Nodes represent paragraphs (search results) and entities (people, organizations, concepts). Edges show relationships between them based on your query.")
            
            try:
                # Create fresh GraphLoader instance
                graph_loader = GraphLoader()
                
                # Verify the method exists (helps with debugging)
                if not hasattr(graph_loader, 'visualize_hybrid_search_results'):
                    st.error("❌ GraphLoader is missing the visualize_hybrid_search_results method.")
                    st.info("💡 Please restart Streamlit to reload the updated module.")
                    st.stop()
                
                # Create temporary file for the graph image
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=DATA_DIR) as tmp_file:
                    temp_path = tmp_file.name
                
                # Generate visualization
                with st.spinner("Generating graph visualization..."):
                    fig = graph_loader.visualize_hybrid_search_results(
                        search_results=results,
                        query_text=query,
                        output_path=temp_path,
                        figsize=(14, 10),
                        node_size=1500,
                        font_size=9
                    )
                
                if fig is not None:
                    # Display the graph
                    st.image(temp_path, caption=f"Graph visualization for query: '{query}'", use_container_width=True)
                    
                    # Provide download button
                    with open(temp_path, "rb") as img_file:
                        st.download_button(
                            label="📥 Download Graph Visualization",
                            data=img_file.read(),
                            file_name=f"graph_{query.replace(' ', '_')[:50]}.png",
                            mime="image/png"
                        )
                    
                    # Clean up temp file after a delay (Streamlit will handle this)
                    try:
                        # Schedule cleanup (Streamlit will handle file cleanup)
                        pass
                    except:
                        pass
                else:
                    st.info("ℹ️ Graph visualization is empty. This might be because:")
                    st.info("   - No graph relationships found in search results")
                    st.info("   - Graph database is not connected")
                    st.info("   - No entities are linked to the retrieved paragraphs")
            
            except ImportError as e:
                st.warning("⚠️ Graph visualization requires networkx and matplotlib. Install with: `pip install networkx matplotlib`")
            except Exception as e:
                st.warning(f"⚠️ Could not generate graph visualization: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
            
            st.divider()
            
            # Display individual results
            for idx, res in enumerate(results, start=1):
                doc = res["vector_result"]

                # Get filename from nested metadata structure
                metadata = doc.get("metadata", {})
                nested_metadata = metadata.get("metadata", {})
                filename = nested_metadata.get("filename") or metadata.get("source") or doc.get("doc_id", "Unknown")
                
                # Get the full text (not truncated)
                full_text = doc.get("text", "")
                paragraph_id = doc.get("paragraph_id", "")

                # Get scores with proper defaults
                vector_score = res.get('vector_score', 0.0)
                graph_score = res.get('graph_score', 0.0)
                final_score = res.get('final_score', 0.0)
                hop = res.get('hop')
                v_weight = res.get('vector_weight', 0.6)
                g_weight = res.get('graph_weight', 0.4)
                
                # Format hop info
                hop_info = f" (hop={hop})" if hop is not None else " (unreachable)"
                
                st.markdown(f"""
                ### 🔹 Result {idx}
                **Filename:** `{filename}`  
                **Vector Score:** `{round(vector_score, 6)}` | **Graph Score:** `{round(graph_score, 6)}{hop_info}` | **Final Score:** `⭐ {round(final_score, 6)}`  
                **Formula:** `{v_weight:.2f} × {round(vector_score, 6)} + {g_weight:.2f} × {round(graph_score, 6)} = {round(final_score, 6)}`
                """)
                
                # Show full text in a clean, readable format
                st.markdown("**Retrieved Content (Full Paragraph):**")
                # Display full text - ensure we show the complete paragraph without truncation
                if full_text:
                    # Check if text is very short (might be a heading)
                    is_short = len(full_text.strip()) < 90
                    if is_short:
                        st.info("ℹ️ This appears to be a short heading or section title from the document.")
                    
                    # Calculate height based on content length (original sizing maintained)
                    # Use approximately 20 pixels per line, with a minimum of 200px
                    estimated_lines = max(len(full_text) // 80, 10)  # Rough estimate: 80 chars per line
                    text_height = max(200, estimated_lines * 20)
                    # Keep original max height (800px) but allow scrolling for very long content
                    text_height = min(text_height, 800)  # Max 800px, but content will scroll if longer
                    
                    st.text_area(
                        "Retrieved Content Text", 
                        full_text, 
                        height=text_height, 
                        key=f"result_text_{idx}", 
                        disabled=True, 
                        label_visibility="collapsed",
                        help=f"Full paragraph content ({len(full_text)} characters, {len(full_text.split())} words)"
                    )
                    
                    # Show detailed content statistics with better formatting
                    word_count = len(full_text.split())
                    char_count = len(full_text)
                    sentence_count = len([s for s in full_text.split('.') if s.strip()])
                    st.caption(f"📊 **Statistics:** {char_count:,} characters | {word_count:,} words | {sentence_count} sentences")
                else:
                    st.warning("No content retrieved for this result.")

                # Add expander to view the specific paragraph from the document
                doc_file_path = DATA_DIR / filename
                if doc_file_path.exists():
                    with st.expander(f"📄 View Source Paragraph from Document: {filename}", expanded=False):
                        try:
                            # Read document content
                            pipeline = IngestionPipeline()
                            doc_content = pipeline.run(str(doc_file_path))
                            
                            # Find the specific paragraph that was retrieved
                            paragraphs = doc_content.get("paragraphs", [])
                            retrieved_paragraph = None
                            
                            # Try to find the paragraph by paragraph_id
                            if paragraph_id:
                                for para in paragraphs:
                                    if para.get("id") == paragraph_id:
                                        retrieved_paragraph = para
                                        break
                            
                            # If not found by ID, try to find by matching text
                            if not retrieved_paragraph:
                                for para in paragraphs:
                                    para_text = para.get("text", "").strip()
                                    if para_text and para_text == full_text.strip():
                                        retrieved_paragraph = para
                                        break
                            
                            # Display the retrieved paragraph with surrounding context
                            if retrieved_paragraph:
                                para_text = retrieved_paragraph.get("text", "")
                                para_index = None
                                
                                # Find the index of the retrieved paragraph to get surrounding context
                                for i, para in enumerate(paragraphs):
                                    if para.get("id") == retrieved_paragraph.get("id"):
                                        para_index = i
                                        break
                                
                                # Build extended context with previous and next paragraphs
                                context_paragraphs = []
                                if para_index is not None:
                                    # Include previous paragraph if available
                                    if para_index > 0:
                                        prev_para = paragraphs[para_index - 1]
                                        context_paragraphs.append(("⬆️ Previous Paragraph", prev_para.get("text", "")))
                                    
                                    # Current paragraph
                                    context_paragraphs.append(("⭐ Current Paragraph (Retrieved)", para_text))
                                    
                                    # Include next paragraph if available
                                    if para_index < len(paragraphs) - 1:
                                        next_para = paragraphs[para_index + 1]
                                        context_paragraphs.append(("⬇️ Next Paragraph", next_para.get("text", "")))
                                else:
                                    context_paragraphs.append(("Retrieved Paragraph", para_text))
                                
                                # Display with context
                                st.subheader(f"📄 Paragraph {retrieved_paragraph.get('id', 'N/A')} with Surrounding Context")
                                st.info("💡 Showing the retrieved paragraph along with its surrounding context for better understanding.")
                                
                                # Show each paragraph in the context
                                for context_label, context_text in context_paragraphs:
                                    st.markdown(f"**{context_label}:**")
                                    # Calculate height for each paragraph (original sizing)
                                    para_lines = max(len(context_text) // 80, 10)
                                    para_height = max(200, min(para_lines * 20, 600))  # Keep reasonable max height
                                    
                                    st.text_area(
                                        context_label, 
                                        context_text, 
                                        height=para_height, 
                                        key=f"para_context_{idx}_{context_label}", 
                                        disabled=True, 
                                        label_visibility="collapsed",
                                        help=f"{context_label} ({len(context_text)} characters, {len(context_text.split())} words)"
                                    )
                                
                                # Show comprehensive statistics
                                total_chars = sum(len(t) for _, t in context_paragraphs)
                                total_words = sum(len(t.split()) for _, t in context_paragraphs)
                                total_sentences = sum(len([s for s in t.split('.') if s.strip()]) for _, t in context_paragraphs)
                                st.caption(f"📊 **Total Context Statistics:** {total_chars:,} characters | {total_words:,} words | {total_sentences} sentences | {len(context_paragraphs)} paragraphs")
                            else:
                                # Fallback: show the full text we retrieved with context
                                st.subheader("Retrieved Content (Full Text)")
                                
                                # Try to find similar paragraphs for context
                                similar_paragraphs = []
                                for para in paragraphs:
                                    para_text = para.get("text", "").strip()
                                    # If paragraph text contains significant overlap with retrieved text
                                    if para_text and len(para_text) > 50:
                                        # Check for overlap (simple substring check)
                                        if full_text[:100] in para_text or para_text[:100] in full_text:
                                            similar_paragraphs.append(para)
                                            if len(similar_paragraphs) >= 3:  # Limit to 3 similar paragraphs
                                                break
                                
                                # Display retrieved text with similar paragraphs if found
                                if similar_paragraphs:
                                    st.info(f"💡 Found {len(similar_paragraphs)} related paragraphs from the document for context:")
                                    for sim_para in similar_paragraphs:
                                        sim_text = sim_para.get("text", "")
                                        st.markdown(f"**Related Paragraph {sim_para.get('id', 'N/A')}:**")
                                        sim_lines = max(len(sim_text) // 80, 10)
                                        sim_height = max(200, min(sim_lines * 20, 600))  # Original sizing
                                        st.text_area(
                                            f"Related Para {sim_para.get('id')}", 
                                            sim_text, 
                                            height=sim_height, 
                                            key=f"related_para_{idx}_{sim_para.get('id')}", 
                                            disabled=True, 
                                            label_visibility="collapsed"
                                        )
                                else:
                                    # Use original height calculation
                                    fallback_lines = max(len(full_text) // 80, 10)
                                    fallback_height = max(200, min(fallback_lines * 20, 800))  # Original max height
                                    
                                    st.text_area(
                                        "Retrieved Content Text", 
                                        full_text, 
                                        height=fallback_height, 
                                        key=f"fallback_para_{idx}", 
                                        disabled=True, 
                                        label_visibility="collapsed",
                                        help=f"Complete retrieved content ({len(full_text)} characters, {len(full_text.split())} words)"
                                    )
                                
                                # Show comprehensive statistics
                                word_count = len(full_text.split())
                                sentence_count = len([s for s in full_text.split('.') if s.strip()])
                                st.caption(f"📊 **Statistics:** {len(full_text):,} characters | {word_count:,} words | {sentence_count} sentences")
                                st.info("ℹ️ Note: Could not locate exact paragraph in document. Showing retrieved content.")
                            
                            # Show document metadata
                            with st.expander("Document Metadata"):
                                st.json(doc_content.get("metadata", {}))
                        except Exception as e:
                            st.error(f"Error reading document: {e}")
                            # Fallback: try to read as plain text
                            try:
                                if doc_file_path.suffix.lower() == ".txt":
                                    with open(doc_file_path, "r", encoding="utf-8", errors="ignore") as f:
                                        content = f.read()
                                    # Try to find the paragraph in the content
                                    if full_text.strip() in content:
                                        # Find the paragraph context
                                        idx_pos = content.find(full_text.strip())
                                        if idx_pos != -1:
                                            # Show some context around the paragraph
                                            start = max(0, idx_pos - 100)
                                            end = min(len(content), idx_pos + len(full_text) + 100)
                                            context = content[start:end]
                                            st.text_area("Document Context", context, height=300, key=f"doc_content_{idx}_fallback", disabled=True, label_visibility="collapsed")
                                    else:
                                        st.text_area("Document Content", full_text, height=300, key=f"doc_content_{idx}_fallback2", disabled=True, label_visibility="collapsed")
                            except:
                                st.warning("Could not display document content.")
                else:
                    st.info(f"📄 Document file not found: {filename}")

                if res["graph_score"] > 0:
                    with st.expander("Graph Relationships"):
                        for g in res["graph_relations"]:
                            st.write(f"{g.get('source_id', 'N/A')} -[{g.get('rel_type', 'N/A')}]-> {g.get('related_id', 'N/A')}")