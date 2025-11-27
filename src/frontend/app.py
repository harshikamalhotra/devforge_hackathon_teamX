import streamlit as st
import sys
import pathlib
ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.crud.crud_operations import CRUDOperations
from src.hybrid_query.hybrid_retriever import HybridRetriever
from src.utils.config import DATA_DIR
from src.ingestion.ingest_pipeline import IngestionPipeline
from src.data_processor.unstructured_processor import UnstructuredDataProcessor

# Initialize services - use cache but allow clearing
@st.cache_resource
def init_services():
    crud = CRUDOperations()
    retriever = HybridRetriever(top_k_vectors=10, top_k_final=2)
    return crud, retriever

# Initialize services
crud, retriever = init_services()

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
                    st.success(f"✅ HTML processed successfully!")
                    st.info(f"📝 Extracted {result['content_length']} characters, {result['paragraphs']} paragraphs")
                    st.info(f"💾 Processed text saved as: `{result['filename']}`")
                    
                    # Show preview of processed content
                    with st.expander("📄 Preview Processed Content"):
                        preview_text = processed_html_path.read_text(encoding='utf-8')
                        st.text_area(
                            "Processed HTML Content Preview", 
                            preview_text[:1000] + ("..." if len(preview_text) > 1000 else ""), 
                            height=200, 
                            disabled=True, 
                            label_visibility="collapsed"
                        )
                else:
                    st.error(f"❌ HTML processing failed: {result.get('error', 'Unknown error')}")
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
            
            # Show where data is stored
            st.info(f"💾 Data stored in: `vector_db_store/vectors.npy` and `vector_db_store/metadata.json`")
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
                    st.info(f"📊 Processed: {result['content_length']} characters, {result['paragraphs']} paragraphs")
                    
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
                            
                            st.info(f"💾 Data stored in: `vector_db_store/vectors.npy` and `vector_db_store/metadata.json`")
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

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter some text before searching!")
    else:
        # Ensure retriever uses the latest vector DB instance
        retriever.vector_db = crud.vector_db
        
        with st.spinner("Running hybrid retrieval..."):
            results = retriever.retrieve(query)

        if not results:
            st.info("No matches found.")
        else:
            st.success(f"Found {len(results)} results")
            for idx, res in enumerate(results, start=1):
                doc = res["vector_result"]

                # Get filename from nested metadata structure
                metadata = doc.get("metadata", {})
                nested_metadata = metadata.get("metadata", {})
                filename = nested_metadata.get("filename") or metadata.get("source") or doc.get("doc_id", "Unknown")
                
                # Get the full text (not truncated)
                full_text = doc.get("text", "")
                paragraph_id = doc.get("paragraph_id", "")

                st.markdown(f"""
                ### 🔹 Result {idx}
                **Filename:** `{filename}`  
                **Vector Score:** `{round(res['vector_score'], 3)}` | **Graph Score:** `{res['graph_score']}` | **Total Score:** `⭐ {round(res['total_score'], 3)}`  
                """)
                
                # Show full text in a clean, readable format
                st.markdown("**Retrieved Content:**")
                # Display full text - use markdown for better formatting, preserve line breaks
                if full_text:
                    # Calculate height based on content length (min 150, max 400)
                    text_height = min(max(150, len(full_text) // 4), 400)
                    st.text_area(
                        "Retrieved Content Text", 
                        full_text, 
                        height=text_height, 
                        key=f"result_text_{idx}", 
                        disabled=True, 
                        label_visibility="collapsed"
                    )
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
                            
                            # Display the retrieved paragraph
                            if retrieved_paragraph:
                                st.subheader(f"Paragraph {retrieved_paragraph.get('id', 'N/A')}")
                                st.text_area("Paragraph Content", retrieved_paragraph.get("text", ""), height=300, key=f"para_content_{idx}", disabled=True, label_visibility="collapsed")
                            else:
                                # Fallback: show the full text we retrieved
                                st.subheader("Retrieved Content")
                                st.text_area("Retrieved Content Text", full_text, height=300, key=f"fallback_para_{idx}", disabled=True, label_visibility="collapsed")
                                st.info("Note: Could not locate exact paragraph in document. Showing retrieved content.")
                            
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