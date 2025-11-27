import streamlit as st
import sys
import pathlib
ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.crud.crud_operations import CRUDOperations
from src.hybrid_query.hybrid_retriever import HybridRetriever
from src.utils.config import DATA_DIR
from src.ingestion.ingest_pipeline import IngestionPipeline

# Initialize services - use cache but allow clearing
@st.cache_resource
def init_services():
    crud = CRUDOperations()
    retriever = HybridRetriever(top_k_vectors=5, top_k_final=5)
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
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "csv", "docx"])

if uploaded_file:
    file_path = DATA_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"📄 Uploaded: {uploaded_file.name}")

    if st.button("Process & Index Document"):
        try:
            with st.spinner("Processing document..."):
                doc_info = crud.add_document(str(file_path))
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
                
                text_preview = doc.get("text", "")[:200].replace("\n", " ")

                st.markdown(f"""
                ### 🔹 Result {idx}
                **Filename:** {filename}  
                **Vector Score:** `{round(res['vector_score'], 3)}`  
                **Graph Score:** `{res['graph_score']}`  
                **Total Score:** `⭐ {round(res['total_score'], 3)}`  
                """)
                st.write(f"> {text_preview}...")

                # Add button to view full document
                doc_file_path = DATA_DIR / filename
                if doc_file_path.exists():
                    with st.expander(f"📄 View Full Document: {filename}", expanded=False):
                        try:
                            # Read and display document content
                            pipeline = IngestionPipeline()
                            doc_content = pipeline.run(str(doc_file_path))
                            
                            # Display the full text
                            st.subheader("Document Content")
                            full_text = "\n\n".join([para.get("text", "") for para in doc_content.get("paragraphs", [])])
                            st.text_area("", full_text, height=400, key=f"doc_content_{idx}", disabled=True)
                            
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
                                    st.text_area("", content, height=400, key=f"doc_content_{idx}_fallback", disabled=True)
                            except:
                                st.warning("Could not display document content.")
                else:
                    st.info(f"📄 Document file not found: {filename}")

                if res["graph_score"] > 0:
                    with st.expander("Graph Relationships"):
                        for g in res["graph_relations"]:
                            st.write(f"{g.get('source_id', 'N/A')} -[{g.get('rel_type', 'N/A')}]-> {g.get('related_id', 'N/A')}")