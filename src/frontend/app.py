import streamlit as st
import numpy as np
import os
import sys
import pathlib
ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.crud.crud_operations import CRUDOperations
from src.hybrid_query.hybrid_retriever import HybridRetriever
from src.utils.config import DATA_DIR

# Initialize services once
@st.cache_resource
def init_services():
    crud = CRUDOperations()
    retriever = HybridRetriever(top_k_vectors=5, top_k_final=5)
    return crud, retriever

crud, retriever = init_services()

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
            doc_info = crud.add_document(str(file_path))
            st.success("Document successfully indexed in Vector+Graph DB!")
            st.json(doc_info["metadata"])
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ---------------------------------------------
# Hybrid Search section
# ---------------------------------------------
st.subheader("🔍 Hybrid Search (Vector + Graph)")
query = st.text_input("Enter your question or text query")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter some text before searching!")
    else:
        with st.spinner("Running hybrid retrieval..."):
            results = retriever.retrieve(query)

        if not results:
            st.info("No matches found.")
        else:
            st.success(f"Found {len(results)} results")
            for idx, res in enumerate(results, start=1):
                doc = res["vector_result"]

                filename = doc.get("metadata", {}).get("filename", "Unknown")
                text_preview = doc.get("text", "")[:200].replace("\n", " ")

                st.markdown(f"""
                ### 🔹 Result {idx}
                **Filename:** {filename}  
                **Vector Score:** `{round(res['vector_score'], 3)}`  
                **Graph Score:** `{res['graph_score']}`  
                **Total Score:** `⭐ {round(res['total_score'], 3)}`  
                """)
                st.write(f"> {text_preview}...")

                if res["graph_score"] > 0:
                    with st.expander("Graph Relationships"):
                        for g in res["graph_relations"]:
                            st.write(f"{g['start']} -[{g['rel_type']}]-> {g['end']}")