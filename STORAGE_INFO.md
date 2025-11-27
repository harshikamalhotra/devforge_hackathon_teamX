# Data Storage Information

## Vector DB Storage

**Location:** `vector_db_store/` directory in project root

**Files:**
- `vectors.npy` - NumPy array containing all embedding vectors (109.62 KB currently)
- `metadata.json` - JSON file containing document IDs, metadata, and text payloads (17.36 KB currently)

**How it works:**
- When you upload and process a document, embeddings are created for each paragraph
- These embeddings are stored in `vectors.npy` as a NumPy array
- Metadata (text, source file, entity IDs) is stored in `metadata.json`
- The vector DB uses cosine similarity to find relevant documents

**To view stored data:**
```python
from src.vector_db.qdrant_client import LocalVectorDB
db = LocalVectorDB()
print(f"Total paragraphs: {len(db.ids)}")
```

## Graph DB Storage

**Location:** Memgraph Docker container (in-memory or persistent volume)

**How it works:**
- Entities extracted from documents are stored as nodes
- Relationships between entities are stored as edges
- Data persists in Docker volume: `memgraph_data` (defined in docker-compose.yml)

**To view stored entities:**
```bash
docker exec memgraph cypher-shell "MATCH (n) RETURN n.id, labels(n) LIMIT 10"
```

**To check entity count:**
```bash
docker exec memgraph cypher-shell "MATCH (n) RETURN count(n)"
```

## Current Status

- ✅ Documents are being stored correctly
- ✅ Search finds documents from all uploaded files (not just sample.txt)
- ✅ Vector DB: 73 paragraphs stored from multiple documents
- ✅ Graph DB: Entities and relationships stored for graph-based retrieval

## Troubleshooting

**If search only shows sample.txt:**
1. Click "🔄 Reload Services (Clear Cache)" in the sidebar
2. Make sure you clicked "Process & Index Document" after uploading
3. Check the sidebar to see which documents are indexed

**If .docx files show 0 paragraphs:**
- This was fixed - .docx files now use `export_to_text()` method
- Re-process the document after the fix

**To see what's stored:**
- Check the sidebar "Storage Info" section
- Or run: `python tests/test_vector_db_contents.py`

