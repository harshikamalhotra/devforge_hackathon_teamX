# Vector Database Backends

This module provides two vector database implementations:

1. **LocalVectorDB** (default) - Pure Python implementation using NumPy
2. **ChromaDBClient** - ChromaDB-based implementation

Both implementations provide the same interface, so you can switch between them easily.

## Configuration

To switch between vector database backends, edit `src/utils/config.py`:

```python
# Use "local" for pure Python implementation (default)
# Use "chromadb" for ChromaDB implementation
VECTOR_DB_TYPE = "local"  # or "chromadb"
```

## Usage

The recommended way to use the vector database is through the factory function:

```python
from src.vector_db import get_vector_db

# This will use the configured backend from config.py
vector_db = get_vector_db(dim=384, db_dir="vector_db_store")
```

Or import directly if you need a specific implementation:

```python
from src.vector_db import LocalVectorDB, ChromaDBClient

# Use pure Python implementation
vector_db = LocalVectorDB(dim=384, db_dir="vector_db_store")

# Use ChromaDB implementation
vector_db = ChromaDBClient(dim=384, db_dir="vector_db_store")
```

## Differences

### LocalVectorDB
- Pure Python, no external dependencies (beyond NumPy)
- Stores vectors as `.npy` files and metadata as JSON
- Simple and lightweight
- Good for small to medium datasets

### ChromaDBClient
- Uses ChromaDB library (requires `chromadb` package)
- More optimized for larger datasets
- Better performance for similarity search
- Persistent storage with better indexing

## Installation

For ChromaDB support, install the required package:

```bash
pip install chromadb>=0.4.0
```

The package is already listed in `requirements.txt` as an optional dependency.

