import os
import sys
import json
from sentence_transformers import SentenceTransformer
import chromadb

# Paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_index_db")
COLLECTION_NAME = "cyberpunk_knowledge"
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"

MALICIOUS_FILE = os.path.join(PROJECT_ROOT, "cyberpunk_pages_modified", "Malicious_Injection.txt")

# ─── 1. Load embedding model ───
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ─── 2. Read malicious file ───
with open(MALICIOUS_FILE, "r", encoding="utf-8") as f:
    malicious_text = f.read().strip()

chunk_id = "Malicious_Injection_chunk_000"
source_file = "Malicious_Injection.txt"

# ─── 3. Connect to existing ChromaDB ───
print("Connecting to ChromaDB index...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

# Check if already exists
try:
    existing = collection.get(ids=[chunk_id])
    if existing and existing["ids"]:
        print(f"Chunk {chunk_id} already exists in index. Skipping.")
        sys.exit(0)
except Exception:
    pass

# ─── 4. Generate embedding ───
print("Generating embedding for malicious chunk...")
embedding = model.encode(f"passage: {malicious_text}").tolist()

# ─── 5. Add to collection ───
collection.add(
    ids=[chunk_id],
    documents=[malicious_text],
    embeddings=[embedding],
    metadatas=[{
        "source_file": source_file,
        "chunk_index": 0,
        "total_chunks": 1,
    }],
)

print(f"Added malicious chunk to index: {chunk_id}")
print(f"Total chunks in index now: {collection.count()}")

# ─── 6. Update metadata ───
METADATA_FILE = os.path.join(SCRIPT_DIR, "index_metadata.json")
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata["total_chunks"] = collection.count()
    if source_file not in metadata.get("files_processed", []):
        metadata.setdefault("files_processed", []).append(source_file)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Updated metadata: {METADATA_FILE}")
