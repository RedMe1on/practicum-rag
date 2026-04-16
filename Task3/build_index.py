import os
import sys
import time
import json
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, "cyberpunk_pages_modified")
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_index_db")
COLLECTION_NAME = "cyberpunk_knowledge"
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"

CHUNK_SIZE = 750       # ~500-1000 tokens
CHUNK_OVERLAP = 100    # overlap to preserve context

METADATA_FILE = os.path.join(SCRIPT_DIR, "index_metadata.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "query_results")

# ─── 1. Load embedding model ───
print("=" * 60)
print("STEP 1: Loading embedding model")
print("=" * 60)
start_model = time.time()

model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_dim = model.get_sentence_embedding_dimension()

print(f"Model: {EMBEDDING_MODEL_NAME}")
print(f"Repository: https://huggingface.co/intfloat/e5-large-v2")
print(f"Embedding dimension: {embedding_dim}")
print(f"Model loaded in: {time.time() - start_model:.2f}s\n")

# ─── 2. Load and split documents ───
print("=" * 60)
print("STEP 2: Loading and splitting documents")
print("=" * 60)
start_split = time.time()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

all_chunks = []
txt_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")])

for filename in txt_files:
    filepath = os.path.join(INPUT_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into chunks
    chunks = text_splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        chunk_id = f"{os.path.splitext(filename)[0]}_chunk_{i:03d}"
        all_chunks.append({
            "id": chunk_id,
            "text": chunk,
            "source_file": filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
        })

    print(f"  {filename}: {len(chunks)} chunks")

print(f"\nTotal chunks: {len(all_chunks)}")
print(f"Split time: {time.time() - start_split:.2f}s\n")

# ─── 3. Generate embeddings ───
print("=" * 60)
print("STEP 3: Generating embeddings")
print("=" * 60)
start_embed = time.time()

# E5 model expects "passage: " prefix for documents
texts_to_embed = [f"passage: {chunk['text']}" for chunk in all_chunks]

embeddings = model.encode(texts_to_embed, show_progress_bar=True, batch_size=32)

print(f"\nEmbeddings shape: {embeddings.shape}")
print(f"Embedding time: {time.time() - start_embed:.2f}s\n")

# ─── 4. Create ChromaDB index ───
print("=" * 60)
print("STEP 4: Creating ChromaDB index")
print("=" * 60)
start_chroma = time.time()

# Clean previous index
if os.path.exists(CHROMA_DIR):
    import shutil
    shutil.rmtree(CHROMA_DIR)

client = chromadb.PersistentClient(path=CHROMA_DIR)

# Create or get collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

# Prepare data for batch insert
ids = [chunk["id"] for chunk in all_chunks]
documents = [chunk["text"] for chunk in all_chunks]
metadatas = [
    {
        "source_file": chunk["source_file"],
        "chunk_index": chunk["chunk_index"],
        "total_chunks": chunk["total_chunks"],
    }
    for chunk in all_chunks
]

# Insert in batches (Chroma has a limit per batch)
batch_size = 500
for i in range(0, len(ids), batch_size):
    batch_end = min(i + batch_size, len(ids))
    collection.add(
        ids=ids[i:batch_end],
        documents=documents[i:batch_end],
        embeddings=embeddings[i:batch_end].tolist(),
        metadatas=metadatas[i:batch_end],
    )
    print(f"  Inserted batch {i // batch_size + 1}: "
          f"chunks {i}-{batch_end - 1}")

print(f"\nChromaDB index created in: {time.time() - start_chroma:.2f}s")
print(f"Index saved to: {CHROMA_DIR}/\n")

# ─── 5. Save metadata ───
print("=" * 60)
print("STEP 5: Saving metadata")
print("=" * 60)

metadata = {
    "embedding_model": EMBEDDING_MODEL_NAME,
    "model_repo": "https://huggingface.co/intfloat/e5-large-v2",
    "embedding_dimension": embedding_dim,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "source_dir": INPUT_DIR,
    "total_documents": len(txt_files),
    "total_chunks": len(all_chunks),
    "collection_name": COLLECTION_NAME,
    "chroma_dir": CHROMA_DIR,
    "build_time_seconds": round(time.time() - start_model, 2),
    "files_processed": txt_files,
}

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Metadata saved to: {METADATA_FILE}\n")

# ─── 6. Test queries ───
print("=" * 60)
print("STEP 6: Test queries")
print("=" * 60)

os.makedirs(RESULTS_DIR, exist_ok=True)

test_queries = [
    "Who is Johnny Silverhand?",
    "What is cyberware and how does it work?",
    "What corporations exist in Night City?",
]

for i, query in enumerate(test_queries):
    print(f"\nQuery {i + 1}: \"{query}\"")
    print("-" * 50)

    # E5 model expects "query: " prefix for queries
    query_embedding = model.encode(f"query: {query}")

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    # Save results
    query_result = {
        "query": query,
        "results": [],
    }

    for j in range(len(results["ids"][0])):
        doc = results["documents"][0][j]
        meta = results["metadatas"][0][j]
        dist = results["distances"][0][j]

        # Preview first 200 chars
        preview = doc[:200] + "..." if len(doc) > 200 else doc

        print(f"  [{j + 1}] {meta['source_file']} (chunk {meta['chunk_index']})")
        print(f"      Distance: {dist:.4f}")
        print(f"      Preview: {preview}")
        print()

        query_result["results"].append({
            "rank": j + 1,
            "chunk_id": results["ids"][0][j],
            "source_file": meta["source_file"],
            "chunk_index": meta["chunk_index"],
            "distance": round(dist, 4),
            "text_preview": preview,
            "full_text": doc,
        })

    # Save to file
    result_file = os.path.join(RESULTS_DIR, f"query_{i + 1}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(query_result, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved to: {result_file}")

# ─── Summary ───
print("\n" + "=" * 60)
print("BUILD COMPLETE")
print("=" * 60)
print(f"Model:              {EMBEDDING_MODEL_NAME}")
print(f"Source directory:   {INPUT_DIR}")
print(f"Documents:          {len(txt_files)}")
print(f"Total chunks:       {len(all_chunks)}")
print(f"Embedding dim:      {embedding_dim}")
print(f"Index location:     {CHROMA_DIR}/")
print(f"Metadata:           {METADATA_FILE}")
print(f"Query results:      {RESULTS_DIR}/")
total_time = time.time() - start_model
print(f"Total build time:   {total_time:.2f}s ({total_time / 60:.1f} min)")
