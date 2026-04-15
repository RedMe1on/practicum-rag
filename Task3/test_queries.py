import os
import json
import time
from sentence_transformers import SentenceTransformer
import chromadb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_index_db")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "query_results")
COLLECTION_NAME = "cyberpunk_knowledge"
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"

# Load model
print("Loading model...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Load ChromaDB index
print("Loading ChromaDB index...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)
print(f"Collection has {collection.count()} documents\n")

# Test queries
test_queries = [
    "Who is Jyra Andros?",
    "What is cyberware and how does it work?",
    "What corporations exist in Night City?",
]

os.makedirs(RESULTS_DIR, exist_ok=True)

for i, query in enumerate(test_queries):
    print(f"Query {i + 1}: \"{query}\"")
    print("-" * 50)

    query_embedding = model.encode(f"query: {query}")

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    query_result = {
        "query": query,
        "results": [],
    }

    for j in range(len(results["ids"][0])):
        doc = results["documents"][0][j]
        meta = results["metadatas"][0][j]
        dist = results["distances"][0][j]
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

    result_file = os.path.join(RESULTS_DIR, f"query_{i + 1}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(query_result, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved to: {result_file}\n")

print("=" * 60)
print("ALL TEST QUERIES COMPLETED")
print("=" * 60)
