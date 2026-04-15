"""
Скрипт для анализа дистанций в ChromaDB.
Показывает cosine similarity (1 - distance) для наглядности.
"""
import os
import json
from sentence_transformers import SentenceTransformer
import chromadb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_index_db")
COLLECTION_NAME = "cyberpunk_knowledge"

model = SentenceTransformer("intfloat/e5-large-v2")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)

test_queries = [
    "Who is Johnny Silverhand?",       # много релевантных чанков
    "What is the weather like?",        # нет релевантных чанков (должно быть далеко)
    "How does netrunning work?",        # средне релевантных
    "Tell me about Adam Smasher",       # много релевантных
]

for query in test_queries:
    print(f"\nQuery: \"{query}\"")
    print("=" * 70)

    query_embedding = model.encode(f"query: {query}").tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    for j in range(len(results["ids"][0])):
        distance = results["distances"][0][j]
        similarity = 1 - distance
        meta = results["metadatas"][0][j]
        preview = results["documents"][0][j][:100].replace("\n", " ")

        bar = "█" * int(similarity * 30) + "░" * (30 - int(similarity * 30))

        print(f"  [{j + 1}] {bar} {similarity:.1%} (dist: {distance:.4f})")
        print(f"       {meta['source_file']} chunk_{meta['chunk_index']:03d}")
        print(f"       {preview}...")
