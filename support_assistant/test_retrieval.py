from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "zepto_policies"


print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_collection(COLLECTION_NAME)

query = "How long does Zepto delivery take?"

print(f"\nQuery: {query}")

query_embedding = model.encode(
    query,
    normalize_embeddings=True,
).tolist()


results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
)


print("\nTop 3 retrieved documents:")

for i, doc_id in enumerate(results["ids"][0]):
    distance = results["distances"][0][i]
    document = results["documents"][0][i]

    print("\n" + "=" * 60)
    print(f"Rank: {i + 1}")
    print(f"Document ID: {doc_id}")
    print(f"Distance: {distance:.4f}")
    print("Content:")
    print(document[:500])