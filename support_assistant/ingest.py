from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded.")
print("Embedding dimension:", model.get_embedding_dimension())


# --------------------------------------------------
# Connect to ChromaDB
# --------------------------------------------------

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    },
)

print(f"ChromaDB collection ready: {COLLECTION_NAME}")


# --------------------------------------------------
# Load documents
# --------------------------------------------------

documents = []
ids = []

for file_path in sorted(DOCS_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8").strip()

    if not text:
        continue

    documents.append(text)
    ids.append(file_path.stem)


print("Documents found:", len(documents))


# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------

print("Generating embeddings...")

embeddings = model.encode(
    documents,
    normalize_embeddings=True,
).tolist()

print("Embeddings generated.")


# --------------------------------------------------
# Store in ChromaDB
# --------------------------------------------------

# Rebuild the collection so this script always produces
# a clean deterministic index.

try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    },
)

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
)

print("Documents stored in ChromaDB.")


# --------------------------------------------------
# Verification
# --------------------------------------------------

count = collection.count()

print("Documents in ChromaDB:", count)

print("\nStored document IDs:")

for doc_id in ids:
    print(" -", doc_id)

print("\nChromaDB ingestion completed successfully.")