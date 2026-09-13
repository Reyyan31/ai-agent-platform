import os
import glob
import uuid
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Splits text into chunks of roughly chunk_size words with overlap."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks


def ingest():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    chroma_dir = os.path.join(base_dir, "chroma_db")

    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Connecting to Chroma PersistentClient at: {chroma_dir}")
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name="personal_knowledge")

    files = glob.glob(os.path.join(data_dir, "*.md")) + glob.glob(os.path.join(data_dir, "*.txt"))
    print(f"Found {len(files)} files in {data_dir}")

    total_chunks = 0
    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            continue

        chunks = chunk_text(content, chunk_size=300, overlap=50)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{os.path.basename(filepath)}_{idx}_{uuid.uuid4().hex[:8]}"
            embedding = model.encode(chunk).tolist()

            ids.append(chunk_id)
            embeddings.append(embedding)
            metadatas.append({"text": chunk, "source": os.path.basename(filepath)})
            documents.append(chunk)

    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        total_chunks = len(ids)

    print(f"Ingestion complete: Ingested {total_chunks} chunk(s) into collection 'personal_knowledge'.")


if __name__ == "__main__":
    ingest()
