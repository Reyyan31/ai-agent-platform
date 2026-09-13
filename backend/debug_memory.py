import os
from sentence_transformers import SentenceTransformer
from app.memory.conversation_memory import memory_collection

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query = "what is my favorite programming language"
    print(f"Querying memory_collection for: '{query}'")
    
    query_embedding = model.encode(query).tolist()
    
    try:
        results = memory_collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
        )
        
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        if not documents:
            print("No documents found in memory_collection.")
            return
            
        print(f"Found {len(documents)} results:")
        for doc, dist in zip(documents, distances):
            print(f"[{dist:.4f}] {doc}")
            
    except Exception as e:
        print(f"Error querying memory_collection: {e}")

if __name__ == "__main__":
    main()
