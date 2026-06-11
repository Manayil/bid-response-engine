import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# 1. Initialize ChromaDB (This creates a local database folder)
chroma_client = chromadb.PersistentClient(path="./data/chroma_db")

# 2. Use a free, high-quality embedding function
default_ef = embedding_functions.DefaultEmbeddingFunction()

# 3. Create or get the collection
collection = chroma_client.get_or_create_collection(
    name="capability_library", 
    embedding_function=default_ef
)

def index_library():
    # Load your CSV
    df = pd.read_csv('data/capability_library.csv')
    
    # Prepare data for the Vector DB
    # We combine Domain and Project Summary so the AI has context
    documents = [f"Domain: {row['Domain']} | Project: {row['Project Summary']}" for _, row in df.iterrows()]
    metadatas = [{"cert": str(row['Certification']), "domain": row['Domain']} for _, row in df.iterrows()]
    ids = [str(i) for i in range(len(df))]

    # Add to the Vector DB
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("✅ Capability Library successfully indexed into Vector Database!")

def search_library(query, n_results=1):
    # This searches by "meaning" rather than just words
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

if __name__ == "__main__":
    index_library()