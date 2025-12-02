from sentence_transformers import SentenceTransformer
import chromadb
import json
from pathlib import Path


embedding_model = SentenceTransformer(
    "/models/bge-large",   
    device="cpu"
)
def load_json(filepath: str):
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []

    for item in data:
        
        text = (
            item.get("text") or
            item.get("content") or
            item.get("description") or
            json.dumps(item)  
        )

        docs.append((text, item))

    return docs


def embed_texts(texts):
    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False
    )
    return embeddings.tolist()

def store_embeddings(docs, embeddings,
                     db_path="chroma_db",
                     collection_name="gov_data_rag"):

    client = chromadb.PersistentClient(path=db_path)

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(collection_name)

    texts = [d[0] for d in docs]
    metadatas = [d[1] for d in docs]
    ids = [f"doc-{i}" for i in range(len(texts))]

    collection.add(
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
        ids=ids
    )

    return collection


def build_chroma_index(acts_path: str, schemes_path: str):
    acts = load_json(acts_path)
    schemes = load_json(schemes_path)

    all_docs = acts + schemes
    texts = [d[0] for d in all_docs]

    embeddings = embed_texts(texts)
    store_embeddings(all_docs, embeddings)


if __name__ == "__main__":
    build_chroma_index(
        "/acts.json",
        "/schemes.json"
    )