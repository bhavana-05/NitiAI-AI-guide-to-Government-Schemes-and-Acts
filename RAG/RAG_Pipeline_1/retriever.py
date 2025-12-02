import os
import requests
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# HF_TOKEN = os.getenv("HF_TOKEN")

# Load local embedding models ONCE
embed_model = SentenceTransformer("./models/bge-large")
reranker = CrossEncoder("./models/bge-reranker-large")


def answer_query(
    query: str,
    hf_token: str,
    chat_history: list = None,
    system_prompt: str = "You are a helpful assistant."
):
    chat_history = chat_history or []

    # ---- 1. ChromaDB: Load collection ----
    client = chromadb.PersistentClient(path="chroma_db")
    collections = client.list_collections()
    if not collections:
        raise ValueError("No collections found inside chroma_db/")
    collection = client.get_collection(collections[0].name)

    # ---- 2. Embed query locally using BGE LARGE ----
    q_emb = embed_model.encode(query, normalize_embeddings=True).tolist()

    # ---- 3. Retrieve top 60 chunks from Chroma ----
    search = collection.query(
        query_embeddings=[q_emb],
        n_results=60,
        include=["documents"]
    )
    docs = search["documents"][0]

    # ---- 4. Rerank them using local BGE reranker ----
    pairs = [[query, d] for d in docs]
    scores = reranker.predict(pairs)

    ranked_docs = [
        doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    ]

    # ---- 5. Summarize top content using HF Llama Chat API ----
    top_context = "\n\n".join(ranked_docs)
    summary_prompt = (
        "Summarize this in no more than 250 words:\n\n" + top_context
    )

    summary = llama_chat(summary_prompt, hf_token)

    # ---- 6. Final answer generation ----
    final_prompt = system_prompt + "\n"

    for turn in chat_history:
        final_prompt += f"{turn['role']}: {turn['content']}\n"

    final_prompt += (
        f"User: {query}\n"
        f"Context Summary: {summary}\n"
        f"Assistant:"
    )

    answer = llama_chat(final_prompt, hf_token)
    return answer


def llama_chat(prompt: str, hf_token: str) -> str:
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are an expert assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300
    }

    r = requests.post(url, headers=headers, json=payload).json()
    return r["choices"][0]["message"]["content"]
