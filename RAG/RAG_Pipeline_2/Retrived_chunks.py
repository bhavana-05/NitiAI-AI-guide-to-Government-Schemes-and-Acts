import json, math, time, os, re
from pathlib import Path
from typing import Any, List, Dict
from tqdm.auto import tqdm
import numpy as np

# HuggingFace transformers / torch
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Weaviate client
import weaviate

# ---------- CONFIG ----------
WEAVIATE_COLLECTION = "GovDocs"
WEAVIATE_URL = "use your weaviate url"
WEAVIATE_API_KEY = "use your weaviate api key"


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = "use your huggingface token"  # set env var in prod
ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

BGE_EMBED_MODEL = "BAAI/bge-m3"
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"

# Embedding & batching
BATCH_SIZE = 32
EMBED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RERANK_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSIFIER_SYSTEM = """
You are a classifier for Indian government queries.

Your task:
- Determine whether the query is about an ACT (law), a SCHEME (government program), or UNKNOWN.
- Determine if the query is SPECIFIC (mentions a particular act/scheme by name/year/section) 
  or GENERIC (general question about acts or schemes).

Rules:
- Acts involve sections, clauses, articles, penalties, definitions, amendments, or legal terms.
- Schemes involve benefits, eligibility, subsidy, grant, target groups, government programs.
- SPECIFIC queries mention: a scheme name, act name, year, section numbers, citations, or IDs.
- GENERIC queries ask about rules, purpose without naming exact titles.

Output format (MUST FOLLOW EXACTLY):
{"doc_type": "...", "specificity": "..."}
"""

# ---------- LLM wrapper (you provided call_hf_router earlier) ----------
def call_hf_router(system_prompt: str, user_prompt: str, max_tokens=512, temperature=0.0) -> str:
    import requests
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    r = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=120)
    data = None
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Non-JSON response: {r.text}")

    if r.status_code >= 400:
        raise RuntimeError(f"Router error: {json.dumps(data, indent=2)}")

    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Bad router response format:\n{json.dumps(data, indent=2)}\nErr: {e}")
    

def classify_query(query: str) -> Dict[str, Any]:
    """Heuristic fallback when LLM classifier is unavailable."""
    text = query.lower()
    doc_type = "unknown"
    specific = False
    scheme_keywords = ["scheme", "subsidy", "benefit", "assistance", "grant", "yojana"]
    act_keywords = ["act", "section", "clause", "article", "law", "rule"]
    if any(k in text for k in scheme_keywords):
        doc_type = "scheme"
    elif any(k in text for k in act_keywords):
        doc_type = "act"
    specific_markers = ["section", "sec", "clause", "rule", "act", "scheme", "yojana", "201", "202"]
    if any(m in text for m in specific_markers):
        specific = True
    return {"doc_type": doc_type, "specific": specific}


def classify_query_llm(query: str) -> Dict[str, str]:
    """
    Uses the LLM to classify query into doc_type and specificity.
    """
    classifier_prompt = f"""
        Classify the following query:

        Query: "{query}"

        Return ONLY a JSON object:
        {{
        "doc_type": "act" | "scheme" | "unknown",
        "specificity": "specific" | "generic"
        }}
    """

    resp = call_hf_router(CLASSIFIER_SYSTEM, classifier_prompt, max_tokens=50, temperature=0.0)

    # Ensure valid JSON output
    try:
        result = json.loads(resp)
        return {
            "doc_type": result.get("doc_type", "unknown"),
            "specific": result.get("specificity", "") == "specific"
        }
    except Exception:
        # If LLM outputs anything weird → fallback to heuristic
        print("⚠️ LLM classification failed! Falling back to heuristic.")
        return classify_query(query)   


def choose_alpha_from_llm_classification(classification: Dict[str, Any]) -> float:
    doc_type = classification["doc_type"]
    specific = classification["specific"]

    if doc_type == "scheme":
        return 0.4 if specific else 0.45
    elif doc_type == "act":
        return 0.55   
    else:
        return 0.45


### Helper functions
def load_embedding_model(model_name=BGE_EMBED_MODEL, device=EMBED_DEVICE):
    print(f"Loading embedding model {model_name} -> {device}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    t = time.time()-t0
    print(f"Loaded embedding model in {t:.1f}s")
    return tokenizer, model, t

# Pooling function (mean pooling)
def mean_pooling(last_hidden, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def embed_texts(tokenizer, model, texts:List[str], batch_size=BATCH_SIZE, device=EMBED_DEVICE):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            # pooling strategy: mean pooling over token embeddings
            last_hidden = out.last_hidden_state
            pooled = mean_pooling(last_hidden, attention_mask)  # (B, D)
            pooled = pooled.cpu().numpy()
            embeddings.append(pooled)
    return np.vstack(embeddings)

def load_reranker(model_name=BGE_RERANKER, device=RERANK_DEVICE):
    print(f"Loading reranker {model_name} -> {device}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    t = time.time() - t0
    print(f"Loaded reranker in {t:.1f}s")
    return tok, model, t

def rerank_with_model(tokenizer, model, query, candidates, device=RERANK_DEVICE, batch_size=32):
    """
    candidates: list[str] texts
    returns scores aligned with candidates
    """
    scores = []
    with torch.no_grad():
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            enc = tokenizer([query]*len(batch), batch, truncation=True, padding=True, return_tensors="pt")
            enc = {k:v.to(device) for k,v in enc.items()}
            out = model(**enc)
            logits = out.logits.squeeze(-1).cpu().numpy()  # shape (B,)
            # If logits are unbounded, optionally pass through sigmoid to get 0-1 score
            scores.extend(logits.tolist())
    return scores

def rerank_candidates(query, candidates, rer_tok, rer_model, device="cuda", batch_size=16):
    """
    Given retrieved candidates, reranks them using BGE reranker.
    Adds 'rerank_score' to each candidate.
    """
    texts = [c["text"] for c in candidates]

    # your existing function
    scores = rerank_with_model(
        rer_tok,
        rer_model,
        query,
        texts,
        device=device,
        batch_size=batch_size
    )

    for i in range(len(candidates)):
        candidates[i]["rerank_score"] = float(scores[i])

    return candidates


def retrieve_hybrid_v4(client, collection_name, query, query_embedding, top_k=50, alpha=0.3):
    collection = client.collections.get(collection_name)

    result = collection.query.hybrid(
        query=query,
        vector=query_embedding,
        alpha=alpha,
        limit=top_k,
        return_properties=["text", "doc_id", "chunk_id", "preview", "metadata_json", "doc_type" ],
        include_vector=False
    )

    docs = []
    for obj in result.objects:
        score = obj.metadata.score
        if score is None:
            score = 0.0

        docs.append({
            "text": obj.properties.get("text", ""),
            "doc_id": obj.properties.get("doc_id", ""),
            "chunk_id": obj.properties.get("chunk_id", ""),
            "preview": obj.properties.get("preview", ""),
            "doc_type": obj.properties.get("doc_type", ""),
            "metadata": json.loads(obj.properties.get("metadata_json", "{}")),
            "hybrid_score": float(score)
        })
    return docs



def retrieve_rerank_topk(
    query: str,
    k: int = 10,
    embed_model_name="BAAI/bge-m3",
    rerank_model_name="BAAI/bge-reranker-v2-m3",
    device="cuda",
):
    """
    For a single query:
    - embeds the query
    - classifies (act/scheme)
    - chooses alpha
    - retrieves top 60 documents from Weaviate
    - reranks them
    - returns top-k reranked documents
    """

    # ---- 1) Load embedding model ----
    tok, emb_model, _ = load_embedding_model(embed_model_name, device=device)

    # ---- 2) Embed query ----
    q_emb = embed_texts(tok, emb_model, [query], batch_size=1, device=device)[0]
    q_emb = q_emb.astype(np.float32)

    # ---- 3) Query classification -> alpha ----
    classification = classify_query_llm(query)
    alpha = choose_alpha_from_llm_classification(classification)

    # ---- 4) Connect to Weaviate ----
    import weaviate
    from weaviate.classes.init import Auth

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )

    # ---- 5) Retrieval (top 60) ----
    retrieved = retrieve_hybrid_v4(
        client,
        WEAVIATE_COLLECTION,
        query,
        q_emb,
        top_k=60,
        alpha=alpha
    )

    # ensure doc_type exists
    for d in retrieved:
        if "doc_type" not in d:
            d["doc_type"] = d.get("metadata", {}).get("doc_type", "unknown")

    # ---- 6) Load reranker ----
    rer_tok, rr_model, _ = load_reranker(rerank_model_name, device=device)

    # ---- 7) Rerank ----
    reranked = rerank_candidates(
        query,
        retrieved,
        rer_tok,
        rr_model,
        device=device,
        batch_size=16
    )

    # ---- 8) Sort by rerank_score ----
    reranked_sorted = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)

    # ---- 9) Close client ----
    client.close()

    # ---- 10) Return top-k texts + metadata ----
    return reranked_sorted[:k]

def extract_texts_from_docs(query, k=5):
    docs = retrieve_rerank_topk(
        query=query,
        k=k
    )
    return [doc["text"] for doc in docs]
