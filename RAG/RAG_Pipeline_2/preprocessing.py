

import json, re, hashlib
from tqdm.auto import tqdm

# --------------------------------------------------------
#                    TEXT CLEANING UTILS
# --------------------------------------------------------

def strip_html_tags(text: str):
    """Remove HTML tags & entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    return text

def normalize_whitespace(text: str):
    """Collapse multiple spaces/newlines."""
    return re.sub(r"\s+", " ", text).strip()

def clean_text(text: str):
    """General cleaning: HTML, unicode junk, whitespace."""
    if not text:
        return ""
    t = str(text)
    t = strip_html_tags(t)
    t = t.replace("\u200b", " ").replace("\ufeff", " ")
    t = normalize_whitespace(t)
    t = re.sub(r"(?:\n\s*){3,}", "\n\n", t)
    return t

def short_preview(text: str, n=300):
    """Preview snippet used in metadata/UI."""
    return text[:n].strip()

# --------------------------------------------------------
#             SLIDING WINDOW CHUNKING (SAFE)
# --------------------------------------------------------

def chunk_text_sliding_safe(text: str, max_chars: int, overlap: int):
    """
    Safe chunker with deterministic step-based iteration.
    Avoids memory issues and infinite loops.
    """
    text = (text or "").strip()
    if not text:
        return []

    step = max_chars - overlap
    if step <= 0:
        raise ValueError("max_chars must be > overlap")

    chunks = []
    for start in range(0, len(text), step):
        end = start + max_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

    return chunks

# --------------------------------------------------------
#               BUILD DOCS (SCHEMES)
# --------------------------------------------------------

def build_docs_from_schemes_enhanced(
        files,
        max_chars=1200,
        overlap=300,
        max_field_chars=500_000,
        keep_provenance=True,
        dedupe=True
    ):
    """
    Enhanced builder for government SCHEME documents.
    - Structured field chunking
    - Cleaning
    - Large-field truncation
    - Duplicate chunk elimination
    - Preview generation
    """
    docs = []
    doc_counter = 0
    seen_hashes = set()

    step = max_chars - overlap
    if step <= 0:
        raise ValueError("max_chars must be > overlap")

    for f in tqdm(files, desc="Scheme files"):
        items = json.load(open(f, "r", encoding="utf-8"))
        # with open(path, "r", encoding="utf-8") as f:
        #     items = json.load(f)

        for s in tqdm(items, desc=f"Schemes in {f}", leave=False):

            keys = [
                "scheme_name",
                "details",
                "objectives",
                "benefits",
                "eligibility",
                "exclusions",
                "application_process",
                "documents_required",
                "faqs",
                "raw_text"
            ]

            for k in keys:
                raw_val = s.get(k)
                if not raw_val:
                    continue

                txt = clean_text(str(raw_val))

                # truncate very large fields
                if len(txt) > max_field_chars:
                    print(f"[WARN] Truncating scheme field {k} for scheme '{s.get('scheme_name')}'")
                    txt = txt[:max_field_chars]

                # if short field → keep atomic
                if len(txt) <= max_chars:
                    chunk_list = [txt]
                else:
                    chunk_list = chunk_text_sliding_safe(txt, max_chars, overlap)

                for cidx, chunk in enumerate(chunk_list):
                    # dedupe identical chunks
                    if dedupe:
                        h = hashlib.md5(chunk.encode("utf-8")).hexdigest()
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)

                    meta = {
                        "scheme_name": s.get("scheme_name"),
                        "original_field": k,
                        "url": s.get("url"),
                        "source_file": f,
                        "sector": s.get("sector"),
                    }
                    if keep_provenance:
                        meta["page_number"] = s.get("page_number")
                        meta["scraped_at"] = s.get("scraped_at")

                    docs.append({
                        "doc_type": "scheme",
                        "doc_id": f"scheme_{doc_counter}",
                        "chunk_id": f"{k}_{cidx}",
                        "text": chunk,
                        "preview": short_preview(chunk),
                        "metadata": meta
                    })

            doc_counter += 1

    print(f"Built {len(docs)} chunks (schemes).")
    return docs

# --------------------------------------------------------
#               BUILD DOCS (ACTS)
# --------------------------------------------------------

def build_docs_from_acts_enhanced(
        files,
        max_chars=1800,
        overlap=300,
        max_field_chars=500_000,
        keep_provenance=True,
        dedupe=True
    ):
    """
    Enhanced builder for ACTS (legal documents).
    """
    docs = []
    doc_counter = 0
    seen_hashes = set()

    step = max_chars - overlap
    if step <= 0:
        raise ValueError("max_chars must be > overlap")

    for f in tqdm(files, desc="Acts files"):
        data = json.load(open(f, "r", encoding="utf-8"))
        # with open(path, "r", encoding="utf-8") as f:
        #     data = json.load(f)
        acts = data if isinstance(data, list) else list(data.values())

        for act in tqdm(acts, desc=f"Acts in {f}", leave=False):

            act_name = act.get("Act Name") or act.get("act_name") or ""
            act_number = act.get("Act Number") or act.get("act_number")
            version = act.get("Version as on") or act.get("version")

            sections = act.get("Sections") or act.get("sections")

            # Prefer section-wise chunking
            if isinstance(sections, dict) and len(sections) > 0:
                for sec_key, sec_val in tqdm(sections.items(), 
                                             desc="Sections", leave=False):

                    txt = clean_text(str(sec_val))

                    if len(txt) > max_field_chars:
                        print(f"[WARN] Truncating section {sec_key} in act '{act_name}'")
                        txt = txt[:max_field_chars]

                    if len(txt) <= max_chars:
                        chunk_list = [txt]
                    else:
                        chunk_list = chunk_text_sliding_safe(txt, max_chars, overlap)

                    for cidx, chunk in enumerate(chunk_list):
                        if dedupe:
                            h = hashlib.md5(chunk.encode("utf-8")).hexdigest()
                            if h in seen_hashes:
                                continue
                            seen_hashes.add(h)

                        meta = {
                            "act_name": act_name,
                            "act_number": act_number,
                            "version_as_on": version,
                            "source_file": f,
                            "section": sec_key
                        }
                        if keep_provenance:
                            meta["source"] = act.get("source")

                        docs.append({
                            "doc_type": "act",
                            "doc_id": f"act_{doc_counter}",
                            "chunk_id": f"{sec_key}_{cidx}",
                            "text": chunk,
                            "preview": short_preview(chunk),
                            "metadata": meta
                        })

            else:
                # Fallback: no sections → chunk entire text
                raw = act.get("Full Text") or act.get("text") or ""
                txt = clean_text(str(raw))

                if len(txt) > max_field_chars:
                    print(f"[WARN] Truncating full text of act '{act_name}'")
                    txt = txt[:max_field_chars]

                chunk_list = chunk_text_sliding_safe(txt, max_chars, overlap)

                for cidx, chunk in enumerate(chunk_list):
                    if dedupe:
                        h = hashlib.md5(chunk.encode("utf-8")).hexdigest()
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)

                    docs.append({
                        "doc_type": "act",
                        "doc_id": f"act_{doc_counter}",
                        "chunk_id": f"fulltext_{cidx}",
                        "text": chunk,
                        "preview": short_preview(chunk),
                        "metadata": {
                            "act_name": act_name,
                            "act_number": act_number,
                            "version_as_on": version,
                            "source_file": f,
                            "section": "full_text"
                        }
                    })

            doc_counter += 1

    print(f"Built {len(docs)} chunks (acts).")
    return docs
