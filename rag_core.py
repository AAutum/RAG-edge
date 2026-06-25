#!/usr/bin/env python3
"""
rag_core.py — shared core for the RAG stack (used by rag_ingest / rag_query / rag_agent).

"""

import json
import os
import re
import sys
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Topology / configuration
# ---------------------------------------------------------------------------
WORKSPACE = os.environ.get("RAG_WORKSPACE") or os.path.expanduser("~/.openclaw/workspace")
DB_PATH = os.path.join(WORKSPACE, "data", "chroma_db3")

OLLAMA_PRIMARY = "http://127.0.0.1:11434"
HAILO_INFERENCE = "http://192.168.0.100:8000"
HAILO_MODEL = "manifests:qwen3"  # Hailo-Ollama loads on demand; /api/ps unreliable

RATE_LIMIT_CODES = (429, 503, 529)

# Embeddings (nomic-embed-text REQUIRES these task prefixes)
EMBED_MODEL = "nomic-embed-text"
QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "

# Context limits / output budgets
HAILO_CTX = 2048          # Hailo-10H Qwen3 1.7B hard ceiling
HAILO_NUM_PREDICT = 256
PRIMARY_CTX = 65536
PRIMARY_NUM_PREDICT = 1024

# Token estimation: ~3 chars/token is conservative for technical English
# (register names, hex, code skew worse than prose's ~4).
CHARS_PER_TOKEN = 2
TEMPLATE_OVERHEAD_TOKENS = 64   # chat template + role markers + safety margin

# Newline substitute for Hailo-Ollama message content. " " costs 1 token-ish
# per line break; the old "\\n" workaround cost 2 visible chars AND garbled
# formatting. Set to "\\n" only if your build demands it.
HAILO_NEWLINE = " "

INSTRUCTION = (
    "Answer the question using only the context below. "
    "If the context does not contain the answer, say so."
)

PRIMARY_MODEL_PREFS = ["glm-5.1:cloud"]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def est_tokens(text):
    """Conservative token estimate from character count."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def strip_think(text):
    """Remove Qwen3 <think> blocks. Returns '' if thinking never closed
    (i.e. the model spent its entire output budget reasoning)."""
    if "<think>" not in text:
        return text.strip()
    if "</think>" in text:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return ""


def _post_json(url, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _get_json(url, timeout):
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Endpoint health
# ---------------------------------------------------------------------------
def check_endpoint(url, path="/api/tags", timeout=15):
    """Check an Ollama endpoint. Returns (status, available_models)."""
    try:
        data = _get_json(f"{url}{path}", timeout)
        models = [m["name"] for m in data.get("models", [])]
        return "available", models
    except urllib.error.HTTPError as e:
        if e.code in RATE_LIMIT_CODES:
            return "rate_limited", []
        return f"error_{e.code}", []
    except Exception:
        return "unavailable", []


def check_secondary(timeout=15):
    """Ping Hailo-Ollama. Models load on demand, so /api/version is the
    lightest reliable liveness signal; /api/ps is the fallback."""
    for path in ("/api/version", "/api/ps"):
        try:
            _get_json(f"{HAILO_INFERENCE}{path}", timeout)
            return "available"
        except urllib.error.HTTPError as e:
            if e.code in RATE_LIMIT_CODES:
                return "rate_limited"
        except Exception:
            continue
    return "unavailable"


def pick_primary_model(models):
    """Pick the best available chat model from the primary Ollama."""
    
    for pref in PRIMARY_MODEL_PREFS:
        if pref in models:
            return pref
    for m in models:
        if "embed" not in m:
            return m
    return None


def is_secondary(infer_url):
    return bool(infer_url) and infer_url.rstrip("/") == HAILO_INFERENCE.rstrip("/")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def get_embeddings(texts, is_query=False, base_url=OLLAMA_PRIMARY,
                   model=EMBED_MODEL, timeout=120):
    """Batch-embed a list of texts (one HTTP round trip). Applies the
    required nomic task prefix. Returns list of vectors, or None on error."""
    prefix = QUERY_PREFIX if is_query else DOC_PREFIX
    payload = {"model": model, "input": [prefix + t for t in texts]}
    try:
        result = _post_json(f"{base_url}/api/embed", payload, timeout)
        return result["embeddings"]
    except Exception as e:
        print(f"Embedding error: {e}", file=sys.stderr)
        return None


def get_embedding(text, is_query=True, **kwargs):
    """Embed a single text (defaults to query-side prefix)."""
    result = get_embeddings([text], is_query=is_query, **kwargs)
    return result[0] if result else None


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------
def get_client(db_path=DB_PATH):
    try:
        import chromadb
    except ImportError:
        print("ERROR: chromadb not found.", file=sys.stderr)
        print("Run from rag venv: ~/.openclaw/venvs/rag/bin/python", file=sys.stderr)
        sys.exit(1)
    return chromadb.PersistentClient(path=db_path)


def list_collection_names(client):
    """Discover collections from the DB itself — no hardcoded lists to drift."""
    try:
        return [c.name for c in client.list_collections()]
    except Exception:
        return []


def collection_space(collection):
    return (collection.metadata or {}).get("hnsw:space", "l2")


def ensure_collection(client, name, fresh=False):
    """Get-or-create a collection with cosine distance. With fresh=True the
    collection is dropped first (required to migrate legacy L2 collections,
    since Chroma can't change the space of an existing collection)."""
    if fresh:
        try:
            client.delete_collection(name)
            print(f"Dropped existing collection '{name}'", file=sys.stderr)
        except Exception:
            pass
    try:
        coll = client.get_collection(name=name)
        if collection_space(coll) != "cosine":
            print(f"WARNING: collection '{name}' uses "
                  f"'{collection_space(coll)}' distance. Re-ingest with "
                  f"--fresh to migrate to cosine.", file=sys.stderr)
        return coll
    except Exception:
        return client.create_collection(name=name,
                                        metadata={"hnsw:space": "cosine"})


# ---------------------------------------------------------------------------
# Chunking (used at ingest time; lives here so query-side code can reason
# about the same constants)
# ---------------------------------------------------------------------------
def chunk_text(text, chunk_size=768, max_block=1200, overlap_tail=200):
    """Boundary-aware chunking. Splits on paragraphs, splitting oversized
    paragraphs on sentence boundaries, then greedily packs blocks up to
    chunk_size chars. The last block of a chunk is carried into the next
    chunk (if short) for continuity, replacing the old mid-word character
    overlap that produced broken fragments."""
    blocks = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_block:
            blocks.append(para)
        else:
            cur = ""
            for sent in re.split(r"(?<=[.!?])\s+", para):
                if cur and len(cur) + len(sent) + 1 > max_block:
                    blocks.append(cur)
                    cur = sent
                else:
                    cur = f"{cur} {sent}".strip()
            if cur:
                blocks.append(cur)

    chunks, cur_blocks, cur_len = [], [], 0
    for b in blocks:
        if cur_len and cur_len + len(b) + 2 > chunk_size:
            chunks.append("\n\n".join(cur_blocks))
            tail = cur_blocks[-1]
            if len(tail) <= overlap_tail:
                cur_blocks, cur_len = [tail, b], len(tail) + len(b) + 2
            else:
                cur_blocks, cur_len = [b], len(b)
        else:
            cur_blocks.append(b)
            cur_len += len(b) + (2 if cur_len else 0)
    if cur_blocks:
        chunks.append("\n\n".join(cur_blocks))
    return chunks


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def _lexical_overlap(query, text):
    """Fraction of distinctive query terms appearing verbatim in the text.
    Rescues exact-term matches that embeddings fuzz over."""
    terms = set(re.findall(r"[a-z0-9_\-]{3,}", query.lower()))
    if not terms:
        return 0.0
    low = text.lower()
    return sum(1 for t in terms if t in low) / len(terms)


def _dedup_adjacent(results):
    """Drop lower-scoring hits that are the same or adjacent chunk of the
    same source — overlapping chunks otherwise waste context slots on
    near-duplicate text."""
    kept = []
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        src = (r["collection"], r["metadata"].get("source"))
        cn = r["metadata"].get("chunk")
        dup = False
        for k in kept:
            if (k["collection"], k["metadata"].get("source")) != src:
                continue
            kc = k["metadata"].get("chunk")
            
            if cn == kc or (isinstance(cn, int) and isinstance(kc, int)
                            and abs(cn - kc) <= 1):
                dup = True
                break
        if not dup:
            kept.append(r)
    return kept


def query_rag(query_text, top_k=5, min_score=0.25, collections=None,
              db_path=DB_PATH, rerank=True, dedup=True):
    """Query ChromaDB across collections. Returns ranked result dicts with
    'score' (vector similarity), 'lex' (lexical overlap) and 'rank_score'
    (the blend actually used for ordering when rerank=True)."""
    client = get_client(db_path)
    names = collections or list_collection_names(client)

    query_embedding = get_embedding(query_text, is_query=True)
    if query_embedding is None:
        return []

    warned_l2 = False
    all_results = []
    # Over-fetch per collection so dedup + rerank have candidates to work with.
    fetch_k = max(top_k * 5, top_k)

    for name in names:
        try:
            coll = client.get_collection(name=name)
        except Exception:
            continue
        count = coll.count()
        if count == 0:
            continue

        space = collection_space(coll)
        if space != "cosine" and not warned_l2:
            print(f"NOTE: collection '{name}' (and possibly others) uses "
                  f"'{space}' distance; scores are approximate. Re-ingest "
                  f"with --fresh for exact cosine scores.", file=sys.stderr)
            warned_l2 = True

        res = coll.query(
            query_embeddings=[query_embedding],
            n_results=min(fetch_k, count),
            include=["documents", "metadatas", "distances"],
        )
        if not (res and res["ids"] and res["ids"][0]):
            continue

        for i, doc_id in enumerate(res["ids"][0]):
            d = res["distances"][0][i]
            if space == "cosine":
                sim = 1.0 - d
            else:
                # Legacy L2 collection: Chroma reports squared L2. For
                # normalized vectors, ||a-b||^2 = 2(1 - cos), so:
                sim = 1.0 - d / 2.0
            sim = max(0.0, min(1.0, sim))
            if sim < min_score:
                continue
            text = res["documents"][0][i]

            all_results.append({
                "id": doc_id,
                "text": text,
                "metadata": res["metadatas"][0][i] or {},
                "score": round(sim, 4),
                "lex": round(_lexical_overlap(query_text, text), 4),
                "collection": name,
            })

    if dedup:
        all_results = _dedup_adjacent(all_results)

    if rerank:
        for r in all_results:
            r["rank_score"] = round(0.75 * r["score"] + 0.25 * r["lex"], 4)
        all_results.sort(key=lambda r: r["rank_score"], reverse=True)
    else:
        all_results.sort(key=lambda r: r["score"], reverse=True)

    return all_results[:top_k]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _build_context(chunks, budget_tokens):
    """Pack chunks (already ranked) into a context string within a token
    budget. Returns (context_text, used_chunks). Trims the final chunk to
    fit rather than dropping it, but never includes useless fragments."""
    parts, used = [], []
    remaining = budget_tokens
    for c in chunks:
        header = f"[Source: {c['metadata'].get('source', 'unknown')}]\n"
        header_cost = est_tokens(header)
        if remaining - header_cost < 40:  # not worth adding another source
            break
        body_chars = (remaining - header_cost) * CHARS_PER_TOKEN
        body = c["text"][:body_chars]
        if parts and len(body) < 120:  # tiny fragment of a 2nd+ chunk: skip
            break
        parts.append(header + body)
        used.append(c)
        remaining -= header_cost + est_tokens(body)
        if remaining < 60:
            break
    return "\n\n".join(parts), used


def build_prompt(query, chunks, ctx_limit, num_predict, no_think=True):
    """Assemble the final prompt under an explicit token budget.
    Returns (prompt, used_chunks, ctx_budget_tokens)."""
    fixed = (est_tokens(INSTRUCTION) + est_tokens(query)
             + TEMPLATE_OVERHEAD_TOKENS + 16)  # 16: 'Context:'/'Question:'/'Answer:' labels
    ctx_budget = ctx_limit - num_predict - fixed
    if ctx_budget < 0:
        return ("(num_predict is too high. A larger context limit is needed."),0,0
    if ctx_budget < 80:
        # Pathological config — shrink output budget before giving up on context.
        ctx_budget = max(80, ctx_limit - max(96, num_predict // 2) - fixed)
    context_text, used = _build_context(chunks, ctx_budget)
    prompt = (f"{INSTRUCTION}\n\nContext:\n{context_text}\n\n"
              f"Question: {query}\nAnswer:")
    return prompt, used, ctx_budget


def generate_answer(query, context_chunks, infer_url, model,
                    ctx_limit=None, num_predict=None, no_think=True,
                    timeout=240):
    """Generate an answer with explicit token budgeting.

    ctx_limit / num_predict default per endpoint: 2048/256 for
    Hailo-10H Qwen3 hard ceiling, 8192/512 for the primary.
    """
    if (not context_chunks):
        return "No relevant context found in knowledge base."

    secondary = is_secondary(infer_url)
    if ctx_limit is None:
        ctx_limit = HAILO_CTX if secondary else PRIMARY_CTX
    if num_predict is None:
        num_predict = HAILO_NUM_PREDICT if secondary else PRIMARY_NUM_PREDICT

    prompt, used, _ = build_prompt(query, context_chunks, ctx_limit,
                                   num_predict, no_think=no_think)
    wire_prompt = prompt.replace("\n", HAILO_NEWLINE) if secondary else prompt
    options = {"num_predict": num_predict, "temperature": 0.3,
               "num_ctx": ctx_limit}

    def _extract(result):
        msg = result.get("message", {})
        raw = msg.get("content", result.get("response", ""))
        answer = strip_think(raw)
        if not answer:
            return ("(model produced no answer — its output budget was "
                    "consumed by thinking or it returned empty; retry with "
                    "a higher num_predict)")
        return answer

    try:
        result = _post_json(f"{infer_url}/api/chat", {
            "model": model,
            "messages": [{"role": "user", "content": wire_prompt}],
            "stream": False,
            "options": options,
        }, timeout)
        return _extract(result)
    except urllib.error.HTTPError as e:
        # Fallback: /api/generate with a REBUILT, smaller prompt. (Never
        # truncate the assembled prompt — the question is at the end.)
        small_prompt, _, _ = build_prompt(
            query, context_chunks[:1], ctx_limit,
            num_predict, no_think=no_think)
        if secondary:
            small_prompt = small_prompt.replace("\n", HAILO_NEWLINE)
        try:
            result2 = _post_json(f"{infer_url}/api/generate", {
                "model": model,
                "prompt": small_prompt,
                "stream": False,
                "options": options,
            }, timeout)
            return _extract(result2)
        except Exception as e2:
            return f"Error: both chat and generate failed ({e.code}, {e2})"
    except Exception as e:
        return f"Error querying LLM: {e}"


# ---------------------------------------------------------------------------
# Endpoint selection (failover logic)
# ---------------------------------------------------------------------------
def select_inference_endpoint():
    """Returns (infer_url, model, note) — or (None, None, note) if nothing
    is available. Primary first; Hailo on failure/rate-limit."""
    primary_status, primary_models = check_endpoint(OLLAMA_PRIMARY)
    primary_model = pick_primary_model(primary_models)
    if primary_status == "available" and primary_model:
        return OLLAMA_PRIMARY, primary_model, f"primary ({primary_model})"
    if check_secondary() == "available":
        return HAILO_INFERENCE, HAILO_MODEL, (
            f"Hailo failover ({HAILO_MODEL}) — primary was {primary_status}")
    return None, None, f"no endpoint (primary={primary_status}, secondary=down)"
