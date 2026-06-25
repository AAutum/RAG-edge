#!/usr/bin/env python3
"""
RAG Failover Agent — thin CLI over rag_core.

Architecture:
  - main: ChromaDB storage + embeddings (Ollama nomic-embed-text)
  - secondary: failover LLM inference (Hailo-10H Qwen3 1.7B, 2048-token ctx)
  - Failover: primary down / rate-limited (429/503/529) -> secondary

Token budgeting, dedup, and rerank all live in rag_core.

Usage:
    rag_agent.py "query" [options]
    rag_agent.py --health
    rag_agent.py --daemon [--interval 300]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
import rag_core as core


def run_health_check(quiet=False):
    primary_status, primary_models = core.check_endpoint(core.OLLAMA_PRIMARY)
    HAILO_status = core.check_secondary()
    primary_chat_model = core.pick_primary_model(primary_models)
    has_embedding = any("embed" in m for m in primary_models)

    status = {
        "timestamp": datetime.now().isoformat(),
        "primary_ollama": primary_status,
        "primary_chat_model": primary_chat_model,
        "primary_embedding": "available" if has_embedding else "missing",
        "hailo_node": HAILO_status,
        "hailo_model": core.HAILO_MODEL,
    }
    if not quiet:
        print(json.dumps(status, indent=2))
        embed_ok = "OK" if has_embedding else "MISSING"
        if primary_chat_model:
            infer = f"OK primary ({primary_chat_model})"
        elif HAILO_status == "available":
            infer = "OK secondary node failover"
        else:
            infer = "DOWN"
        print(f"\nEmbeddings: {embed_ok} | Inference: {infer}",
              file=sys.stderr)
    return status


def run_daemon(interval):
    """Health loop: emit one JSON line on every state TRANSITION (and one at
    startup), so OpenClaw cron / Telegram piping only fires on change."""
    print(f"RAG failover daemon: checking every {interval}s", file=sys.stderr)
    last_state = None
    while True:
        status = run_health_check(quiet=True)
        state = (status["primary_ollama"],
                 status["primary_chat_model"] is not None
                 if "primary_chat_model" in status else False,
                 status["HAILO_status"])
        if state != last_state:
            print(json.dumps({"event": "state_change", **status}), flush=True)
            last_state = state
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="RAG Failover Agent")
    parser.add_argument("query", nargs="?", type=str, help="Search query")
    parser.add_argument("--collection", type=str, help="Search specific collection")
    parser.add_argument("--top-k", type=int, default=5, help="Total results")
    parser.add_argument("--min-score", type=float, default=0.25,
                        help="Minimum vector similarity")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--health", action="store_true", help="Check both endpoints")
    parser.add_argument("--daemon", action="store_true", help="Run health-check daemon")
    parser.add_argument("--interval", type=int, default=300, help="Daemon interval (s)")
    parser.add_argument("--no-infer", action="store_true", help="Retrieval only")
    parser.add_argument("--ctx-limit", type=int, default=None,
                        help="Override context window")
    parser.add_argument("--num-predict", type=int, default=None,
                        help="Override output token budget")

    args = parser.parse_args()

    if args.health:
        run_health_check()
        return
    if args.daemon:
        run_daemon(args.interval)
        return
    if not args.query:
        parser.print_help()
        return

    # Pick inference endpoint (primary -> secondary failover)
    infer_url, infer_model, note = core.select_inference_endpoint()
    if infer_url:
        print(f"Inference: {note}", file=sys.stderr)
    else:
        print(f"Inference: {note} — retrieval only", file=sys.stderr)
        args.no_infer = True

    # Step 1: retrieve
    collections = [args.collection] if args.collection else None
    results = core.query_rag(args.query, top_k=args.top_k,
                             min_score=args.min_score,
                             collections=collections)
    if not results:
        if args.format == "json":
            print(json.dumps({"answer": None, "sources": [],
                              "note": "no results"}))
        else:
            print("No results found in knowledge base.")
        return

    if args.no_infer:
        if args.format == "json":
            print(json.dumps(results, indent=2))
        else:
            for i, r in enumerate(results, 1):
                src = r["metadata"].get("source", "unknown")
                print(f"\n--- Result {i} [vec: {r['score']} | "
                      f"rank: {r.get('rank_score', r['score'])}] ---")
                print(f"Collection: {r['collection']}")
                print(f"Source: {src}")
                print(f"Text:\n{r['text'][:500]}")
        return

    # Step 2: generate
    answer = core.generate_answer(args.query, results, infer_url,
                                  model=infer_model,
                                  ctx_limit=args.ctx_limit,
                                  num_predict=args.num_predict)

    if args.format == "json":
        print(json.dumps({
            "answer": answer,
            "endpoint": infer_url,
            "model": infer_model,
            "sources": [{"source": r["metadata"].get("source", "unknown"),
                         "collection": r["collection"],
                         "score": r["score"]} for r in results],
        }, indent=2))
        return

    print(f"\n{'='*60}\nQUESTION: {args.query}\n{'='*60}\n\n{answer}")
    print(f"\n{'='*60}\nSources ({len(results)} chunks):")
    for r in results[:5]:
        print(f"  [{r['score']:.2f}] {r['metadata'].get('source', 'unknown')}")


if __name__ == "__main__":
    main()
