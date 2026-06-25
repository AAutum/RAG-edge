#!/usr/bin/env python3
"""
RAG Query - thin CLI over rag_core. Search the knowledge base and
optionally generate an answer through a given inference endpoint.

Collections are discovered from the DB (no hardcoded list to drift).

Usage:
    rag_query.py "search terms" [options]

Examples:
    rag_query.py "Hailo HEF format" --collection hailo-re --top-k 3
    rag_query.py "BAR0 register map" --format json
    rag_query.py "What is the sales trend from 2025-2026?" --infer-url http://192.168.0.100:8000
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_core as core


def print_results(results):
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, 1):
        src = r["metadata"].get("source", "unknown")
        chunk = r["metadata"].get("chunk", "?")
        print(f"\n--- Result {i} [vec: {r['score']} | lex: {r.get('lex', 0)}"
              f" | rank: {r.get('rank_score', r['score'])}] ---")
        print(f"Collection: {r['collection']}")
        print(f"Source: {src} (chunk {chunk})")
        print(f"Text:\n{r['text'][:500]}")
        if len(r["text"]) > 500:
            print(f"... ({len(r['text'])} chars total)")


def main():
    parser = argparse.ArgumentParser(description="Query RAG knowledge base")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--collection", type=str, help="Search specific collection")
    parser.add_argument("--top-k", type=int, default=5, help="Total results returned")
    parser.add_argument("--min-score", type=float, default=0.3,
                        help="Minimum vector similarity (cosine)")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--db-path", type=str, default=core.DB_PATH)
    parser.add_argument("--infer-url", type=str,
                        help="LLM endpoint for answer generation "
                             "(e.g. http://192.168.0.100:8000)")
    parser.add_argument("--model", type=str, default=core.HAILO_MODEL,
                        help="Model name for inference")
    parser.add_argument("--ctx-limit", type=int, default=None,
                        help="Override context window (default: 2048 for "
                             "Hailo, 8192 otherwise)")
    parser.add_argument("--num-predict", type=int, default=None,
                        help="Override output token budget")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Disable lexical reranking")
    args = parser.parse_args()

    print(f'Querying: "{args.query}"', file=sys.stderr)

    collections = [args.collection] if args.collection else None
    results = core.query_rag(args.query, top_k=args.top_k,
                             min_score=args.min_score,
                             collections=collections,
                             db_path=args.db_path,
                             rerank=not args.no_rerank)

    if args.infer_url and results:
        answer = core.generate_answer(args.query, results, args.infer_url,
                                      model=args.model,
                                      ctx_limit=args.ctx_limit,
                                      num_predict=args.num_predict)
        if args.format == "json":
            print(json.dumps({"answer": answer, "sources": results}, indent=2))
        else:
            print(f"\n{'='*60}\nQUESTION: {args.query}\n{'='*60}\n\n{answer}")
            print(f"\n{'='*60}\nSources ({len(results)} chunks):")
            for r in results[:5]:
                print(f"  [{r['score']:.2f}] {r['metadata'].get('source', 'unknown')}")
        return

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        print_results(results)


if __name__ == "__main__":
    main()
