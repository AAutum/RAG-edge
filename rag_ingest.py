#!/usr/bin/env python3
"""
RAG Ingest - Ingests markdown files into ChromaDB via rag_core.

Usage:
    rag_ingest.py --list
    rag_ingest.py --dir ~/notes --collection research
    rag_ingest.py --all
    rag_ingest.py --all --fresh        # migrate everything to cosine space
"""

import argparse
import glob
import hashlib
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_core as core

EMBED_BATCH = 16  # chunks per /api/embed call


def ingest_directory(doc_dir, collection_name, chroma_path, fresh=False,
                     force=False):
    client = core.get_client(chroma_path)
    collection = core.ensure_collection(client, collection_name, fresh=fresh)

    md_files = sorted(glob.glob(os.path.join(doc_dir, "*.md")))
    print(f"Found {len(md_files)} markdown files in {doc_dir}")
    if not md_files:
        return 0

    total_ingested = 0
    today = datetime.now().strftime("%Y-%m-%d")

    for i, filepath in enumerate(md_files, 1):
        filename = os.path.basename(filepath)
        print(f"\n[{i}/{len(md_files)}] {filename}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            if not text.strip():
                print("  empty file, skipping")
                continue

            file_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

            # Skip unchanged files (cheap metadata probe)
            if not force:
                try:
                    existing = collection.get(where={"source": filename},
                                              limit=1, include=["metadatas"])
                    metas = existing.get("metadatas") or []
                    if metas and metas[0].get("file_hash") == file_hash:
                        print("  unchanged, skipping (--force to re-ingest)")
                        continue
                except Exception:
                    pass

            chunks = core.chunk_text(text)
            print(f"  {len(chunks)} chunks")

            # Remove ALL old chunks for this file first, so a shrinking file
            # doesn't leave stale chunk_N entries behind.
            try:
                collection.delete(where={"source": filename})
            except Exception:
                pass

            inserted = 0
            for start in range(0, len(chunks), EMBED_BATCH):
                batch = chunks[start:start + EMBED_BATCH]
                embeddings = core.get_embeddings(batch, is_query=False)
                if embeddings is None:
                    print(f"  embedding failed for batch at chunk {start}, "
                          f"skipping rest of file")
                    break
                ids, metas = [], []
                for j, _ in enumerate(batch, start=start):
                    ids.append(f"{collection_name}::{filename}::chunk_{j}")
                    metas.append({
                        "source": filename,
                        "chunk": j,
                        "total_chunks": len(chunks),
                        "path": filepath,
                        "date_ingested": today,
                        "file_hash": file_hash,
                    })
                collection.upsert(ids=ids, documents=batch,
                                  embeddings=embeddings, metadatas=metas)
                inserted += len(batch)

            total_ingested += inserted
            print(f"  ingested {inserted} chunks")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\nTotal ingested: {total_ingested} chunks")
    print(f"Collection '{collection_name}' now has {collection.count()} documents")
    return total_ingested


def list_collections(chroma_path):
    client = core.get_client(chroma_path)
    print(f"ChromaDB at {chroma_path}")
    print(f"{'Collection':<30} {'Docs':>8} {'Space':>8}")
    print("-" * 48)
    for name in core.list_collection_names(client):
        col = client.get_collection(name=name)
        print(f"{name:<30} {col.count():>8} {core.collection_space(col):>8}")


def main():
    parser = argparse.ArgumentParser(description="RAG Document Ingestion")
    parser.add_argument("--list", action="store_true", help="List collections")
    parser.add_argument("--collection", default="data", help="Collection name")
    parser.add_argument("--dir", default=None, help="Directory to ingest from")
    parser.add_argument("--all", action="store_true",
                        help="Ingest data/ directories")
    parser.add_argument("--fresh", action="store_true",
                        help="Drop collection(s) first — REQUIRED once to "
                             "migrate legacy L2 collections to cosine")
    parser.add_argument("--force", action="store_true",
                        help="Re-ingest even if file content is unchanged")
    args = parser.parse_args()

    chroma_path = core.DB_PATH

    if args.list:
        list_collections(chroma_path)
        return

    if args.all:

        jobs = [(f.path,"data") for f in os.scandir(os.path.join(core.WORKSPACE,"rag/corpus")) if f.is_dir()]
        for doc_dir, coll in jobs:
            print("=" * 50)
            print(f"Ingesting {doc_dir} -> '{coll}'")
            print("=" * 50)
            ingest_directory(doc_dir, coll, chroma_path,
                             fresh=args.fresh, force=args.force)
    elif args.dir:
        ingest_directory(args.dir, args.collection, chroma_path,
                         fresh=args.fresh, force=args.force)
    else:
        ingest_directory(os.path.join(core.WORKSPACE, "rag/corpus"),
                         args.collection, chroma_path,
                         fresh=args.fresh, force=args.force)


if __name__ == "__main__":
    main()
