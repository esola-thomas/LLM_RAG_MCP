#!/usr/bin/env python3
import argparse, json
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import requests

def embed(ollama_url: str, model: str, text: str):
    r = requests.post(f"{ollama_url}/api/embeddings", json={"model": model, "input":[text]}, timeout=60)
    r.raise_for_status()
    return r.json()["embeddings"][0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--corpus", default="default")
    ap.add_argument("--qdrant", required=True)
    ap.add_argument("--ollama", required=True)
    ap.add_argument("--model", default="nomic-embed-text")
    ap.add_argument("--collection-prefix", default="rag_")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    vec = embed(args.ollama, args.model, args.query)
    client = QdrantClient(url=args.qdrant, prefer_grpc=False)
    collection = f"{args.collection_prefix}{args.corpus}"
    res = client.search(
        collection_name=collection,
        query_vector=vec,
        limit=args.topk,
        with_payload=True,
        query_filter=qm.Filter(must=[qm.FieldCondition(key="corpus_id", match=qm.MatchValue(value=args.corpus))]),
    )
    for i, r in enumerate(res, 1):
        pl = r.payload or {}
        print(f"\n#{i} score={r.score:.4f} {pl.get('source_path')}#{pl.get('section','').strip()}")
        print("-" * 80)
        print(pl.get("chunk_text","")[:1000])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
