#!/usr/bin/env python3
import argparse, hashlib, os, sys, time, json, subprocess, tempfile
from pathlib import Path
from typing import List, Tuple
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

def sha1(s: str) -> str: return hashlib.sha1(s.encode("utf-8")).hexdigest()

def docx_to_md(path: Path) -> str:
    # prefer pandoc if available, fallback to python-docx (simpler)
    try:
        subprocess.run(["pandoc", "-v"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out = subprocess.check_output(["pandoc", "-f", "docx", "-t", "gfm", str(path)])
        return out.decode("utf-8", errors="ignore")
    except Exception:
        from docx import Document
        d = Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)

def read_text(p: Path) -> str:
    if p.suffix.lower() == ".docx":
        return docx_to_md(p)
    return p.read_text(encoding="utf-8", errors="ignore")

def chunk_markdown(md: str, target_tokens: int = 800, overlap: int = 120) -> List[Tuple[int, str, str]]:
    # very simple tokenizer proxy: ~4 chars ≈ 1 token
    max_len = target_tokens * 4
    olap_len = overlap * 4
    chunks = []
    start = 0
    n = 0
    while start < len(md):
        end = min(len(md), start + max_len)
        # try to end at a heading or paragraph break
        cut = md.rfind("\n## ", start, end)
        if cut == -1:
            cut = md.rfind("\n# ", start, end)
        if cut == -1:
            cut = md.rfind("\n\n", start, end)
        if cut == -1 or cut <= start + max_len * 0.6:
            cut = end
        text = md[start:cut].strip()
        if text:
            section = ""
            # capture last heading
            hidx = md.rfind("\n#", 0, cut)
            if hidx != -1:
                section = md[hidx:md.find("\n", hidx+1)]
            chunks.append((n, text, section))
            n += 1
        start = max(cut - olap_len, cut)
    return chunks

def ensure_collection(client: QdrantClient, name: str, dim: int):
    try:
        client.get_collection(name)
    except Exception:
        client.recreate_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

def embed(ollama_url: str, model: str, texts: List[str]) -> List[List[float]]:
    resp = requests.post(f"{ollama_url}/api/embeddings",
                         json={"model": model, "input": texts}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # compatible with common ollama response shapes
    if "embeddings" in data:
        return data["embeddings"]
    if isinstance(data, dict) and "data" in data:  # some variants
        return [d["embedding"] for d in data["data"]]
    raise RuntimeError("Unexpected embeddings response")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--corpus", default="default")
    ap.add_argument("--qdrant", required=True)
    ap.add_argument("--ollama", required=True)
    ap.add_argument("--model", default="nomic-embed-text")
    ap.add_argument("--collection-prefix", default="rag_")
    args = ap.parse_args()

    root = Path(args.path)
    files = [p for p in root.rglob("*") if p.suffix.lower() in (".md", ".markdown", ".docx")]
    if not files:
        print("No .md/.docx found", file=sys.stderr)
        return 0

    # Assume embedding dims (nomic-embed-text=768). You can probe once by calling /api/embeddings
    dim = 768
    collection = f"{args.collection_prefix}{args.corpus}"
    client = QdrantClient(url=args.qdrant, prefer_grpc=False)
    ensure_collection(client, collection, dim)

    for f in files:
        text = read_text(f)
        doc_id = sha1(f"{f}:{f.stat().st_size}:{int(f.stat().st_mtime)}")
        chunks = chunk_markdown(text)
        if not chunks: continue

        # delete old points for this doc_id
        client.delete(collection, qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]), wait=True)

        batch_texts = [c[1] for c in chunks]
        vecs = embed(args.ollama, args.model, batch_texts)

        points = []
        for (chunk_id, chunk_text, section), vec in zip(chunks, vecs):
            pid = sha1(f"{doc_id}:{chunk_id}")
            payload = {
                "corpus_id": args.corpus,
                "doc_id": doc_id,
                "source_path": str(f),
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "section": section.strip(),
                "timestamp": int(time.time()),
            }
            points.append(qm.PointStruct(id=pid, vector=vec, payload=payload))

        client.upsert(collection_name=collection, points=points, wait=True)
        print(f"Upserted {len(points)} chunks from {f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
