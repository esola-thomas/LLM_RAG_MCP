import fetch from "node-fetch";

const QDRANT_URL = process.env.QDRANT_URL || "http://localhost:6333";
const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const EMBED_MODEL = process.env.EMBED_MODEL || "nomic-embed-text";
const COLLECTION_PREFIX = process.env.COLLECTION_PREFIX || "rag_";

type Json = any;

async function readStdio(): Promise<void> {
  process.stdin.setEncoding("utf8");
  let buf = "";
  process.stdin.on("data", async (chunk) => {
    buf += chunk;
    // naive framing: assume one JSON per line
    let idx;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line) continue;
      try {
        const msg = JSON.parse(line);
        const res = await handle(msg);
        process.stdout.write(JSON.stringify(res) + "\n");
      } catch (e: any) {
        process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: null, error: { code: -32603, message: String(e) }}) + "\n");
      }
    }
  });
}

async function embed(input: string): Promise<number[]> {
  const r = await fetch(`${OLLAMA_URL}/api/embeddings`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: EMBED_MODEL, input: [input] })
  });
  if (!r.ok) throw new Error(`embed failed: ${r.status}`);
  const j = await r.json() as any;
  return j.embeddings[0];
}

async function qdrantSearch(collection: string, vec: number[], corpus: string, top = 8): Promise<any[]> {
  const r = await fetch(`${QDRANT_URL}/collections/${collection}/points/search`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      vector: vec, top,
      with_payload: true,
      filter: { must: [{ key: "corpus_id", match: { value: corpus } }] }
    })
  });
  if (!r.ok) throw new Error(`qdrant search failed: ${r.status}`);
  const j = await r.json() as any;
  return j.result || [];
}

async function handle(msg: Json): Promise<Json> {
  const { id, method, params } = msg;
  if (method === "rag.search") {
    const query = params?.query as string;
    const corpus = params?.corpus_id ?? "default";
    const top = params?.top_k ?? 8;
    const vec = await embed(query);
    const col = `${COLLECTION_PREFIX}${corpus}`;
    const hits = await qdrantSearch(col, vec, corpus, top);
    const items = hits.map((h: any) => {
      const p = h.payload || {};
      return {
        text: p.chunk_text, score: h.score,
        source_path: p.source_path, section: p.section,
        doc_id: p.doc_id, chunk_id: p.chunk_id
      };
    });
    return { jsonrpc: "2.0", id, result: { items } };
  }
  if (method === "rag.health") {
    return { jsonrpc: "2.0", id, result: { qdrant: QDRANT_URL, ollama: OLLAMA_URL, model: EMBED_MODEL } };
  }
  return { jsonrpc: "2.0", id, error: { code: -32601, message: "Method not found" } };
}

readStdio().catch(err => {
  console.error(err);
  process.exit(1);
});
