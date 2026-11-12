"""分块策略对照实验：md（标题分块）/ 256 / 512 / 1024 四档召回率对照。

实验设计：
- 自变量：切分策略。md = 按 ## 条目切（一 chunk 一条目，W4 主线）；
  256/512/1024 = 字符分块，overlap 各取 chunk_size//10（比例一致，W3 基线）
- 因变量：top-3 召回率（命中标准出处条目号即算召回；跨文档题需双出处都命中）
- 尺子：data/eval/eval_v0.json（22 条，5 种题型）
- query 向量四档共用（同一 query 在四个 collection 里检索）
- chunk embedding 走落盘缓存（各档文本不同，缓存键自然区分）
- 所有 collection 统一声明 cosine 度量，否则分数没法横着比

用法：uv run python experiments/chunk_compare.py
"""
import json
import logging
from pathlib import Path

import chromadb

from llm_cli.config import load_settings
from llm_cli.kb.chunk import chunk_by_heading, chunk_document
from llm_cli.kb.embedder import Embedder

logging.basicConfig(level=logging.WARNING)

KB_DIR = Path("data/kb")
EVAL_FILE = Path("data/eval/eval_v0.json")
PERSIST_DIR = Path("data/chroma")
TOP_K = 3

# 四档：md 打头（本周主角），后三档是 W3 的字符分块基线
STRATEGIES = [
    {"label": "md", "kind": "heading", "size": 512, "overlap": 0},
    {"label": "256", "kind": "chars", "size": 256, "overlap": 25},
    {"label": "512", "kind": "chars", "size": 512, "overlap": 50},
    {"label": "1024", "kind": "chars", "size": 1024, "overlap": 100},
]


def load_corpus() -> dict[str, str]:
    return {p.name: p.read_text("utf-8") for p in sorted(KB_DIR.glob("*.md"))}


def is_hit(q: dict, sources: list[str]) -> bool:
    """召回判定：top-K 的 source 命中 expected 即算召回。

    跨文档题额外要求同时命中 also_expected（两条来自不同文档，缺一不算）——
    这类题的靶子是 W5「必须跨 2 个文档才能答」的验收。
    """
    if not any(q["expected"] in s for s in sources):
        return False
    also = q.get("also_expected")
    if also and not any(also in s for s in sources):
        return False
    return True


def chunk_corpus(corpus: dict[str, str], st: dict) -> list[dict]:
    """按策略切分全部语料。"""
    chunks: list[dict] = []
    for source, text in corpus.items():
        if st["kind"] == "heading":
            chunks.extend(chunk_by_heading(text, source, max_chars=st["size"]))
        else:
            chunks.extend(chunk_document(text, source, st["size"], st["overlap"]))
    return chunks


def run_one(st: dict, corpus, embedder, client, query_vecs):
    """跑一档：切分 → 建独立 collection → 检索全部评测题 → 返回明细。"""
    all_chunks = chunk_corpus(corpus, st)
    vectors = embedder.embed_texts([c["text"] for c in all_chunks])

    coll_name = f"kb_{st['label']}"
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    coll = client.create_collection(
        coll_name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"chunk_mode": st["kind"], "max_chars": st["size"]},
    )
    coll.add(
        ids=[str(i) for i in range(len(all_chunks))],
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"]} for c in all_chunks],
        embeddings=vectors,
    )

    details = []
    for q, qvec in zip(eval_data["queries"], query_vecs):
        res = coll.query(query_embeddings=[qvec], n_results=TOP_K)
        sources = [m["source"] for m in res["metadatas"][0]]
        details.append({
            "id": q["id"],
            "query": q["query"],
            "expected": q["expected"],
            "also_expected": q.get("also_expected", ""),
            "type": q["type"],
            "hit": is_hit(q, sources),
            "top1": sources[0] if sources else "",
            "top3": sources,
        })
    return len(all_chunks), details


eval_data = json.loads(EVAL_FILE.read_text("utf-8"))


def main():
    settings = load_settings()
    corpus = load_corpus()
    embedder = Embedder(settings)
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))

    queries = eval_data["queries"]
    n_q = len(queries)
    print(f"向量化 {n_q} 条 query（四档共用）...")
    query_vecs = embedder.embed_texts([q["query"] for q in queries])

    labels = [st["label"] for st in STRATEGIES]
    print(f"\n{'档位':>6} | {'kind':>8} | {'chunks':>6} | {'召回':>4}/{n_q} | {'召回率':>6}")
    print("-" * 52)

    all_results = {}
    for st in STRATEGIES:
        n_chunks, details = run_one(st, corpus, embedder, client, query_vecs)
        hits = sum(1 for d in details if d["hit"])
        recall = hits / n_q
        all_results[st["label"]] = {
            "chunks": n_chunks, "hits": hits, "recall": recall, "details": details,
        }
        print(f"{st['label']:>6} | {st['kind']:>8} | {n_chunks:>6} | {hits:>4}/{n_q} | {recall:>5.0%}")

    # 分题型召回率
    types = ["直答", "措辞陷阱", "跨条目", "跨文档", "否定边界"]
    header = f"\n{'题型':<8}" + "".join(f" | {lb:>6}" for lb in labels)
    print(header)
    print("-" * 50)
    for t in types:
        row = f"{t:<8}"
        for lb in labels:
            td = [d for d in all_results[lb]["details"] if d["type"] == t]
            th = sum(1 for d in td if d["hit"])
            row += f" | {th}/{len(td):<4}"
        print(row)

    print("\n=== 失败题（四档都没召回的）===")
    for q in queries:
        if all(not all_results[lb]["details"][q["id"] - 1]["hit"] for lb in labels):
            exp = q["expected"] + ("+" + q["also_expected"] if q.get("also_expected") else "")
            print(f"  #{q['id']} [{q['type']}] {q['query']} → 期望 {exp}，四档全败")

    print("\n=== md 档独有失败/独有命中（与 256 档对比）===")
    for q in queries:
        d_md = all_results["md"]["details"][q["id"] - 1]
        d_256 = all_results["256"]["details"][q["id"] - 1]
        if d_md["hit"] and not d_256["hit"]:
            print(f"  md 救回 #{q['id']} [{q['type']}] {q['query']}（256 档 top1={d_256['top1']}）")
        elif d_256["hit"] and not d_md["hit"]:
            print(f"  md 丢掉 #{q['id']} [{q['type']}] {q['query']}（md 档 top1={d_md['top1']}）")


if __name__ == "__main__":
    main()
