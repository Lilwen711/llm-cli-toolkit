"""W3 分块对照实验：chunk_size 256/512/1024 三档召回率对照。

实验设计：
- 自变量：chunk_size（256/512/1024），overlap 各取 chunk_size//10（比例一致）
- 因变量：top-3 召回率（命中标准出处条目号即算召回）
- 尺子：data/eval/eval_v0.json（20 条，5 种题型）
- query 向量三档共用（同一 query 在三个 collection 里检索）
- chunk embedding 走落盘缓存（三档文本不同，缓存键自然区分）

用法：uv run python experiments/chunk_compare.py
"""
import json
import logging
from pathlib import Path

import chromadb

from llm_cli.config import load_settings
from llm_cli.kb.chunk import chunk_document
from llm_cli.kb.embedder import Embedder

logging.basicConfig(level=logging.WARNING)

KB_DIR = Path("data/kb")
EVAL_FILE = Path("data/eval/eval_v0.json")
PERSIST_DIR = Path("data/chroma")
TOP_K = 3
CONFIGS = [(256, 25), (512, 50), (1024, 100)]  # (chunk_size, overlap)


def load_corpus() -> dict[str, str]:
    return {p.name: p.read_text("utf-8") for p in sorted(KB_DIR.glob("*.md"))}


def run_one(chunk_size: int, overlap: int, corpus, embedder, client, query_vecs):
    """跑一档：切分 → 建独立 collection → 检索 20 条 → 返回明细。"""
    all_chunks = []
    for source, text in corpus.items():
        all_chunks.extend(chunk_document(text, source, chunk_size, overlap))

    vectors = embedder.embed_texts([c["text"] for c in all_chunks])

    coll_name = f"kb_{chunk_size}"
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    coll = client.create_collection(
        coll_name, configuration={"hnsw": {"space": "cosine"}}
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
        hit = any(q["expected"] in s for s in sources)
        details.append({
            "id": q["id"],
            "query": q["query"],
            "expected": q["expected"],
            "type": q["type"],
            "hit": hit,
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

    # query 向量三档共用：20 条一批
    queries = eval_data["queries"]
    query_texts = [q["query"] for q in queries]
    print("向量化 20 条 query（三档共用）...")
    query_vecs = embedder.embed_texts(query_texts)

    print(f"\n{'chunk_size':>10} | {'overlap':>7} | {'chunks':>6} | {'召回':>4}/20 | {'召回率':>6}")
    print("-" * 55)

    all_results = {}
    for chunk_size, overlap in CONFIGS:
        n_chunks, details = run_one(chunk_size, overlap, corpus, embedder, client, query_vecs)
        hits = sum(1 for d in details if d["hit"])
        recall = hits / len(details)
        all_results[chunk_size] = {"chunks": n_chunks, "hits": hits, "recall": recall, "details": details}
        print(f"{chunk_size:>10} | {overlap:>7} | {n_chunks:>6} | {hits:>4}/20 | {recall:>5.0%}")

    # 分题型召回率
    types = ["直答", "措辞陷阱", "跨条目", "跨文档", "否定边界"]
    print(f"\n{'题型':<10} | {'256':>6} | {'512':>6} | {'1024':>6}")
    print("-" * 42)
    for t in types:
        row = f"{t:<10}"
        for cs, _,  in CONFIGS:
            td = [d for d in all_results[cs]["details"] if d["type"] == t]
            th = sum(1 for d in td if d["hit"])
            row += f" | {th}/{len(td):<4}"
        print(row)

    # 逐题明细（只打印失败的）
    print("\n=== 失败题（三档都没召回的）===")
    for q in queries:
        fails = [cs for cs, _,  in CONFIGS if not all_results[cs]["details"][q["id"] - 1]["hit"]]
        if len(fails) == len(CONFIGS):
            print(f"  #{q['id']} [{q['type']}] {q['query']} → 期望 {q['expected']}，三档全败")

    print("\n=== 各档独有失败（其他档命中、这档失败）===")
    for cs, _,  in CONFIGS:
        others = [c for c, _,  in CONFIGS if c != cs]
        for q in queries:
            d = all_results[cs]["details"][q["id"] - 1]
            if not d["hit"] and all(all_results[o]["details"][q["id"] - 1]["hit"] for o in others):
                print(f"  chunk={cs} #{q['id']} [{q['type']}] {q['query']} → 期望 {q['expected']}，top1={d['top1']}")


if __name__ == "__main__":
    main()
