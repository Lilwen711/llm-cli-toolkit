"""rerank 增益实验：22 条评测集，检索 top20 -> rerank -> top3，对比「直接取 top3」。

三个指标：
- top1 命中率：正解排第一的比例（md 分块后 top3 召回已 100%，增益只能从 top1 上看）
- top3 召回：正解出现在前 3 的比例（跨文档题需双出处都出现）
- 正解平均排名：越小越好，最能反映「排序质量」而不只是「有没有进前 3」

用法：uv run python experiments/rerank_compare.py
"""
import json
import logging
from pathlib import Path

from llm_cli.config import load_settings
from llm_cli.kb.embedder import Embedder
from llm_cli.kb.rerank import rerank
from llm_cli.kb.store import get_collection

logging.basicConfig(level=logging.WARNING)

EVAL_FILE = Path("data/eval/eval_v0.json")
TOP_N = 20  # 粗召回条数
TOP_K = 3   # 精排后保留条数

eval_data = json.loads(EVAL_FILE.read_text("utf-8"))
queries = eval_data["queries"]


def is_hit(q: dict, sources: list[str]) -> bool:
    """top-K 命中判据：expected 必须出现；跨文档题还需 also_expected 同时出现。"""
    if not any(q["expected"] in s for s in sources):
        return False
    also = q.get("also_expected")
    if also and not any(also in s for s in sources):
        return False
    return True


def rank_of(q: dict, sources: list[str]) -> int:
    """正解在列表里的名次（1 起）；没进榜返回 0。跨文档题取两条里更靠后的那个。"""
    ranks = []
    for key in (q["expected"], q.get("also_expected")):
        if not key:
            continue
        hit = [i + 1 for i, s in enumerate(sources) if key in s]
        ranks.append(hit[0] if hit else 0)
    if not ranks or 0 in ranks:
        return 0
    return max(ranks)


def main():
    settings = load_settings()
    embedder = Embedder(settings)
    coll = get_collection()

    print(f"向量化 {len(queries)} 条 query ...")
    qvecs = embedder.embed_texts([q["query"] for q in queries])

    rows = []
    for q, qv in zip(queries, qvecs):
        res = coll.query(
            query_embeddings=[qv], n_results=TOP_N,
            include=["metadatas", "distances", "documents"],
        )
        candidates = [
            {"text": d, "source": m["source"], "score": round(1 - dist, 4)}
            for d, m, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]
        base = candidates[:TOP_K]
        ranked = rerank(q["query"], candidates, top_k=TOP_K)

        base_src = [c["source"] for c in base]
        rr_src = [c["source"] for c in ranked]
        rows.append({
            "id": q["id"], "type": q["type"], "query": q["query"],
            "exp": q["expected"] + ("+" + q["also_expected"] if q.get("also_expected") else ""),
            "base_hit1": bool(base_src) and q["expected"] in base_src[0],
            "rr_hit1": bool(rr_src) and q["expected"] in rr_src[0],
            "base_hit3": is_hit(q, base_src),
            "rr_hit3": is_hit(q, rr_src),
            "base_rank": rank_of(q, base_src),
            "rr_rank": rank_of(q, rr_src),
            "base_src": base_src,
            "rr_src": rr_src,
        })

    n = len(rows)
    # 双出处题（有 also_expected）没有唯一「第一」，top1 指标只统计单出处题
    singles = [r for r in rows if "+" not in r["exp"]]
    n1 = len(singles)

    def pct(k):
        return sum(1 for r in rows if r[k]) / n

    def pct1(k):
        return sum(1 for r in singles if r[k]) / n1

    def mean_rank(k):
        vals = [r[k] for r in rows if r[k] > 0]
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\n{'指标':<16} | {'直接 top3':>10} | {'rerank 后':>10}")
    print("-" * 44)
    print(f"{'top1 命中率':<16} | {pct1('base_hit1'):>9.0%} | {pct1('rr_hit1'):>10.0%}   ({n1} 条单出处题)")
    print(f"{'top3 召回率':<16} | {pct('base_hit3'):>9.0%} | {pct('rr_hit3'):>10.0%}   ({n} 条全量)")
    print(f"{'正解平均排名':<16} | {mean_rank('base_rank'):>10.2f} | {mean_rank('rr_rank'):>10.2f}   (只统计进榜的)")

    print("\n=== top1 发生变化的题 ===")
    changed = [r for r in rows if r["base_hit1"] != r["rr_hit1"]]
    if not changed:
        print("  （无变化）")
    for r in changed:
        arrow = "修好" if r["rr_hit1"] else "修坏"
        print(f"  #{r['id']} [{r['type']}] {r['query']}")
        print(f"     {arrow}：base top1={r['base_src'][0]} -> rerank top1={r['rr_src'][0]}")

    print("\n=== 单出处题里仍然错在 top1 的（rerank 没救回来）===")
    for r in rows:
        if "+" not in r["exp"] and not r["rr_hit1"]:
            print(f"  #{r['id']} [{r['type']}] {r['query']}")
            print(f"     期望 {r['exp']}；rerank top3 = {r['rr_src']}")


if __name__ == "__main__":
    main()
