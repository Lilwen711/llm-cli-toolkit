"""端到端评测：22 条评测集跑完整 RAG（检索 -> rerank -> 生成 -> 引用）。

三个可自动统计的数字：
  1. 检索命中率：喂给模型的 top3 里有没有正解条目（跨文档题需两条都在）
  2. 引用格式合法率：回答末尾有「依据：」行、且条目号都在候选资料里（没编造）
  3. 引用命中率：合法引用的条目号里有没有正解（这是「答案是否用对了资料」的代理指标）

注意：答案本身对不对需要人工勾（见结果 JSON 里的 manual_ok 字段），脚本不猜。

跑一次存一次：结果落盘 data/eval/e2e_<时间戳>.json，不重复跑烧额度。
用法：uv run python experiments/e2e_eval.py
"""
import json
import logging
from datetime import datetime
from pathlib import Path

from llm_cli.config import load_settings
from llm_cli.kb.ask import ask

logging.basicConfig(level=logging.WARNING)

EVAL_FILE = Path("data/eval/eval_v0.json")
OUT_DIR = Path("data/eval")

eval_data = json.loads(EVAL_FILE.read_text("utf-8"))
queries = eval_data["queries"]


def sources_have(q: dict, sources: list[str]) -> bool:
    """检索命中判据：expected 出现；跨文档题还要 also_expected 同时出现。"""
    if not any(q["expected"] in s for s in sources):
        return False
    also = q.get("also_expected")
    if also and not any(also in s for s in sources):
        return False
    return True


def cites_have(q: dict, valid: list[str]) -> bool:
    """引用命中判据：按题目类型分两种口径。

    - 单出处题：引用里出现 expected 即算命中
    - 跨文档题：两条出处都要被引用（否则等于只用了一半资料）
    """
    if q["expected"] not in valid:
        return False
    also = q.get("also_expected")
    if also and also not in valid:
        return False
    return True


def main():
    settings = load_settings()
    rows = []

    for q in queries:
        r = ask(q["query"], settings=settings)
        srcs = [s["source"] for s in r["sources"]]
        row = {
            "id": q["id"],
            "type": q["type"],
            "query": q["query"],
            "expected": q["expected"] + ("+" + q["also_expected"] if q.get("also_expected") else ""),
            "retrieved": srcs,
            "retrieval_hit": sources_have(q, srcs),
            "answer": r["answer"],
            "cited": r["cited"],
            "valid_cites": r["valid"],
            "format_ok": r["format_ok"],
            "cite_hit": cites_have(q, r["valid"]),
            "refused": "资料里没有" in r["answer"],
            "manual_ok": None,  # 人工勾：True / False，跑完自己填
        }
        rows.append(row)
        flag = "OK " if row["retrieval_hit"] and row["format_ok"] else "!! "
        print(f"{flag}#{row['id']:<3} {row['query'][:20]:<22} 引用={','.join(row['valid_cites']) or '无':<16} 期望={row['expected']}")

    n = len(rows)
    retrieval = sum(r["retrieval_hit"] for r in rows) / n
    fmt = sum(r["format_ok"] for r in rows) / n
    cite = sum(r["cite_hit"] for r in rows) / n
    refused = sum(r["refused"] for r in rows)

    print("\n" + "=" * 52)
    print(f"检索命中率    {retrieval:>6.0%}   ({sum(r['retrieval_hit'] for r in rows)}/{n})")
    print(f"引用格式合法率 {fmt:>6.0%}   ({sum(r['format_ok'] for r in rows)}/{n})")
    print(f"引用命中率    {cite:>6.0%}   ({sum(r['cite_hit'] for r in rows)}/{n})")
    print(f"拒答（资料里没有）{refused} 条")
    print("=" * 52)

    if fmt < 1:
        print("\n格式不合法/拒答的题：")
        for r in rows:
            if not r["format_ok"]:
                print(f"  #{r['id']} {r['query']} -> 引用={r['cited']} 合法={r['valid_cites']}")
                print(f"     答案：{r['answer'][:80]}")

    if cite < 1:
        print("\n引用没命中正解的题（答案可能对，但引错条目 = 溯源失效）：")
        for r in rows:
            if not r["cite_hit"]:
                print(f"  #{r['id']} {r['query']} -> 期望 {r['expected']}，实际引用 {r['valid_cites']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"e2e_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_file.write_text(
        json.dumps({
            "time": datetime.now().isoformat(timespec="seconds"),
            "eval_version": eval_data["version"],
            "metrics": {"retrieval": retrieval, "format_ok": fmt, "cite": cite},
            "rows": rows,
        }, ensure_ascii=False, indent=2),
        "utf-8",
    )
    print(f"\n结果已落盘：{out_file}")


if __name__ == "__main__":
    main()
