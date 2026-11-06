"""W3 验证脚本：建库 + 3 个真实问题检索 + 缓存二次零调用验证。

用法：uv run python scripts/verify_kb.py
"""
import logging
from pathlib import Path

from llm_cli.config import load_settings
from llm_cli.kb import build_store, search

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

KB_DIR = Path("data/kb")


def load_corpus() -> dict[str, str]:
    """读 data/kb/ 下所有 .md，返回 {文件名: 文本}。"""
    corpus = {}
    for p in sorted(KB_DIR.glob("*.md")):
        corpus[p.name] = p.read_text("utf-8")
    return corpus


def main() -> None:
    settings = load_settings()
    corpus = load_corpus()
    print(f"\n=== 语料：{len(corpus)} 篇 ===")
    for name, text in corpus.items():
        print(f"  {name}: {len(text)} 字符")

    print("\n=== 第一次建库（调 embedding）===")
    total = build_store(corpus, settings)
    print(f"库共 {total} chunk")

    print("\n=== 第二次建库（应零 embedding 调用）===")
    total2 = build_store(corpus, settings)
    print(f"库共 {total2} chunk（应与上次相同）")

    queries = [
        "耳机拆封了还能退货吗",
        "快递丢了怎么索赔",
        "过保修期了还能修吗",
    ]
    for q in queries:
        print(f"\n=== 检索：{q} ===")
        results = search(q, k=3, settings=settings)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] score={r['score']}  source={r['source']}")
            print(f"      text: {r['text'][:80].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
