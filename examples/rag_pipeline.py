"""一条命令跑完 RAG 六段，并把每一段的中间产物打印出来。

这是「手搓 RAG」的全链路视图：不是为了功能，是为了让你能一段一段看清数据长什么样。
每一段都调用 src/llm_cli/kb/ 里的真实实现，不复制逻辑（避免两份代码走偏）。

用法：
    uv run python examples/rag_pipeline.py                    # 用默认问题
    uv run python examples/rag_pipeline.py "退款多久能到账"     # 自定义问题
    uv run python examples/rag_pipeline.py --no-build         # 跳过建库（库已存在时更快）
"""
import logging
import sys
from pathlib import Path

from llm_cli.config import load_settings
from llm_cli.kb.ask import ask
from llm_cli.kb.chunk import chunk_by_heading
from llm_cli.kb.rerank import rerank
from llm_cli.kb.search import search
from llm_cli.kb.store import build_store, get_collection

logging.basicConfig(level=logging.WARNING)  # 屏蔽 httpx 的逐条请求日志，只留关键信息

KB_DIR = Path("data/kb")
DEFAULT_QUERY = "耳机拆封了还能退货吗"


def section(idx: int, title: str) -> None:
    """统一的分段标题，方便肉眼对照六段链路。"""
    print(f"\n{'=' * 64}\n第 {idx} 段 · {title}\n{'=' * 64}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    query = args[0] if args else DEFAULT_QUERY
    do_build = "--no-build" not in sys.argv
    settings = load_settings()

    # ---------- 第 1 段：加载 ----------
    section(1, "加载（读 Markdown 源文件）")
    corpus = {p.name: p.read_text("utf-8") for p in sorted(KB_DIR.glob("*.md"))}
    for name, text in corpus.items():
        print(f"  {name:<12} {len(text):>5} 字符")

    # ---------- 第 2 段：切分 ----------
    section(2, "切分（一 chunk = 一个 ## 条目）")
    for name, text in corpus.items():
        chunks = chunk_by_heading(text, name)
        print(f"  {name:<12} -> {len(chunks)} 块  {[c['source'].split('#')[-1] for c in chunks]}")
    sample = chunk_by_heading(corpus["退款政策.md"], "退款政策.md")[1]
    print(f"\n  样例块 text 前 60 字：{sample['text'][:60]}…")
    print(f"  样例块 source        ：{sample['source']}")

    # ---------- 第 3 段：向量化 + 第 4 段：存储 ----------
    section(3, "向量化 + 入库（1024 维，缓存命中则零 API 调用）")
    if do_build:
        n = build_store(corpus, settings)
    else:
        n = get_collection().count()
    print(f"  库内 chunk 总数：{n}")
    print("  度量：", get_collection().configuration["hnsw"]["space"])

    # ---------- 第 5 段（上）：粗召回 ----------
    section(4, "检索（向量余弦，粗召回 top-20）")
    candidates = search(query, k=20, settings=settings)
    print(f"  问题：{query}")
    print(f"  召回 {len(candidates)} 条，前 5：")
    for i, c in enumerate(candidates[:5], 1):
        print(f"    {i}. {c['score']:.4f}  {c['source']}")

    # ---------- 第 5 段（下）：精排 ----------
    section(5, "精排（交叉编码器 rerank，取 top-3）")
    reranked = rerank(query, candidates, top_k=3, settings=settings)
    print(f"    {'':<4}{'向量排序':<34}{'rerank 后':<34}")
    for i in range(3):
        base = f"{candidates[i]['source']} ({candidates[i]['score']:.4f})"
        new = f"{reranked[i]['source']} ({reranked[i]['score']:.4f})"
        print(f"    {i + 1}.  {base:<32}{new:<34}")

    # ---------- 第 6 段：生成 + 引用 ----------
    section(6, "生成（拼 prompt → qwen3.8-flash → 答案 + 依据）")
    result = ask(query, settings=settings)
    print("  答案：")
    for line in result["answer"].splitlines():
        print(f"    {line}")
    print(f"\n  模型标注的依据：{result['cited']}")
    print(f"  其中真实存在于资料里的：{result['valid']}")
    print(f"  引用格式是否合法：{result['format_ok']}")


if __name__ == "__main__":
    main()
