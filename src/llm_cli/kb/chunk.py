"""文本切分：长文档 -> 带 source 的 chunk 列表。

第一版策略：段落感知的字符分块（递归字符分块简化版）。
- 按 \\n\\n 切段落
- 累积到接近 chunk_size 输出，相邻块用 overlap 保持连续
- 条目号从 ## 标题提取，塞进 source 字段（引用溯源的物理载体）

参考 all-in-rag ch2/05。W2 复盘承诺由用户手写第一版，实际由 AI 带做（记入复盘）。
"""
import re

# 匹配条目标题：## 退款-03 / ## 物流-01 / ## 保修-02 等
_SECTION_RE = re.compile(r"^##\s+(\S+-\d+)")


def _extract_section(line: str, fallback: str) -> str:
    """从一行里提条目号（如 退款-03），提不到就沿用 fallback。"""
    m = _SECTION_RE.match(line.strip())
    return m.group(1) if m else fallback


def chunk_document(text: str, source: str, chunk_size: int = 512, overlap: int = 50) -> list[dict]:
    """切分单个文档。

    Args:
        text: 文档纯文本
        source: 来源标识（文件名）
        chunk_size: 每块目标大小（字符数）
        overlap: 相邻块重叠字符数

    Returns:
        list[dict]: 每项 {"text": ..., "source": "文件名#条目号"}
    """
    if not text:
        return []

    paragraphs = text.split("\n\n")
    chunks: list[dict] = []
    current = ""
    current_section = ""

    for para in paragraphs:
        if not para.strip():
            continue
        # 段落首行可能带条目标题，更新当前条目号
        first_line = para.split("\n", 1)[0]
        current_section = _extract_section(first_line, current_section)

        candidate = para if not current else current + "\n\n" + para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # 当前块满了，先输出
            if current:
                chunks.append({"text": current, "source": f"{source}#{current_section}"})
            # 新块 = 上一块末尾 overlap 字符 + 当前段（保持连续性）
            tail = current[-overlap:] if overlap > 0 and len(current) > overlap else ""
            current = (tail + "\n\n" + para) if tail else para

    # 收尾：最后剩的不满一块也要输出
    if current:
        chunks.append({"text": current, "source": f"{source}#{current_section}"})

    return chunks
