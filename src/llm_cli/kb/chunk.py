"""文本切分：长文档 -> 带 source 的 chunk 列表。

第一版策略：段落感知的字符分块（递归字符分块简化版）。
- 按 \\n\\n 切段落
- 累积到接近 chunk_size 输出，相邻块用 overlap 保持连续
- 条目号从 ## 标题提取，塞进 source 字段（引用溯源的物理载体）

参考 all-in-rag ch2/05。W2 复盘承诺由用户手写第一版，实际由 AI 带做（记入复盘）。
"""

# 导入正则表达式模块，用于从标题行中提取条目标识
import re


# 匹配条目标题：## 退款-03 / ## 物流-01 / ## 保修-02 等
# 说明：
# ^##\s+     表示行首是 "##" 加至少一个空白字符
# (\S+-\d+)  捕获一个非空白片段 + "-" + 数字，例如 "退款-03"
_SECTION_RE = re.compile(r"^##\s+(\S+-\d+)")


def _extract_section(line: str, fallback: str) -> str:
    """从一行里提条目号（如 退款-03），提不到就沿用 fallback。"""

    # 去掉行首行尾空白后，用正则匹配条目标题
    m = _SECTION_RE.match(line.strip())

    # 如果匹配成功，返回捕获到的条目号；否则返回传入的 fallback
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

    # 空文本直接返回空列表，避免后续无意义处理
    if not text:
        return []

    # 按连续两个换行符切分段落，即把 "\n\n" 视为段落边界
    paragraphs = text.split("\n\n")

    # 存放最终切分结果的列表，每个元素是 {"text": ..., "source": ...}
    chunks: list[dict] = []

    # 当前正在累积的块内容
    current = ""

    # 当前块所属的条目号，例如 "退款-03"
    current_section = ""

    # 逐段处理文档内容
    for para in paragraphs:

        # 跳过空段落或只包含空白字符的段落
        if not para.strip():
            continue

        # 取当前段落的第一行，用于判断是否带有 "## 条目号" 标题
        first_line = para.split("\n", 1)[0]

        # 从第一行提取条目号；如果提不到，就沿用之前的 current_section
        current_section = _extract_section(first_line, current_section)

        # 如果当前块为空，候选内容就是当前段落；
        # 否则把当前段落追加到当前块后面，中间用两个换行分隔
        candidate = para if not current else current + "\n\n" + para

        # 如果加入当前段落后仍未超过 chunk_size，则继续累积
        if len(candidate) <= chunk_size:
            current = candidate

        # 如果加入当前段落后会超过 chunk_size，说明当前块已满
        else:
            # 先输出当前已经累积的块（如果非空）
            if current:
                chunks.append({"text": current, "source": f"{source}#{current_section}"})

            # 计算新块开头需要保留的 overlap 尾部内容
            # 只有 overlap > 0 且当前块长度大于 overlap 时才保留尾部
            tail = current[-overlap:] if overlap > 0 and len(current) > overlap else ""

            # 新块 = 上一块末尾 overlap 字符 + 当前段落
            # 如果 tail 为空，则新块直接等于当前段落
            current = (tail + "\n\n" + para) if tail else para

    # 收尾：最后剩的不满一块也要输出
    if current:
        chunks.append({"text": current, "source": f"{source}#{current_section}"})

    # 返回所有切分好的 chunk
    return chunks