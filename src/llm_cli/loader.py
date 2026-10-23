"""加载层：把磁盘上的文件变成纯文本 + 来源信息。

只负责读，不负责判断内容是什么，也不负责报错给人看。
"""

from dataclasses import dataclass, field
from pathlib import Path
from pypdf import PdfReader



SUPPORTED_SUFFIXES = (".txt", ".pdf")


@dataclass
class Document:
    text: str
    meta: dict = field(default_factory=dict)


def load(path: str | Path) -> Document:
    p = Path(path)

    # 检查 1：文件存在且真的是文件（不是目录）
    if not p.is_file():
        raise FileNotFoundError(f"文件不存在或不是文件：{p}")

    # 检查 2：格式支持吗（转小写，这样 .PDF / .Txt 也能过）
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"暂不支持的文件类型：{p.suffix or '无扩展名'}，"
            f"当前支持 {'、'.join(SUPPORTED_SUFFIXES)}"
        )

    # 分发：两个分支都返回 (正文, 额外信息) 两元组，
    # 这样下面的组装代码只写一份，不用 if 两遍
    if suffix == ".pdf":
        text, extra = _read_pdf(p)
    else:
        text, encoding = _read_text(p)
        extra = {"encoding": encoding}

    extra.update({"path": str(p), "chars": len(text)})
    return Document(text=text, meta=extra)


def _read_text(p: Path) -> tuple[str, str]:
    for encoding in ("utf-8", "gbk"):
        try:
            return p.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法用 utf-8 或 gbk 解码，可能是二进制文件：{p}")

def _read_pdf(p: Path) -> tuple[str, dict]:
    """用 pypdf 逐页抽文字，返回 (全文, {"pages": 页数})。

    为什么要在这里主动判空：
        扫描件的每一页其实是一张图片，抽出来是空字符串。
        如果不拦，程序会拿着空文本去调模型，模型只能瞎编，
        然后你在"为什么输出这么离谱"上耗掉半小时。
        宁可在这里就停下，明确告诉用户"这个 PDF 没有文字层"。
    """
    reader = PdfReader(str(p))

    pages = len(reader.pages)

    # 逐页抽，拼起来。
    # extract_text() 在"这一页没有文字层"时返回 None，所以要用 or "" 兜住，
    # 否则 "\n".join() 会因为混进 None 而报 TypeError
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")

    text = "\n".join(parts)

    # 38 页抽出 3 万多字是正常的；抽出几十个字就说明它根本没有文字层
    if len(text) < 50:
        raise ValueError(
            f"PDF 里没有可提取的文字（{pages} 页只抽出 {len(text)} 个字符），"
            f"可能是扫描件或纯图片 PDF"
        )

    return text, {"pages": pages}
