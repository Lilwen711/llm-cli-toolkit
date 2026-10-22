"""加载层：把磁盘上的文件变成纯文本 + 来源信息。

只负责读，不负责判断内容是什么，也不负责报错给人看。
"""

from dataclasses import dataclass, field
from pathlib import Path


SUPPORTED_SUFFIXES = (".txt",)


@dataclass
class Document:
    text: str
    meta: dict = field(default_factory=dict)


def load(path: str | Path) -> Document:
    p = Path(path)

    if not p.is_file():
        raise FileNotFoundError(f"文件不存在或不是文件：{p}")

    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"暂不支持的文件类型：{p.suffix or '无扩展名'}，"
            f"当前支持 {'、'.join(SUPPORTED_SUFFIXES)}"
        )

    text, encoding = _read_text(p)

    return Document(
        text=text,
        meta={
            "path": str(p),
            "chars": len(text),
            "encoding": encoding,
        },
    )


def _read_text(p: Path) -> tuple[str, str]:
    for encoding in ("utf-8", "gbk"):
        try:
            return p.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法用 utf-8 或 gbk 解码，可能是二进制文件：{p}")
