"""kb 知识库子包：RAG 检索能力。

子模块：
  chunk    - 文本切分（段落感知字符分块 + 条目号提取）
  embedder - 向量化（qwen3.7-text-embedding-flash，落盘缓存）
  store    - 向量存储（Chroma 本地文件持久化）
  search   - 检索入口（query -> top-K [{text, source, score}]）

本周(W3)只做检索，接生成是 W4。
"""
from .search import search
from .store import build_store

__all__ = ["search", "build_store"]
