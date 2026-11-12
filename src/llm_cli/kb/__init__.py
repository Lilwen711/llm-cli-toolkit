"""kb 知识库子包：RAG 检索能力。

子模块：
  chunk    - 文本切分（Markdown 标题分块，一 chunk 一条目；chars 分块留作对照）
  embedder - 向量化（qwen3.7-text-embedding-flash，落盘缓存）
  store    - 向量存储（Chroma 本地持久化，cosine 度量 + 配置指纹）
  search   - 检索入口（query -> top-K [{text, source, score}]）
  rerank   - 精排（qwen3.7-text-rerank，交叉编码器，top-N -> top-K）

W4：检索 + rerank 完成，生成段（ask）下一步。
"""
from .rerank import rerank
from .search import search
from .store import build_store

__all__ = ["search", "rerank", "build_store"]
