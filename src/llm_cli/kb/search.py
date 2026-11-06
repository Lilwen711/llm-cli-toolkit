"""检索入口：query -> top-K [{text, source, score}]。

本周(W3)验收点：给定 query，返回 3 段带出处的相关文本。
source 含条目号，是总计划「引用溯源是防幻觉第一道闸」的物理实现。
score = 1 - cosine_distance（Chroma 默认返回距离，转成相似度）。
"""
import logging

from llm_cli.config import Settings, load_settings

from .embedder import Embedder
from .store import get_collection

logger = logging.getLogger(__name__)


def search(query: str, k: int = 3, settings: Settings | None = None) -> list[dict]:
    """检索 top-K 相关 chunk。

    Args:
        query: 用户问题
        k: 返回条数，默认 3
        settings: 可选，不传则从 .env 加载

    Returns:
        list[dict]: [{"text": ..., "source": "退款政策.md#退款-03", "score": 0.82}, ...]
    """
    if settings is None:
        settings = load_settings()

    embedder = Embedder(settings)
    qvec = embedder.embed_query(query)
    coll = get_collection()
    res = coll.query(query_embeddings=[qvec], n_results=k)

    out = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        # Chroma 默认 cosine distance ∈ [0,2]，相似度 = 1 - distance
        score = max(0.0, 1.0 - dist)
        out.append({
            "text": doc,
            "source": meta.get("source", ""),
            "score": round(score, 4),
        })
    return out
