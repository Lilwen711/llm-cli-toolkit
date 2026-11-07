"""检索入口：query -> top-K [{text, source, score}]。

本周(W3)验收点：给定 query，返回 3 段带出处的相关文本。
source 含条目号，是总计划「引用溯源是防幻觉第一道闸」的物理实现。
score = 1 - cosine_distance（Chroma 默认返回距离，转成相似度）。
"""

# 导入标准库 logging，用于记录本模块的日志
import logging

# 从项目配置模块导入 Settings 类型和配置加载函数
from llm_cli.config import Settings, load_settings

# 从同包内导入 Embedder，用于把文本编码成向量
from .embedder import Embedder

# 从同包内导入 get_collection，用于获取 Chroma 的 collection 句柄
from .store import get_collection

# 创建模块级 logger，日志中会带上本模块的名字，便于定位来源
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

    # 如果调用方没有传入 settings，则从 .env 等配置源惰性加载
    # 这样测试时可以注入 mock 配置，避免依赖真实环境
    if settings is None:
        settings = load_settings()

    # 用配置初始化 Embedder（需与建库时使用同一个 embedding 模型）
    embedder = Embedder(settings)

    # 把用户 query 编码成查询向量
    qvec = embedder.embed_query(query)

    # 获取 Chroma collection 句柄
    coll = get_collection()

    # 执行向量检索：
    # - query_embeddings 是二维列表（外层是 batch 维度），此处只查一个 query
    # - n_results=k 表示取 top-K
    res = coll.query(query_embeddings=[qvec], n_results=k)

    # 用于存放最终返回结果的列表
    out = []

    # 遍历第一个（也是唯一一个）query 的文档、metadata、距离
    # res 的结构是按 query 分组的嵌套列表，所以取 [0]
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        # Chroma 默认 cosine distance ∈ [0,2]，相似度 = 1 - distance
        # 用 max(0.0, ...) 把可能的负值截断，保证对外 score 恒在 [0,1]
        score = max(0.0, 1.0 - dist)

        # 组装单条结果：
        # - text: chunk 原文
        # - source: 来源标识，含条目号，用于引用溯源；缺失时兜底为空字符串
        # - score: 相似度，保留 4 位小数便于展示
        out.append({
            "text": doc,
            "source": meta.get("source", ""),
            "score": round(score, 4),
        })

    # 返回 top-K 结果列表
    return out