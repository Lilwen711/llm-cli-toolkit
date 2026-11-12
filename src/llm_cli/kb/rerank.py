"""重排：向量粗召回 top-N -> 交叉编码器精排 -> top-K。

为什么要 rerank：
  向量检索是「双塔」——query 和 document 各自编码成向量后再算距离，两者从没见过面，
  只能比「方向像不像」。rerank 是「交叉编码器」——把 query 和 document 拼成一段喂给
  模型，模型能看见两边的词如何互相作用，分数更准。但每次只能算一对，20 个候选就是
  20 次推理，贵且慢。所以标准用法是：向量检索粗召回 top-N，rerank 只在 N 上精排取 top-K。

契约：入参 candidates 就是 search() 的返回值（[{"text","source","score"}]），
返回结构完全一致，这样 ask() 里可以无缝替换 search 的结果。

score 字段存 rerank 分（relevance_score），保证「排序」和「分数」自洽；
原向量分保留在 vec_score 里，便于对比两个阶段各自给了多少分——调试时非常有用。
"""
import logging

import httpx

from llm_cli.config import Settings, load_settings

logger = logging.getLogger(__name__)

RERANK_MODEL = "qwen3.7-text-rerank"
RERANK_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"


def _service_host(base_url: str) -> str:
    """从 OpenAI 兼容 base_url 剥出服务根域名。

    https://ws-xxx.maas.aliyuncs.com/compatible-mode/v1 -> https://ws-xxx.maas.aliyuncs.com
    rerank 走的是 DashScope 原生协议，不在 /compatible-mode 下，所以要单独拼路径。
    """
    return base_url.split("/compatible-mode")[0].rstrip("/")


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 3,
    settings: Settings | None = None,
) -> list[dict]:
    """对粗召回结果精排。

    Args:
        query: 用户问题
        candidates: search() 返回的 [{"text", "source", "score"}, ...]，顺序无所谓
        top_k: 精排后保留几条
        settings: 可选，不传则从 .env 加载

    Returns:
        list[dict]: 按 rerank 分数降序的 top_k 条
                    [{"text", "source", "score": rerank分, "vec_score": 原向量分}, ...]

    Raises:
        httpx.HTTPStatusError: 接口非 2xx（先抛，不静默降级）
        httpx.TimeoutException: 超时
    """
    # 空候选直接返回，不打网络请求（省一次调用，也避免接口对空 documents 报错）
    if not candidates:
        return []

    if settings is None:
        settings = load_settings()

    # rerank 接口只吃文本列表，来源/原分数要自己按下标映射回去
    documents = [c["text"] for c in candidates]

    url = _service_host(settings.base_url) + RERANK_PATH
    payload = {
        "model": RERANK_MODEL,
        "input": {"query": query, "documents": documents},
        "parameters": {
            "return_documents": False,
            # top_n 不能超过候选数，否则接口可能报参数错误
            "top_n": min(top_k, len(documents)),
        },
    }

    resp = httpx.post(
        url,
        headers={"Authorization": f"Bearer {settings.api_key}"},
        json=payload,
        timeout=settings.timeout,
    )
    resp.raise_for_status()

    results = resp.json()["output"]["results"]
    # 自己按分数降序排一次：不依赖服务端返回顺序，这是防御性写法
    ranked = sorted(results, key=lambda r: r["relevance_score"], reverse=True)[:top_k]

    out: list[dict] = []
    for item in ranked:
        # index 是传入 documents 列表的下标，用它把原文和来源取回来
        cand = candidates[item["index"]]
        out.append({
            "text": cand["text"],
            "source": cand["source"],
            "score": round(float(item["relevance_score"]), 4),
            "vec_score": cand.get("score"),
        })
    return out
