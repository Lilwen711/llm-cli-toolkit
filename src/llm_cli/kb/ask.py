"""生成段：检索（+rerank）→ 拼 prompt → 大模型生成 → 解析引用。

这是 RAG 六段的最后一段：前面五段决定「找到什么」，这一段决定「怎么用、用哪条」。
输出固定包含两部分：答案正文 + 引用条目号。引用能被程序校验，是防幻觉的第一道闸。

设计取舍：
- 不做「多轮对话」、不做「流式输出」、不做「重排后再检索」，本周只求闭环 + 可评测。
- 引用解析失败不抛异常，而是标记 format_ok=False —— 评测时需要知道「格式错」和
  「模型拒答」是两回事，不能混成一个错。
"""
import logging
import re

from llm_cli.config import Settings, load_settings
from llm_cli.llm_client import LLMClient

from .prompts import INSTRUCTIONS, build_user_prompt
from .rerank import rerank
from .search import search

logger = logging.getLogger(__name__)

TOP_N = 20  # 粗召回条数
TOP_K = 3   # 精排后喂给模型的条数

# 匹配最后那行「依据：退款-02、物流-05」（中英文冒号都忍）
_CITE_RE = re.compile(r"依据\s*[:：]\s*(.+)")


def parse_citations(answer: str, candidates: list[dict]) -> dict:
    """从模型输出里解析引用，并校验条目号是否真实存在于候选资料中。

    Returns:
        {"answer": 去掉依据行的正文,
         "cited": ["退款-02", ...],        # 模型写的全部条目号（原样）
         "valid": ["退款-02", ...],        # 其中确实在候选里的
         "format_ok": bool}                # 有依据行 + 至少一条引用 + 全部合法
    """
    # 候选里出现过的条目号集合，用来校验模型有没有编造
    known = {c["source"].split("#")[-1] for c in candidates}

    match = _CITE_RE.search(answer)
    if not match:
        # 没写依据行：格式非法，但答案正文照常保留（可能是模型拒答）
        return {"answer": answer.strip(), "cited": [], "valid": [], "format_ok": False}

    raw = match.group(1)
    # 逗号、顿号、斜杠、空格都当分隔符，容忍模型写法不统一
    cited = []
    for token in re.split(r"[、,，/\s]+", raw):
        token = token.strip()
        if not token:
            continue
        # 兜底：模型可能把模板里的「条目号」三个字一起抄进来（依据：条目号保修-03）
        token = re.sub(r"^条目号?号?", "", token).strip()
        # 纯数字是序号（依据：条目号1），不是条目编号，直接丢掉
        if not token or token.isdigit():
            continue
        cited.append(token)

    valid = [x for x in cited if x in known]

    # 正文 = 依据行之前的全部内容
    body = answer[: match.start()].strip()

    # 合法标准：有引用、且每一条都能在候选里找到（编造条目号直接判非法）
    format_ok = bool(cited) and len(valid) == len(cited)
    return {"answer": body, "cited": cited, "valid": valid, "format_ok": format_ok}


def ask(
    query: str,
    top_n: int = TOP_N,
    top_k: int = TOP_K,
    use_rerank: bool = True,
    settings: Settings | None = None,
) -> dict:
    """端到端问答：query -> 答案 + 引用。

    Args:
        query: 用户问题
        top_n: 粗召回条数（向量检索取多少条）
        top_k: 精排后喂给模型的条数
        use_rerank: 关掉可复现「不接 rerank」的对照组
        settings: 可选，不传则从 .env 加载

    Returns:
        {"query", "answer", "cited", "valid", "format_ok", "sources"}
        sources 是真正喂给模型的那 k 条（含分数），评测时用来算检索命中率。
    """
    if settings is None:
        settings = load_settings()

    # 第一段路：粗召回。多召回一些（20），交给 rerank 精排
    candidates = search(query, k=top_n, settings=settings)

    # 第二段路：精排。rerank 挂掉时不让整个问答崩，降级为「直接用向量前 k 条」并留日志
    if use_rerank and candidates:
        try:
            candidates = rerank(query, candidates, top_k=top_k, settings=settings)
        except Exception as exc:  # noqa: BLE001 - 这里故意兜住，降级路径要留痕
            logger.warning("rerank 失败，降级为向量排序：%s", exc)
            candidates = candidates[:top_k]
    else:
        candidates = candidates[:top_k]

    if not candidates:
        # 库里什么都没检索到：不调用模型，直接给结论，省一次生成调用
        return {
            "query": query, "answer": "资料里没有相关规定。",
            "cited": [], "valid": [], "format_ok": False, "sources": [],
        }

    # 第三段路：拼 prompt -> 生成。复用 W1 写好的 LLMClient（自带 3 次重试）
    client = LLMClient(settings)
    # json_mode=False：这里要的是「答案正文 + 依据行」，不是 JSON
    raw = client.complete(INSTRUCTIONS, build_user_prompt(query, candidates), json_mode=False)

    # 第四段路：解析引用并回填结构化结果
    parsed = parse_citations(raw, candidates)
    return {
        "query": query,
        "answer": parsed["answer"],
        "cited": parsed["cited"],
        "valid": parsed["valid"],
        "format_ok": parsed["format_ok"],
        "sources": candidates,
    }
