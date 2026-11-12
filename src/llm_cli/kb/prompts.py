"""生成段的 prompt 模板：RAG 的「第六段」，也是防幻觉的最后一道闸。

设计原则只有一条：**只让模型用给定的资料回答，并且必须报出依据**。
引用不是装饰，是可以被程序校验的东西 —— 模型真引用了哪一条，我们就去查那一条对不对。

v2 修正（第一轮实测发现的问题）：
  第一版把资料写成「[1] 条目号：物流-02（来源：…）」，指令又写「依据：条目号1、条目号2」，
  结果模型把「条目号」这三个字当成编号的一部分抄了出来（`依据：条目号1`、`条目号保修-03`），
  引用格式合法率只有 77%。根因是**模板措辞歧义**，不是模型不行。
  v2 做三件事：① 资料行去掉"条目号："标签，改成纯数据 `[1] 物流-02 | 来源：…`；
  ② 把「可选条目编号」显式列在 prompt 里，禁止自造；③ 解析器兜底剥掉误抄的前缀。
"""
# 指令（放在 system 位）。四条规则各自对应一类失败：
#   1 防幻觉（资料里没有就明说）  2 控长度  3/4 让引用可解析、可校验
INSTRUCTIONS = """你是售后客服助手。只能依据下面提供的「参考资料」回答问题，禁止使用资料以外的知识。

规则：
1. 如果参考资料不足以回答这个问题，只回答这一句：资料里没有相关规定。不要猜测、不要补充。
2. 回答控制在 3 句话以内，先给结论，再给关键条件。
3. 回答结束后另起一行，只写：依据：编号1、编号2
   - 编号必须原样取自「可选条目编号」列表，不许自造。
   - 只写编号本身（例如 物流-05），不要写「条目号」三个字，不要写序号。
   - 至少写一个。"""

# 用户消息模板：资料 + 可选编号白名单 + 问题
# 白名单是给模型的「合法取值范围」，也是给解析器的校验依据
USER_TEMPLATE = """参考资料：
{context}

可选条目编号：{allowed}

问题：{query}"""

# 单条资料的排版：纯数据行，不带「条目号：」这类会被误抄的标签词
DOC_TEMPLATE = """[{idx}] {section} | 来源：{source}
{text}"""


def build_context(candidates: list[dict]) -> str:
    """把检索结果拼成 prompt 里的「参考资料」块。

    Args:
        candidates: rerank/search 返回的 [{"text","source","score"}, ...]

    Returns:
        形如 "[1] 退款-02 | 来源：退款政策.md#退款-02\\n正文..." 的多段文本
    """
    blocks = []
    for i, c in enumerate(candidates, 1):
        # source 形如 "退款政策.md#退款-02"，取 # 后半段当条目编号；没有 # 就用文件名兜底
        section = c["source"].split("#")[-1] or c["source"]
        blocks.append(
            DOC_TEMPLATE.format(idx=i, section=section, source=c["source"], text=c["text"])
        )
    return "\n\n".join(blocks)


def build_user_prompt(query: str, candidates: list[dict]) -> str:
    """拼出最终发给模型的用户消息。"""
    # 可选编号白名单：只列本次真正喂进去的那几条，模型照抄即可，不会编造
    allowed = "、".join(c["source"].split("#")[-1] for c in candidates)
    return USER_TEMPLATE.format(
        context=build_context(candidates), allowed=allowed, query=query
    )
