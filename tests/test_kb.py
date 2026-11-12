"""kb 模块测试：只测不依赖 API 的部分（chunk 切分正确性 + search 返回结构契约）。

embedding/检索的端到端验证走 scripts/verify_kb.py，不放 pytest（避免每次跑测试都调 API 花钱）。
"""
from llm_cli.kb.chunk import chunk_by_heading, chunk_document


def test_chunk_empty_returns_empty():
    """空文本返回空列表，不报错。"""
    assert chunk_document("", "x.md") == []


def test_chunk_basic_split_and_source():
    """正常文本切分，每项含 text 和 source（带条目号）。"""
    text = "# 标题\n\n## 条-01 第一条\n这是第一条内容，足够长用来测试切分。\n\n## 条-02 第二条\n这是第二条内容。"
    chunks = chunk_document(text, "demo.md", chunk_size=60, overlap=10)
    assert len(chunks) >= 1
    for c in chunks:
        assert "text" in c and "source" in c
        assert c["source"].startswith("demo.md#")


def test_chunk_section_tracking():
    """source 记「块首条目号」：块从哪个条目开始，就标哪个。

    旧断言（source 含 退款-02）钉住的是 bug：一个块跨了 01、02 两个条目，
    却被标成最后遇到的 02，等于把「跨条目 + 标错」写成了期望行为。
    现约定：字符分块允许跨条目，source 只承诺「块首条目」，因此这里是 01。
    周四按 ## 标题分块后，一个块只含一个条目，这条断言会升级为 len==2。
    """
    text = "## 退款-01 七天无理由\n内容A\n\n## 退款-02 不适用\n内容B"
    chunks = chunk_document(text, "退款.md", chunk_size=1000, overlap=0)
    # chunk_size 够大，切成 1 块；块从 退款-01 开始，故 source 标 01
    assert len(chunks) == 1
    assert chunks[0]["source"] == "退款.md#退款-01"


def test_chunk_respects_size_limit():
    """chunk_size 限制生效，不会产出超长块（除非单段超长）。"""
    text = "## a-01 标题\n" + "x" * 200 + "\n\n## a-02 标题\n" + "y" * 200
    chunks = chunk_document(text, "a.md", chunk_size=100, overlap=0)
    # 每块文本长度应受 chunk_size 约束（单段超长例外）
    for c in chunks:
        # 允许略超（含标题行），但不应是全文 400+
        assert len(c["text"]) < 300


def test_chunk_overlap_creates_continuity():
    """overlap 让相邻块有重叠内容。"""
    text = "## a-01 标题\n" + "abcdefghij" * 30  # 300 字符
    chunks = chunk_document(text, "a.md", chunk_size=100, overlap=20)
    if len(chunks) >= 2:
        # 第二块开头应包含第一块末尾的部分内容
        assert chunks[1]["text"][:10] in chunks[0]["text"]


# ==================== 策略二：Markdown 标题分块（W4） ====================
# 这四个 case 就是周三设计稿的四个边界，一个都不能少。


def test_md_one_chunk_per_section():
    """边界 1：一个 chunk = 一个条目，source 天然唯一，不再跨条目。"""
    text = "# 退款政策\n\n## 退款-01 七天无理由\n内容A\n\n## 退款-02 不适用\n内容B"
    chunks = chunk_by_heading(text, "退款.md")
    assert [c["source"] for c in chunks] == ["退款.md#退款-01", "退款.md#退款-02"]
    assert "内容A" in chunks[0]["text"]
    assert "内容B" in chunks[1]["text"]


def test_md_preface_kept_only_if_has_body():
    """边界 2：文档头（第一个 ## 之前）有正文才独立成块，只有标题就丢掉。"""
    with_body = "# 政策\n\n适用范围：所有实物商品。\n\n## 退款-01 x\nA"
    assert [c["source"] for c in chunk_by_heading(with_body, "p.md")] == [
        "p.md#前言",
        "p.md#退款-01",
    ]

    title_only = "# 政策\n\n## 退款-01 x\nA"
    assert [c["source"] for c in chunk_by_heading(title_only, "p.md")] == ["p.md#退款-01"]


def test_md_long_section_split_keeps_source():
    """边界 3：单条目超长时二次切分，多块共用同一条目号（source 不加后缀）。"""
    body = "## 保修-02 保修范围\n" + "\n\n".join(["段落" + "x" * 100] * 6)
    chunks = chunk_by_heading("# 保修条款\n\n" + body, "保修.md", max_chars=200)
    assert len(chunks) >= 2
    assert {c["source"] for c in chunks} == {"保修.md#保修-02"}
    assert all(len(c["text"]) <= 200 for c in chunks)


def test_md_unnumbered_heading_merges():
    """边界 4：提不出编号的 ## 标题（如「## 注意事项」）不切块，并入上一块。"""
    text = "## 退款-01 x\nA\n\n## 注意事项\nB"
    chunks = chunk_by_heading(text, "p.md")
    assert len(chunks) == 1
    assert chunks[0]["source"] == "p.md#退款-01"
    assert "B" in chunks[0]["text"]
