"""kb 模块测试：只测不依赖 API 的部分（chunk 切分正确性 + search 返回结构契约）。

embedding/检索的端到端验证走 scripts/verify_kb.py，不放 pytest（避免每次跑测试都调 API 花钱）。
"""
from llm_cli.kb.chunk import chunk_document


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
