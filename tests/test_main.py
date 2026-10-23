"""parse_json 的测试：模型输出不稳定，这是最该防的地方。

为什么这个模块最值得测：
    模型不保证只输出 JSON。它可能裹一层 markdown 代码块、可能先说句废话、
    也可能完全跑偏。这三种情况在真实使用中都会遇到，
    而且没法靠"再跑一次"复现——所以只能写成测试固定下来。
"""

import pytest

from llm_cli.main import parse_json


def test_parse_plain_json():
    """① 最理想：模型老老实实吐了一段干净 JSON。

    防的是：正常路径被改坏。
    比如有人调整了 strip() 的顺序，或者给函数加了新参数，
    结果连最普通的输入都解析不了了。
    """
    result = parse_json('{"title": "测试", "keywords": ["a", "b"]}')

    assert isinstance(result, dict)                 # 必须返回字典，不是字符串
    assert result["title"] == "测试"                 # 中文没被转义、没被截断
    assert isinstance(result["keywords"], list)     # 数组不能变成字符串
    assert len(result["keywords"]) == 2             # 元素个数没丢


def test_parse_json_in_code_fence():
    """② 模型裹了 markdown 代码块 —— 最常见的跑偏方式。

    防的是：围栏剥离逻辑失效。
    如果不剥，json.loads 会直接报
        Expecting value: line 1 column 1
    因为文本第一个字符是反引号，不是左大括号。
    """
    raw = '```json\n{"title": "测试"}\n```'

    result = parse_json(raw)

    assert result["title"] == "测试"


def test_parse_json_fence_without_language_tag():
    """③ 围栏不带 json 标记 —— 模型有时只写 ``` 不写 ```json。

    这条是我额外补的。真实输出里两种情况大约各占一半，
    如果你的代码只判断 "开头是不是 json"，这一条就会挂。
    """
    raw = '```\n{"title": "测试"}\n```'

    result = parse_json(raw)

    assert result["title"] == "测试"


def test_parse_json_with_chatter():
    """④ 模型先说句废话再给 JSON。

    防的是：最后一道防线失效 —— 靠"第一个 { 到最后一个 }"把内容抠出来。
    前两层（直接解析、剥围栏）都失败时，全靠这一步。
    """
    raw = '好的，结果如下：{"title": "测试"} 以上就是全部内容。'

    result = parse_json(raw)

    assert result["title"] == "测试"


def test_parse_garbage_raises():
    """⑤ 完全不是 JSON：必须抛 ValueError，不能返回 None。

    防的是：静默失败。
    如果这里悄悄 return None，下游会拿着 None 去写文件，
    最后报一个跟真正原因毫无关系的 TypeError —— 那是最难查的一类 bug。
    """
    with pytest.raises(ValueError):
        parse_json("我不确定你在问什么")
