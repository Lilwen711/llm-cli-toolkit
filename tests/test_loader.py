"""loader 模块的测试：只测"读文件"这件事，不碰网络。"""

import pytest

from llm_cli.loader import load


def test_load_txt_returns_text():
    """① 正常路径：能读出内容和来源信息。"""
    doc = load("data/a.txt")

    assert len(doc.text) > 100                    # 不能读成空字符串
    assert doc.meta["chars"] == len(doc.text)     # 字数要自洽
    assert doc.meta["encoding"] == "utf-8"
    assert "a.txt" in doc.meta["path"]


def test_load_missing_file_raises():
    """② 文件不存在：必须抛 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        load("data/根本不存在.txt")


def test_load_unsupported_suffix_raises(tmp_path):
    """③ 不支持的格式：必须抛 ValueError。

    tmp_path 是 pytest 送的"临时文件夹"，每个测试独立一个，
    用完自动清理。这样测试不依赖仓库里有没有某个文件。
    """
    f = tmp_path / "note.md"
    f.write_text("随便写点什么", encoding="utf-8")

    with pytest.raises(ValueError):
        load(f)
