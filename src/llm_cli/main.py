"""编排层：命令行入口 + 三段流程的串联。"""

import argparse
import json
import sys
from pathlib import Path

from llm_cli.config import load_settings
from llm_cli.llm_client import LLMClient, LLMError
from llm_cli.loader import load


INSTRUCTIONS = """你是一个文本结构化助手。请阅读用户提供的文本，输出一个 JSON 对象。

要求：
1. 只输出 JSON，不要任何解释，不要 markdown 代码块
2. 必须包含这五个字段：
   - title：字符串，文本的标题
   - summary：字符串，不超过 80 字
   - keywords：字符串数组，3 到 6 个
   - category：字符串，文本所属类别
   - sentiment：字符串，只能是 positive、neutral、negative 之一
3. 所有内容必须来自原文，禁止编造原文中没有的信息
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm-cli",
        description="读本地文件，调大模型，输出结构化 JSON",
    )
    sub = p.add_subparsers(dest="command", required=True)
    parse_cmd = sub.add_parser("parse", help="解析一个文件并输出 JSON")
    parse_cmd.add_argument("file")
    parse_cmd.add_argument("--out", default="out")
    parse_cmd.add_argument("--model", default=None)
    return p


def parse_json(raw: str) -> dict:
    text = raw.strip()

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

    raise ValueError("模型输出不是合法 JSON")


def cmd_parse(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.model:
        settings.model = args.model

    try:
        doc = load(args.file)
    except FileNotFoundError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 3

    client = LLMClient(settings)
    try:
        raw = client.complete(INSTRUCTIONS, doc.text)
    except LLMError as exc:
        print(f"[模型错误] {exc}", file=sys.stderr)
        return 4

    try:
        data = parse_json(raw)
    except ValueError as exc:
        print(f"[格式错误] {exc}", file=sys.stderr)
        return 5

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(args.file).stem + ".json")
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[完成] {args.file} -> {out_path}")
    return 0


def main(argv=None) -> int:
    return cmd_parse(build_parser().parse_args(argv))
