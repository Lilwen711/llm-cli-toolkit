"""编排层：命令行入口 + 三段流程的串联。"""

import argparse
import json
import logging                    # Python 自带的日志库，不用额外安装
from datetime import datetime     # 用来给日志文件起名字（带日期）
from pathlib import Path

from llm_cli.config import load_settings
from llm_cli.llm_client import LLMClient, LLMError
from llm_cli.loader import load


# __name__ 在这里等于 "llm_cli.main"，所以日志里能看出是哪个模块打的
logger = logging.getLogger(__name__)


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


def setup_logging(log_dir):
    """配置日志系统：让日志同时出现在屏幕和文件里。

    为什么需要这个函数：
        print() 只能打到屏幕上，关掉终端就全没了。
        日志会写进文件，出问题时能回头翻。
    这个函数在程序启动时调用一次就够了。
    """

    # ---------- 第 1 步：确保 logs 文件夹存在 ----------
    # parents=True  → 上层目录也不存在的话，一起创建
    # exist_ok=True → 文件夹已存在时不报错（否则第二次运行就崩）
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 第 2 步：决定日志文件叫什么 ----------
    # datetime.now() 是当前时间，:%Y-%m-%d 让它变成 "2026-10-22"
    # 同一天不管跑多少次，都追加进同一个文件
    log_file = log_dir / f"{datetime.now():%Y-%m-%d}.log"

    # ---------- 第 3 步：规定每一行长什么样 ----------
    # 最终效果：
    #   18:12:03 | INFO    | llm_cli.main | 完成：data/a.txt -> out\a.json
    #
    #   %(asctime)s   → 时间
    #   %(levelname)s → 级别名；"-7s" 的意思是"占 7 个字符，不够的补空格"，
    #                   纯粹为了让 INFO / ERROR 后面的竖线对齐，好看而已
    #   %(name)s      → 哪个模块打的（因为每个模块里都有 getLogger(__name__)）
    #   %(message)s   → 你真正要写的那句话
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",   # 只要时分秒，日期在文件名里已经有了
    )

    # ---------- 第 4 步：准备两个"出口" ----------
    # 出口①：屏幕（StreamHandler 默认就是往控制台打）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)      # 让屏幕输出也用上面那个格式

    # 出口②：文件。
    # ⚠ encoding="utf-8" 千万不能省！
    #   Windows 上默认用 GBK 写文件，中文会变成乱码——今天你已经在这上面踩过一次了。
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # ---------- 第 5 步：把出口挂到"总闸"上 ----------
    # getLogger() 不传参数 = 拿到"根日志器"，所有模块的日志最后都汇到这里
    root = logging.getLogger()

    # 设最低门槛：INFO 表示"INFO 及以上"都输出，
    # 也就是 ERROR / WARNING / INFO 会出来，DEBUG 不会
    root.setLevel(logging.INFO)

    # 先清空已有的出口。
    # 为什么？万一这函数被调用两次，出口会重复挂载，
    # 结果就是同一行日志打印两遍。先清掉最保险。
    root.handlers.clear()

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # ---------- 第 6 步：压掉第三方库的日志噪音 ----------
    # httpx 是 OpenAI SDK 底层用的网络库，它会以 INFO 级别记录每一次 HTTP 请求。
    # 你把根日志器设成 INFO，等于顺手把它也打开了 ——
    # 结果就是每调一次模型，日志里多一行又长又难读的完整 URL。
    # 把它单独压到 WARNING 以上：平时安静，只在它真出问题时才说话。
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


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

    # 日志必须在"开始干活"之前装好。
    # 如果放在 if 语句里或放到后面，前面发生的事就留不下记录。
    setup_logging(Path("logs"))
    logger.info("开始处理：%s", args.file)

    if args.model:
        settings.model = args.model
        logger.info("模型被命令行覆盖为：%s", args.model)

    try:
        doc = load(args.file)
    except FileNotFoundError as exc:
        # 人看日志，脚本看退出码，两个都要
        logger.error("文件读取失败：%s", exc)
        return 2
    except ValueError as exc:
        logger.error("文件读取失败：%s", exc)
        return 3

    # 这条 INFO 是"中间状态"的价值所在：
    # 出问题时，日志里能看出文件到底读到没有、读到多少
    logger.info("读取完成：%s 字符", doc.meta.get("chars"))

    client = LLMClient(settings)
    try:
        raw = client.complete(INSTRUCTIONS, doc.text)
    except LLMError as exc:
        logger.error("模型调用失败：%s", exc)
        return 4

    try:
        data = parse_json(raw)
    except ValueError as exc:
        logger.error("JSON 解析失败：%s", exc)
        return 5

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(args.file).stem + ".json")
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("完成：%s -> %s", args.file, out_path)
    return 0


def main(argv=None) -> int:
    return cmd_parse(build_parser().parse_args(argv))
