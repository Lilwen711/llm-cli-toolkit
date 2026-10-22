"""配置层：把 .env 里的东西变成 Settings 对象。

全项目只有这一个文件碰环境变量，其他模块只认 Settings。
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


REQUIRED_KEYS = ("DASHSCOPE_API_KEY",)


@dataclass
class Settings:
    api_key: str
    model: str = "qwen3.8-flash"
    base_url: str = "https://ws-c9cdhmp9u779ibse.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    timeout: float = 30.0
    max_retries: int = 3
    out_dir: Path = Path("out")



def _fail(message: str) -> None:
    print(f"[配置错误] {message}", file=sys.stderr)
    sys.exit(1)


def load_settings(env_file: str = ".env") -> Settings:
    load_dotenv(env_file)

    missing = [key for key in REQUIRED_KEYS if not os.getenv(key)]
    if missing:
        _fail(
            "缺少环境变量：" + "、".join(missing)
            + "\n请填入你的百炼 Key"
        )

    return Settings(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        model=os.getenv("LLM_MODEL", "qwen3.8-flash"),
        base_url=os.getenv(
            "LLM_BASE_URL",
            "https://ws-c9cdhmp9u779ibse.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        ),
        timeout=float(os.getenv("LLM_TIMEOUT", 30)),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", 3)),
        out_dir=Path(os.getenv("LLM_OUT_DIR", "out")),
    )
