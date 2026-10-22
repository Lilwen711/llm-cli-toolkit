"""模型层：只负责把一段话发给模型，把原始字符串拿回来。"""

import logging     # 打日志
import time        # 重试之间要等一会儿

from openai import (
    APIConnectionError,     # 连不上服务器（断网、DNS 解析失败）
    APITimeoutError,        # 连上了，但等太久没回应
    InternalServerError,    # 服务器自己出错了（5xx）
    OpenAI,                 # 客户端本体
    OpenAIError,            # "所有 OpenAI 错误"的总类，下面那几个都是它的子类
    RateLimitError,         # 你调得太快被限流了（429）
)

from llm_cli.config import Settings

# __name__ 在这里会等于 "llm_cli.llm_client"，日志里就能看出是哪个文件打的
logger = logging.getLogger(__name__)


# ★★ 这个元组是本文件的核心：只有这四类错误值得"再试一次"。
#    共同点：都是"对面的临时问题"，过一会儿可能就恢复了。
#
#    故意不放进去的（都是"你的问题"，重试一万次结果一样）：
#      400 BadRequestError       请求格式写错了
#      401 AuthenticationError   Key 不对
#      403 PermissionDeniedError 没开通权限
#      404 NotFoundError         模型名不存在
RETRYABLE = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


class LLMError(RuntimeError):
    """所有模型调用失败，最后都收敛成这一个异常，上层只要抓这一种。"""


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # 真正干活的客户端，只在初始化时创建一次。
        # 放 __init__ 里而不是每次调用时创建，是为了复用连接（更快、更省资源）
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
        )

    def complete(self, instructions: str, user_input: str) -> str:
        """发一次请求，返回模型输出的原始字符串（不做任何解析）。"""

        last_error = None      # 记住最后一次的错误，三次都挂了要把它带出去

        # range(3) 会依次给出 0, 1, 2 —— 一共最多试 3 次
        for attempt in range(self.settings.max_retries):
            try:
                # 真正发请求。成功就直接 return，循环到此结束
                response = self._client.responses.create(
                    model=self.settings.model,
                    instructions=instructions,     # 相当于 system 提示
                    input=user_input,              # 用户内容
                    text={"format": {"type": "json_object"}},   # 要求输出 JSON
                )
                return response.output_text

            # ---------- 情况 A：可重试的错误 ----------
            # except 是从上往下匹配的，所以"具体的"必须写在"宽泛的"前面。
            # 如果这里和下面的顺序调换，所有错误都会被下面的 OpenAIError 吃掉，
            # 分流就彻底失效了 —— 这是这段代码最容易写错的地方。
            except RETRYABLE as exc:
                last_error = exc

                # 用 WARNING 而不是 ERROR：因为还没到"彻底失败"，还能再试
                logger.warning(
                    "第 %d/%d 次调用失败（可重试）：%s",
                    attempt + 1,                  # 人是从一开始数的，所以要 +1
                    self.settings.max_retries,
                    exc,
                )

                # 最后一次失败后就不用再等了，因为马上要抛异常了
                if attempt < self.settings.max_retries - 1:
                    # 2 ** attempt 会得到 1, 2, 4 —— 每次等更久，叫"退避"
                    # 为什么要越来越久？因为如果是对面太忙，你越快重试越添乱
                    time.sleep(2 ** attempt)

            # ---------- 情况 B：不可重试的错误 ----------
            # 走到这里说明既不是网络问题也不是服务端问题，
            # 而是请求本身有问题（400 / 401 / 404 …），再试也没意义。
            except OpenAIError as exc:
                logger.error("调用失败且不可重试：%s", exc)
                # 直接抛出去，不再进下一轮循环
                raise LLMError(f"模型调用失败（不可重试）：{exc}") from exc

        # 循环正常跑完 = 三次都是可重试的错，都没成功
        raise LLMError(
            f"模型调用失败，已重试 {self.settings.max_retries} 次：{last_error}"
        ) from last_error
