"""模型层：只负责把一段话发给模型，把原始字符串拿回来。"""

import time

from openai import OpenAI, OpenAIError

from llm_cli.config import Settings


class LLMError(RuntimeError):
    """所有模型调用失败，统一收敛成这一个异常。"""


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
        )

    def complete(self, instructions: str, user_input: str) -> str:
        last_error = None

        for attempt in range(self.settings.max_retries):
            try:
                response = self._client.responses.create(
                    model=self.settings.model,
                    instructions=instructions,
                    input=user_input,
                    text={"format": {"type": "json_object"}},
                )
                return response.output_text
            except OpenAIError as exc:
                last_error = exc
                if attempt < self.settings.max_retries - 1:
                    time.sleep(2 ** attempt)

        raise LLMError(
            f"模型调用失败，已重试 {self.settings.max_retries} 次：{last_error}"
        ) from last_error
