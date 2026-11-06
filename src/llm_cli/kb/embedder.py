"""向量化：文本 -> 1024 维浮点向量。

走标准 dashscope 端点（OpenAI 兼容 /v1/embeddings），模型 qwen3.7-text-embedding-flash。
W2 硬约束①：embedding 非确定性（同输入两次结果余弦 0.9987），
            因此向量入库一次落盘复用，缓存键 = 文本哈希 + 模型名。
"""
import hashlib
import json
import logging
import os
from pathlib import Path

from openai import OpenAI

from llm_cli.config import Settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding-flash")
BATCH_SIZE = 20  # qwen3.7-text-embedding-flash 单次最多 20 条
DEFAULT_CACHE = Path("data/embeddings_cache.json")


def _cache_key(text: str, model: str) -> str:
    return hashlib.md5(f"{model}:{text}".encode("utf-8")).hexdigest()


class Embedder:
    def __init__(self, settings: Settings, cache_file: Path = DEFAULT_CACHE) -> None:
        self.model = EMBEDDING_MODEL
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            max_retries=0,
        )
        self._cache: dict[str, list[float]] = {}
        self.cache_file = cache_file
        self._load_cache()

    def _load_cache(self) -> None:
        if self.cache_file.exists():
            self._cache = json.loads(self.cache_file.read_text("utf-8"))
            logger.info("embedding 缓存命中：%d 条", len(self._cache))

    def _save_cache(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(self._cache, ensure_ascii=False), "utf-8")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量向量化，命中缓存的不调 API。"""
        results: list[list[float] | None] = [None] * len(texts)
        miss_idx: list[int] = []
        for i, t in enumerate(texts):
            key = _cache_key(t, self.model)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                miss_idx.append(i)

        # 未命中的分批调 API（每批 ≤ BATCH_SIZE=20）
        for start in range(0, len(miss_idx), BATCH_SIZE):
            batch_pos = miss_idx[start:start + BATCH_SIZE]
            batch_texts = [texts[i] for i in batch_pos]
            resp = self._client.embeddings.create(model=self.model, input=batch_texts)
            for pos, item in zip(batch_pos, resp.data):
                vec = item.embedding
                self._cache[_cache_key(texts[pos], self.model)] = vec
                results[pos] = vec
            logger.info("embedding 调用：本批 %d 条", len(batch_pos))

        self._save_cache()
        return results  # type: ignore

    def embed_query(self, query: str) -> list[float]:
        """单条查询向量化（用同一模型，向量空间须一致；不缓存，量小）。"""
        resp = self._client.embeddings.create(model=self.model, input=query)
        return resp.data[0].embedding
