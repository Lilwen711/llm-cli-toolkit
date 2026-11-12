"""向量存储：Chroma 本地文件持久化。

入库时每个 chunk 绑定 metadata: {"source": "文件名#条目号"}。
二次加载已有库不重算 embedding（Chroma 持久化 + embedder 缓存双保险）。
去重：按 chunk 文本 hash 做 id，已存在的不重 add 不重算。
"""
import hashlib
import logging
from pathlib import Path

import chromadb

from llm_cli.config import Settings

from .chunk import chunk_by_heading
from .embedder import Embedder

logger = logging.getLogger(__name__)

PERSIST_DIR = Path("data/chroma")
COLLECTION = "kb_store"  # Chroma 要求 3-512 字符、[a-zA-Z0-9._-]、首尾字母数字
EXPECTED_SPACE = "cosine"  # 距离度量必须显式声明，不能吃默认值（默认是 l2）
CHUNK_MODE = "heading"  # 切分策略：heading=按 ## 条目切；chars=字符分块
MAX_CHARS = 512
COLLECTION_META = {
    "note": "kb-4docs / Markdown 标题分块 / cosine 度量（W4）",
    "chunk_mode": CHUNK_MODE,
    "max_chars": MAX_CHARS,
}


def _chunk_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def _open_collection(client):
    """拿到度量正确的 collection；旧库若是 l2 建的，删掉重建。

    collection 的度量（hnsw:space）在建库那一刻就写进索引，之后**不可修改**：
    get_or_create_collection 传新 metadata 会被静默忽略，不报错。
    所以这里显式检查一次，不一致就删库——向量从 embeddings_cache.json 取，零 API 调用。
    """
    try:
        coll = client.get_collection(COLLECTION)
    except Exception:
        coll = None

    if coll is not None:
        space = coll.configuration["hnsw"]["space"]
        mode = (coll.metadata or {}).get("chunk_mode", "")
        if space == EXPECTED_SPACE and mode == CHUNK_MODE:
            return coll
        logger.warning(
            "库的度量是 %s、切分策略是 %s，代码要求 %s + %s —— 配置不可改，删库重建",
            space,
            mode or "<未声明>",
            EXPECTED_SPACE,
            CHUNK_MODE,
        )
        client.delete_collection(COLLECTION)

    return client.create_collection(
        COLLECTION,
        configuration={"hnsw": {"space": EXPECTED_SPACE}},
        metadata=COLLECTION_META,
    )


def build_store(texts_by_source: dict[str, str], settings: Settings) -> int:
    """建库：{文件名: 文档文本} -> 切分 -> 向量化 -> 存 Chroma。

    已存在的 chunk（按 text hash 去重）不重算不重 add。
    Returns: 入库后库内 chunk 总数。
    """
    embedder = Embedder(settings)
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    coll = _open_collection(client)

    # 切分所有文档（W4：按 ## 条目切，一 chunk 一条目，source 天然唯一）
    all_chunks: list[dict] = []
    for source, text in texts_by_source.items():
        all_chunks.extend(chunk_by_heading(text, source, max_chars=MAX_CHARS))

    # 去掉库里已存在的
    existing_ids = set(coll.get()["ids"]) if coll.count() > 0 else set()
    new_chunks = [c for c in all_chunks if _chunk_id(c["text"]) not in existing_ids]

    if not new_chunks:
        logger.info("库已存在 %d chunk，无新增，零 embedding 调用", coll.count())
        return coll.count()

    logger.info("切分 %d chunk，新增 %d，调 embedding", len(all_chunks), len(new_chunks))
    vectors = embedder.embed_texts([c["text"] for c in new_chunks])
    coll.add(
        ids=[_chunk_id(c["text"]) for c in new_chunks],
        documents=[c["text"] for c in new_chunks],
        metadatas=[{"source": c["source"]} for c in new_chunks],
        embeddings=vectors,
    )
    logger.info("入库完成，库共 %d chunk", coll.count())
    return coll.count()


def get_collection():
    """加载已有库的 collection。"""
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return client.get_collection(COLLECTION)
