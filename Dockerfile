FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ uv

COPY pyproject.toml uv.lock README.md ./

COPY src ./src

RUN UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ uv sync --no-dev

COPY data ./data

ENTRYPOINT ["uv", "run", "llm-cli"]
