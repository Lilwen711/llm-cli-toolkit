#选基镜像
FROM python:3.12-slim
#定工作目录
WORKDIR /app
#装工具
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ uv
#拷描述
COPY pyproject.toml uv.lock README.md ./
#拷代码
COPY src ./src
#装依赖
RUN UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ uv sync --no-dev
#拷数据
COPY data ./data
#定入口
ENTRYPOINT ["uv", "run", "--no-sync", "llm-cli"]

