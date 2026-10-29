# llm-cli-toolkit

把本地 TXT / PDF 文件交给大模型，输出一份结构化 JSON 的命令行工具。

## 这是什么

给一个文件路径，它读出正文 → 调 qwen3.8-flash → 产出包含
`title` / `summary` / `keywords` / `category` / `sentiment` 五个字段的 JSON。

**支持的格式**

| 格式 | 说明 |
|---|---|
| `.txt` | UTF-8 或 GBK 编码，自动识别 |
| `.pdf` | 有文字层的电子版 PDF（扫描件不支持，见「已知限制」） |

## 快速开始

```bash
uv sync
cp .env.example .env      # 打开 .env，把 DASHSCOPE_API_KEY 换成你自己的
uv run llm-cli parse data/a.txt
```

跑完结果在 `out/a.json`。

**需要装什么**

- Python 3.12 以上
- [uv](https://docs.astral.sh/uv/)（装依赖、跑命令都靠它）
- 一个阿里云百炼 API Key：控制台 → API-KEY 管理 → 创建

## 跑起来长什么样

终端输出：

```
19:21:26 | INFO    | llm_cli.main | 开始处理：data/a.txt
19:21:26 | INFO    | llm_cli.main | 读取完成：243 字符
19:21:51 | INFO    | llm_cli.main | 完成：data/a.txt -> out\a.json
```

`out/a.json`：

```json
{
  "title": "关于耳机退货政策的说明",
  "summary": "说明本店耳机七天无理由退货条件，包括商品完好、入耳式拆封使用不退换、质量问题可换新、定制刻字不支持退货及退货申请流程。",
  "keywords": ["耳机", "退货政策", "七天无理由", "质量问题", "定制刻字", "卫生考虑"],
  "category": "售后服务",
  "sentiment": "neutral"
}
```

## 部署到服务器

以下步骤在**一台全新的 Ubuntu 22.04 / 24.04** 上从零走完，不需要事先装 Python。
国内网络环境下的两个坑已在对应步骤标注。

### 1. 拉代码

```bash
git clone https://github.com/Lilwen711/llm-cli-toolkit.git
cd llm-cli-toolkit
```

### 2. 装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

> **坑 1｜`astral.sh` 在国内常被重置连接**（报 `Connection reset by peer`）。
> 装不上就改用 pip：
>
> ```bash
> apt install -y python3-pip
> python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --break-system-packages uv
> uv --version
> ```
>
> `--break-system-packages` 是 Ubuntu 24.04 的 PEP 668 保护，不加会拒绝安装。

### 3. 装依赖

```bash
uv sync
```

> **坑 2｜这一步会卡在下载上**。`uv.lock` 里记录的是 `files.pythonhosted.org` 的地址，
> 而 `--frozen` 的语义是"严格照 lock 执行"，所以**光加 `-i` 换源是无效的**。
> 必须让 uv 重新解析，同时把源指到国内镜像：
>
> ```bash
> cp uv.lock /tmp/uv.lock.bak
> UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ uv sync
> git checkout -- uv.lock     # uv 会改写 lock 里的下载地址，恢复原状再提交
> ```
>
> 不改 `uv.lock` 的原因：那会把所有人的依赖源绑死在阿里云。

### 4. 配 `.env`

```bash
cp .env.example .env
vim .env          # 把 DASHSCOPE_API_KEY 换成你自己的
```

`.env` 被 `.gitignore` 挡在仓库外，**不会跟着 clone 下来**，必须手工创建或从原机器 `scp` 过来：

```bash
scp <本机路径>/.env root@<服务器IP>:/root/llm-cli-toolkit/
```

### 5. 验证

```bash
uv run llm-cli parse data/a.txt    # 产出 out/a.json，五个字段
uv run pytest -q                   # 期望 8 passed
```

两条都过，部署即完成。

## 用 Docker 跑

```bash
docker build -t llm-cli:0.1 .
docker run --rm --env-file .env -v "$PWD/out:/app/out" llm-cli:0.1 parse data/a.txt
```

四个参数各自的作用，少一个都会出问题：

| 参数 | 作用 | 省了会怎样 |
|---|---|---|
| `--env-file .env` | 运行时把密钥注入成环境变量 | 报「缺少环境变量：DASHSCOPE_API_KEY」 |
| `-v "$PWD/out:/app/out"` | 把容器的 `out/` 映射回宿主机 | 容器退出后 JSON 跟着容器一起删掉 |
| `parse data/a.txt` | 接到 `ENTRYPOINT` 的 `llm-cli` 后面 | —— |
| `--rm` | 跑完自动删容器 | 容器越积越多，`docker ps -a` 一屏 |

**密钥不进镜像**：`.env` 被 `.dockerignore` 挡住，`Dockerfile` 用的是白名单 `COPY`（只拷 `src/`、`data/` 和三个依赖描述文件），
所以镜像里没有任何密钥。验证方式：

```bash
docker run --rm --entrypoint ls llm-cli:0.1 -la /app     # 不应出现 .env
docker history --no-trunc llm-cli:0.1 | grep -c "sk-"    # 应为 0
```

**入口带 `--no-sync`，不要去掉**：`ENTRYPOINT` 是 `uv run --no-sync llm-cli`。
不加 `--no-sync` 的话，`uv run` 每次启动都会同步 dev 依赖（pytest 那 5 个包）——
在能联网的机器上表现为启动慢几秒，在**只有内网**的环境里会直接卡住。
依赖在 `build` 阶段就该装全，运行时不该再碰网络。

**想在容器里执行别的命令**，必须用 `--entrypoint` 覆盖（因为默认入口是 `uv run llm-cli`，
后面跟的参数会被当成子命令）：

```bash
docker run --rm --entrypoint sh llm-cli:0.1        # 进 shell
docker run --rm --entrypoint ls llm-cli:0.1 -la /app
```

> **坑 3｜拉基础镜像失败**（报 `failed to resolve reference ... not found`）。
> 阿里云的个人版加速器对 `library/python` 这类多架构镜像经常返回 404，换一个：
>
> ```bash
> cat > /etc/docker/daemon.json <<'EOF'
> { "registry-mirrors": ["https://docker.m.daocloud.io"] }
> EOF
> systemctl daemon-reload && systemctl restart docker
> ```

> **坑 4｜容器日志时间比本地慢 8 小时**。基础镜像默认 UTC，需要在运行时注入时区：
>
> ```bash
> docker run --rm -e TZ=Asia/Shanghai -v /etc/localtime:/etc/localtime:ro ...
> ```

## 架构

```
                ┌──────────────┐
                │   main.py    │  唯一认识所有模块
                └──┬────┬───┬──┘
        ┌──────────┘    │   └──────────┐
        ▼               ▼              ▼
  ┌───────────┐  ┌─────────────┐  ┌────────────┐
  │ loader.py │  │llm_client.py│  │ config.py  │
  └───────────┘  └──────┬──────┘  └────────────┘
                        │                ▲
                        └────────────────┘
```

| 模块 | 职责 | 它不知道什么 |
|---|---|---|
| `config.py` | 唯一读环境变量的地方；缺 Key 直接退出，退出码 1 | 不知道有文件、有模型 |
| `loader.py` | 文件 → `Document(text, meta)`，按后缀分发 txt / pdf | 不知道有模型；也不依赖 config |
| `llm_client.py` | 文本 → 模型返回的原始字符串；失败时重试，异常统一收敛成 `LLMError` | 不知道有文件，不解析 JSON |
| `main.py` | 编排 + 命令行参数 + JSON 兜底解析 + 落盘 | —— |

依赖是**单向**的：`main → 三者`，`llm_client → config`。没有回指箭头，
所以 `loader` 可以脱离整个项目单独测试，`llm_client` 换模型时不碰 `loader`。

## 退出码

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 1 | 配置缺失（如没配 `DASHSCOPE_API_KEY`） |
| 2 | 文件不存在或不是文件 |
| 3 | 文件格式不支持 / 无法解码 |
| 4 | 模型调用失败 |
| 5 | 模型输出不是合法 JSON |

批量处理时可以直接用退出码判断失败原因，不必翻日志。

## 日志

运行日志同时输出到屏幕和 `logs/YYYY-MM-DD.log`。

- `INFO`：正常流程
- `WARNING`：某次调用失败，正在重试
- `ERROR`：某一步彻底失败

**重试只针对可恢复的错误**：网络中断、超时、限流（429）、服务端 5xx。
请求本身有问题（400 / 401 / 403 / 404）不会重试——重试多少次结果都一样。

## 已知限制

- **不支持扫描件 PDF**（整页是图片，没有文字层）。检测到会明确报错，不会静默产出空结果。
- **PDF 抽出的文本会有硬换行和多余空格**：它按页面位置还原文字，不是按语义，长句可能被切断。
- **整份文档一次性发给模型**：3 万字以上的文档成本更高，摘要也更容易丢细节。
  后续版本会用切分 + 检索来解决，而不是全量塞入。
- 提示词模板是中文场景专用的。

## 开发

```bash
uv run pytest                              # 跑测试
uv run python -m llm_cli parse data/a.txt  # 不依赖安装好的命令，直接跑模块
```

项目使用 `src` 布局，包的入口在 `src/llm_cli/`，命令入口定义在 `pyproject.toml` 的
`[project.scripts]` 里。
