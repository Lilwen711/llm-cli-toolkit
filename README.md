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
