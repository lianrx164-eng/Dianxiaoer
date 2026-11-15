# 店小二 Dianxiaoer

> 基于 LangChain / LangGraph 的电商售后智能客服:意图分流 + RAG 知识问答 + 确定性退款流程,判不准的稳妥转人工

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-orange.svg)](https://www.langchain.com/)
[![Milvus](https://img.shields.io/badge/Milvus_Lite-3.2-blue.svg)](https://milvus.io/)
[![MCP](https://img.shields.io/badge/MCP-Model_Context_Protocol-black.svg)](https://modelcontextprotocol.io/)
[![ONNX](https://img.shields.io/badge/ONNX_Runtime-1.32-purple.svg)](https://onnxruntime.ai/)

一个可本地运行的后台客服系统:用户问退换货政策、催物流、要退款时,系统先判断意图,再决定走确定性流程、知识检索还是 Agent;证据不足或置信度低就转人工,不硬答。

> **说明**
>
> - 内置的物流 / 售后两个 MCP 服务使用**模拟数据**(订单、物流轨迹、保修、退货状态),用于在没有真实后台时跑通全流程;换成真实系统只需改 `MCP_LOGISTICS_URL` / `MCP_AFTERSALES_URL`。
> - 业务数据存 MySQL 8.0,向量库用 **Milvus Lite**(本地单文件,不需要 Docker);链路追踪 Phoenix 是可选的,不启动时相关代码自动跳过。
> - 上游模型走 OpenAI 兼容端点:对话用 `CHAT_*`,嵌入用 `EMBED_*`,重排用 `RERANK_*`。换供应商只改 `.env`,不动代码。

## ✨ 功能特性

- **意图分流**:八类业务意图 + 「其他」,分流到确定性流程、知识检索、Agent 三条路,省掉不必要的模型往返
- **知识问答**:BM25 + 向量混合召回 → 重排 → 取 Top-K;知识库外的问题明确拒答
- **业务工具**:订单、物流、保修、退货状态查询,后端以 MCP Server 独立进程提供;写操作(建工单、退款)需确认令牌
- **退款流程**:LangGraph `interrupt` 暂停等用户点选订单并确认,`Command(resume)` 续跑,涉及钱的动作模型没有自主权
- **兜底转人工**:证据置信度闸,阈值在 300 题评估集上校准;低置信度问题落池并给出转人工动作
- **低置信度飞轮**:落池 → 标准化 + 全表查重 + occurrence 累加 → 人工审核 → 写回知识库,下一问即可命中
- **主题归类**:17 类多标签分类器(微调 RoBERTa + ONNX),离线统计问题集中在哪块知识,指导补什么
- **运维后台**:`/admin` 聚合各模块关键数,`/acceptance` 九道验收闸各自带重跑按钮,依赖挂掉只影响自己那张卡

## 🏗️ 架构

```
用户 / 浏览器(app/static/)
        │  POST /api/chat,SSE 帧:tool / delta / done / error
        ▼
FastAPI 路由  /api/chat · /api/agent · /api/kb/* · /api/review/* · /api/topics/* · /api/acceptance/*
        │
        ▼
LangGraph 编排 —— 意图分流(8 类 + 其他)
        │
        ├── 确定性 Workflow    政策判定、订单查询、退款、建工单(条件写死在代码)
        ├── 知识检索           hybrid 召回 → rerank → 组装证据
        └── Agent ReAct        开放问题,工具自选,步数封顶
        │
        ▼
治理层  置信度闸 · 权限门 · 工具审计 · interrupt 确认
        │
        ├── MySQL 8.0          业务数据、飞轮队列、主题归类
        ├── Milvus Lite        知识块向量
        └── MCP :8101/:8102    物流与售后(mock 数据)
        │
        ▼
Phoenix(OTel span 全链路)+ 按意图聚合的 token 成本账
```

**技术栈**

| 层次 | 选型 |
|------|------|
| Web 框架 | FastAPI + SSE |
| 编排 | LangGraph(意图路由 / interrupt 确认 / SQLite checkpoint) |
| LLM / Embedding | OpenAI 兼容端点(DeepSeek 对话 + 百炼 `text-embedding-v4`、`qwen3-rerank`) |
| 向量库 | Milvus Lite(本地单文件,无需 Docker) |
| 业务库 | MySQL 8.0 + SQLAlchemy 2 async |
| 工具协议 | MCP(FastMCP,streamable-http) |
| 主题分类 | RoBERTa-wwm-ext 全参微调 → ONNX Runtime 独立服务 |
| 可观测 | Phoenix(OpenTelemetry 全链路 span) |

## 🚀 快速开始

### 前置要求

| 依赖 | 要求 |
|------|------|
| Python | 3.12 |
| MySQL | 8.0(业务库与测试库) |
| uv | 推荐,依赖锁定在 `uv.lock` |
| 上游 API Key | 对话与嵌入各一组,均为 OpenAI 兼容端点 |
| NVIDIA GPU(可选) | 只有重训主题分类器时需要;只做推理则不需要 torch |

### 1. 获取代码并安装依赖

```bash
git clone https://github.com/lianrx164-eng/dianxiaoer.git
cd dianxiaoer

uv sync                  # 运行依赖
uv sync --group ml       # 可选:训练/导出分类器时才需要(torch、transformers、onnx)
```

### 2. 配置环境变量

```bash
cp .env.example .env     # Windows: copy .env.example .env
# 至少填写 CHAT_API_KEY、EMBED_API_KEY,并确认 DATABASE_URL 指向本机 MySQL
```

配置项见下方「⚙️ 配置说明」。

### 3. 建库并灌种子数据

```bash
mysql -uroot -p --default-character-set=utf8mb4 -e \
  "CREATE DATABASE IF NOT EXISTS dianxiaoer_ch02 CHARACTER SET utf8mb4"

for f in ch02 ch03 ch04 ch06-ticket-type ch07 ch08 ch09 ch10; do
  mysql -uroot -p --default-character-set=utf8mb4 dianxiaoer_ch02 < "sql/${f}-ddl.sql"
done

uv run python scripts/seed_db.py      # FAQ 种子,幂等(等价于 make seed)
```

### 4. 启动

```bash
make dev        # 起物流/售后 MCP(:8101/:8102)+ FastAPI(:8000)
```

`make dev` 等价于 `scripts/dev.sh`:先 `make mcp-up`,再 `uv run uvicorn app.main:app --port 8000`。端口可用 `PORT=8002 bash scripts/dev.sh` 覆盖。

可观测与分类器(可选,各自独立进程):

```bash
make phoenix-up     # 链路追踪 :6006
make classifier-up  # 主题分类器 :8110(需先 make ch10-export 产出 ONNX)
```

### 5. 访问服务

| 服务 | 地址 |
|------|------|
| Web 聊天界面 | http://localhost:8000 |
| 后台聚合首页 | http://localhost:8000/admin |
| 分类器验收(九道闸) | http://localhost:8000/acceptance |
| API 文档(Swagger) | http://localhost:8000/docs |
| 分类器 | http://localhost:8110/healthz |
| MCP 物流 / 售后 | http://localhost:8101/mcp、http://localhost:8102/mcp |
| Phoenix | http://localhost:6006 |

## ⚙️ 配置说明

配置通过 `.env` 加载(Pydantic Settings,变量名不区分大小写),未填写的项使用默认值:

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CHAT_MODEL` / `CHAT_BASE_URL` / `CHAT_API_KEY` | 空 | **必填**,对话模型与上游地址 |
| `EXTRACT_THINKING` | 空 | 结构化任务的上游思考开关;DeepSeek 需设 `disabled` |
| `EMBED_BASE_URL` / `EMBED_MODEL` / `EMBED_API_KEY` | 百炼 / `text-embedding-v4` / 空 | **`EMBED_API_KEY` 必填** |
| `RERANK_URL` / `RERANK_MODEL` / `RERANK_API_KEY` | 百炼 `qwen3-rerank` / 空 | 不填 key 时复用 `EMBED_API_KEY` |
| `DATABASE_URL` | `mysql+asyncmy://root:root@localhost:3306/dianxiaoer_ch02` | 业务库 |
| `TEST_DATABASE_URL` | `mysql+asyncmy://root:root@localhost:3306/dianxiaoer_test` | 测试库,名称必须含 `test`(会被 DROP 重建) |
| `MILVUS_URI` | `data/milvus_knowledge.db` | Milvus Lite 单文件 |
| `RETRIEVAL_TOP_K` / `RECALL_TOP_K` / `RERANK_TOP_K` | `3` / `50` / `10` | 召回与重排数量 |
| `EVIDENCE_CONFIDENCE_THRESHOLD` | `0.59` | 置信度闸,由 `make calibrate-confidence` 校准后回填 |
| `MAX_AGENT_STEPS` | `6` | Agent ReAct 最大步数,超了走兜底 |
| `MCP_LOGISTICS_URL` / `MCP_AFTERSALES_URL` | `:8101` / `:8102` | 两个 MCP 后端 |
| `TOOL_MAX_RETRIES` / `TOOL_DEFAULT_TIMEOUT` | `2` / `5.0` | 工具重试与超时 |
| `PHOENIX_ENDPOINT` | 空 | 留空则完全不启用链路追踪 |
| `INTENT_MODEL` / `INTENT_MODE` | 空 / `accuracy` | 意图识别模型;空则复用 `CHAT_MODEL` |

> **注意**:`TEST_DATABASE_URL` 会被测试夹具 **DROP 并重建**,所以库名必须包含 `test`,且不要指向业务库。`conftest.py` 在启动时会校验这一点。

## 📡 API 接口

| 功能 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 流式对话 | POST | `/api/chat` | SSE 主入口,帧类型 `tool` / `delta` / `done` / `error` |
| 程序化对话 | POST | `/api/agent` | 一次性返回 `tool_calls` + `tool_results` + `answer` |
| 结构化提取 | POST | `/api/extract` | 从一句话抽出售后工单字段 |
| 续跑中断 | POST | `/api/actions/resume` | 订单选择 / 退款确认后继续流程 |
| 建工单 / 退款 | POST | `/api/actions/create-ticket`、`/api/actions/create-refund` | 需确认令牌 |
| 知识库管理 | GET/POST | `/api/kb/overview`、`/api/kb/ingest`、`/api/kb/vectorize`、`/api/kb/search` | 入库、向量化、检索 |
| 飞轮审核 | GET/POST | `/api/review/queue`、`/api/review/{id}/approve`、`/api/review/{id}/reject` | 审核通过即写回知识库 |
| 主题分布 | GET | `/api/topics/distribution`、`/api/topics/questions` | 17 类计数与类目下问题列表 |
| 分类器验收 | GET/POST | `/api/acceptance/overview`、`/api/acceptance/classify` | 九闸判定与单句试分类 |
| 作业管理 | GET/POST | `/api/jobs`、`/api/jobs/{name}` | 后台作业的起停与日志尾 |
| 后台聚合 | GET | `/api/admin/overview` | 各模块一张卡的关键数 |
| 会话查询 | GET | `/api/conversations`、`/api/conversations/{id}/messages` | 历史会话 |

### 调用示例

> Git Bash / GBK 终端里直接 `curl -d '{"message":"中文"}'` 会被转码成非 UTF-8,服务返回 `400 There was an error parsing the body`。含中文的请求用下面的 `uv run python`,或把 JSON 存成 UTF-8 文件再 `curl --data-binary @body.json`。

```bash
# 流式对话
uv run python - <<'PY'
import httpx
with httpx.stream("POST", "http://localhost:8000/api/chat",
                  json={"user_id": "u1", "message": "订单 1001 的猫爬架散架了,我想退"},
                  timeout=120) as r:
    for line in r.iter_lines():
        print(line)
PY
```

```bash
# 程序化对话(带工具调用结果)
uv run python -c "
import httpx
r = httpx.post('http://localhost:8000/api/agent',
               json={'user_id':'u1','message':'退货政策是什么'}, timeout=120)
print(r.json())"
```

```bash
# 单句试分类(需分类器在跑)
uv run python -c "
import httpx
print(httpx.post('http://localhost:8000/api/acceptance/classify',
                 json={'texts':['猫窝买大了想退']}, timeout=60).json())"
```

## 🎯 一条消息的旅程

```
用户消息
   │
   ▼
1. 意图识别(八类 + 其他)                    → 命中确定类 → 走对应流程
2. 三条执行路                               → SSE: tool / delta
   ├─ 确定性 Workflow  政策判定、订单查询、退款(条件写死在代码)
   ├─ 知识检索         hybrid 召回 → rerank → 组装证据
   └─ Agent ReAct      开放问题,工具自选,步数封顶
3. 治理                                     
   ├─ 写操作 → interrupt 等用户确认 → Command(resume) 续跑
   └─ 证据置信度 < 阈值 → 转人工,问题落池
4. 收尾                                     → SSE: done
```

- 意图分流放在 Agent 之前:售后问法大部分是确定的,先分类能省一次模型往返,行为也可预测
- 业务铁律写死在代码而不是 prompt:退款条件、金额、政策条款不能赌模型记性
- 判不准就转人工:转人工是可接受的失败,编答案不是

## 🤖 主题分类器(可选)

17 类多标签,用于统计低置信度问题集中在哪块知识。全流程可复现:

```bash
make ch10-golden          # 预标 prompt 黄金样例验证(≥80% 才放行)
make ch10-corpus          # 捞池 → 清洗脱敏 → 预标 → 模拟补足 → 人工抽审
make ch10-dataset         # 分层划分 80/10/10 + 仅训练集增强
make ch10-train           # RoBERTa-wwm-ext 全参微调
make ch10-eval            # 每类 P/R/F1 + 混淆矩阵 + 严档红线
make ch10-export          # 导出 ONNX,校验与 torch 预测逐条一致
make classifier-up        # 起 ONNX 推理服务 :8110
make classify-pool        # 攒够一批再离线归类,写 topic_classifications
```

分类器**走旁路**:不进实时对话链路,实时路径的延迟和可用性不依赖它。

## 📁 项目结构

(仅列主要文件)

```
app/
├── main.py                 # FastAPI 入口,注册 14 个路由与静态页
├── config.py               # Pydantic Settings
├── api/                    # chat / agent / kb / review / topics / acceptance / jobs ...
├── core/                   # 意图、置信度、记忆、上下文预算、作业运行器、taxonomy
├── graph/                  # LangGraph:状态、节点、路由、运行时
├── tools/                  # 工具注册表、执行引擎、MCP 客户端、内置工具
├── kb/                     # 分块、嵌入、Milvus、双写、去重
├── db/                     # ORM 模型与仓储层
└── static/                 # 后台页面(纯 HTML/CSS/JS,无构建步骤)
scripts/                    # 各章评估脚本 + ch10 训练流水线
mcp_servers/                # 物流 / 售后 MCP(模拟数据)
sql/                        # 各章 DDL 与种子
tests/                      # 单测与评估夹具
docs/                       # 设计文档、实施计划、开发记录
dev-notes/                  # 每一步的决策、返工与验收记录
```

运行时产物:`data/milvus_knowledge.db`(向量库)、`data/*.sqlite`(checkpoint)、`data/ch10/`(语料与模型)、`log/`、`*.pid`,均已在 `.gitignore` 中排除。

## 🛠️ 开发

```bash
make test             # uv run pytest -v(需要 MySQL 在跑)
uv sync --group ml    # 需要跑训练/导出时
```

| 命令 | 作用 |
|------|------|
| `make dev` | 起 MCP + FastAPI(:8000) |
| `make test` | 全量单测(自动重建 `dianxiaoer_test` 库) |
| `make seed` | 灌 FAQ 种子,幂等 |
| `make eval-rag` | 检索 + 生成评估(300 题) |
| `make calibrate-confidence` | 在评估集上校准置信度阈值,结果回填 `.env` |
| `make flywheel` | 跑一轮飞轮:捞取 → 归并 → 入审核队列 |
| `make mcp-up` / `mcp-down` | 起停两个 MCP 后端 |
| `make phoenix-up` / `phoenix-down` | 起停链路追踪 |

## 🐛 常见问题

**1. 含中文的请求返回 400 "There was an error parsing the body"**

终端把 UTF-8 转成了 GBK。改用 `uv run python` + `httpx`(见上面的调用示例),或把 JSON 存成 UTF-8 文件后 `curl --data-binary @body.json`。

**2. 端口被占用(8000 / 8101 / 8102 / 8110 / 6006)**

```bash
netstat -ano | grep :8101
taskkill //PID <PID> //T //F     # Windows;Linux/macOS 用 kill
```

`make classifier-down` 会先 `kill` 再 `taskkill /T /F`,能连带子进程一起停。

**3. `git push` 报 `Connection was reset`**

本机直连 GitHub 被重置。给仓库配代理即可:

```bash
git config http.proxy http://127.0.0.1:7890
git config https.proxy http://127.0.0.1:7890
```

**4. 首次训练下载模型卡住**

Hugging Face 直连超时。`make ch10-train` 默认带 `HF_ENDPOINT=https://hf-mirror.com`,手工跑脚本时自己加:

```bash
HF_ENDPOINT=https://hf-mirror.com uv run --group ml python scripts/ch10/train.py
```

**5. `uv run` 卡住不动 / 提示环境被锁**

同一时间只能有一个 uv 在改虚拟环境。等 `uv sync` 或后台作业结束后再跑;实在卡住就重开终端。

**6. ONNX 导出报 logits 超容差**

`torch.onnx.export` 的 TorchScript 导出器会错译带 padding 的注意力。本项目用 `dynamo=True`,需要 `onnxscript`(已在 `ml` 组里);缺它会静默回退到旧导出器。

**7. 测试数据库被删了**

这是设计如此:`conftest.py` 每次跑测试都会 DROP 并重建 `dianxiaoer_test`。所以 `TEST_DATABASE_URL` 的库名必须含 `test`,不要指向业务库。

## ⚠️ 已知限制

- 单机部署,后台页面没有鉴权,只适合内网或演示环境
- 没有 CI,测试靠本地 `pytest`(410 个用例)
- 内置 MCP 服务为演示用途:订单、物流、保修、退货状态都是模拟数据,替换为真实系统后才有生产价值
- 主题分类器语料是真实池样本 + 模拟补量(每类约 100 条),不等于线上分布;严档里「尺码」「退换货」两类未过 0.9 红线(分别为 0.783、0.895),原因与错例见 `data/ch10/reports/eval_report.md`
- 分类器走旁路批处理,归类结果有延迟,不适合需要秒级一致性的场景
- 上游模型的能力与价格随供应商变化,成本账按当前价格估算
- 没有多租户与细粒度权限模型

## 📚 参考资源

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [LangChain 文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [Milvus 文档](https://milvus.io/docs)
- [MCP 协议](https://modelcontextprotocol.io/)
- [Phoenix 可观测](https://docs.arize.com/phoenix)
