# MewHelp — 电商智能客服(Ch01:纯对话)

FastAPI + LangChain,模型上游直连。

## 启动

    cp .env.example .env   # 填入 CHAT_API_KEY(上游真实 key)
    bash scripts/dev.sh    # :8000 应用(Windows 未装 make,make dev 不可用;PORT=x 可换端口)

应用内预览走 `.claude/launch.json`(8002,带 `--reload`),与命令行入口等价,只是端口不同。

## 验收

> Git Bash / GBK 终端里直接 `curl -d '{"message":"中文"}'` 会被终端转码成非 UTF-8,服务返回 `400 There was an error parsing the body`。含中文的请求用下面的 `uv run python`(httpx 按 UTF-8 编码),或把 JSON 存成 UTF-8 文件再用 `curl --data-binary @body.json`。

```bash
# 1. 流式对话(逐 token;ch02 起请求体为 user_id/conversation_id)
uv run python - <<'PY'
import httpx
with httpx.stream("POST", "http://localhost:8000/api/chat",
                  json={"user_id": "u1", "message": "你们卖猫粮吗?"},
                  timeout=120) as r:
    for line in r.iter_lines():
        print(line)
PY

# 2. 两轮上下文
bash scripts/demo_chat.sh

# 3. 结构化提取
uv run python - <<'PY'
import httpx
r = httpx.post("http://localhost:8000/api/extract",
               json={"text": "订单 MH20260701123 的猫爬架散架了,我要退款"},
               timeout=120)
print(r.json())
PY
```

## 测试

    uv run pytest -v              # 单测(不打真实模型)
    uv run python scripts/eval_extract.py   # 标注样例评估(需服务运行中)

---

## Ch02:Function Calling 工具链

在 ch01 聊天入口上装「查数据」能力:模型经 Function Calling 自选工具(订单/商品/物流 mock、FAQ 查表、建工单),后端单轮执行回灌,最终回答 SSE 逐 token 流式,聊天页气泡显示工具徽章。

### 数据库(本机 MySQL 8.0;仓库也带 docker-compose.yml)

本机开发用已有 MySQL 服务,库名 `mewhelp_ch02`(隔离,不动已有数据):

    mysql -uroot -p --default-character-set=utf8mb4 -e "CREATE DATABASE IF NOT EXISTS mewhelp_ch02 CHARACTER SET utf8mb4"
    mysql -uroot -p --default-character-set=utf8mb4 mewhelp_ch02 < sql/ch02-ddl.sql
    # .env 里 DATABASE_URL / TEST_DATABASE_URL 指向 localhost:3306/mewhelp_ch02 与 mewhelp_test
    uv run python scripts/seed_db.py   # faq 种子,幂等(也可 make seed)

有 Docker 的机器:`docker compose up -d` 首启按序执行 `sql/ch02-ddl.sql`、`sql/ch02-seed.sql`。

### 接口

- POST `/api/chat`(SSE,前端主入口):请求 `{user_id, message, conversation_id?}`;帧 `tool` / `delta` / `done` / `event: error` / `[DONE]`
- POST `/api/agent`(JSON,程序化/测试出口):一次性返回 `tool_calls` + `tool_results` + `answer`

### 验收

    bash scripts/dev.sh                       # :8000(PORT 可换)
    BASE_URL=http://localhost:8000 uv run python scripts/eval_agent.py   # 标注样例评估
    BASE_URL=http://localhost:8000 bash scripts/demo_agent.sh            # 三条验收走 /api/agent

浏览器打开聊天页:问「订单 1001 的物流到哪了」「退货政策是什么」能看到工具徽章并基于结果作答;「邮费是多少」是**预期**的关键词漏召回(答案在「运费怎么算」条目里),留给 ch03 向量检索升级。

### 测试

    uv run pytest -v   # 需 MySQL 在跑;conftest 自动重建 mewhelp_test 库并从 sql/ch02-ddl.sql 建表

---

## Ch03:知识库向量语义检索(RAG 基础)

把 `query_faq` 的内部实现从关键词查表升级为 dense 向量检索(阿里云百炼 `text-embedding-v4` + Milvus Lite),工具入参出参契约不变;MySQL 存原文权威源并记录双写状态,离线建库与录入页都在仓库里。

### 配置

`.env` 增加嵌入那一组(密钥只存本地):

    EMBED_BASE_URL=https://maas.qianwenaiapi.com/compatible-mode/v1
    EMBED_MODEL=text-embedding-v4
    EMBED_API_KEY=sk-xxx

数据库:在已建好的 `mewhelp_ch02` 库上补 ch03 两张表:

    mysql -uroot -p --default-character-set=utf8mb4 mewhelp_ch02 < sql/ch03-ddl.sql

脚本方式(无 mysql CLI 时):用 SQLAlchemy 执行 `sql/ch03-ddl.sql`,语句切分复用 `app/db/sql_script.py`。

### 建库与检索

    make kb-preview        # dry-run:按材料清单切块并打印各文件结果
    make kb-build          # data/kb/*.md → knowledge_chunks(pending)
    make kb-vectorize      # pending → 嵌入 → Milvus,回填 done(幂等可重跑)
    make eval-retrieval    # 换说法召回评估(验收 1)
    make seed-conv         # 合成历史客服对话(幂等)
    make kb-mine           # 对话挖 QA → staging → 去重 → pending
    make eval-mining       # 抽取质量评估
    make kb-reset          # 清空 knowledge_chunks + drop Milvus 集合(重建前用;需先停服务)

> 作业按钮与 Makefile 是同一份配方;本机若没有 `make`,先安装 GNU make(`winget install --id ezwinports.make -e`)。
> Milvus Lite 对数据目录是进程级独占:服务跑着时,**向量化与清库走页面/API**(`/api/kb/vectorize`、`/api/kb/reset`),终端 `make kb-vectorize` / `make kb-reset` 需先停 dev server;`kb-preview / kb-build / kb-mine / seed-conv` 不受影响。

### 页面

- `/` 聊天页:问「邮费是多少」会触发 `query_faq`,召回运费条款并作答(验收 1)
- `/kb` 知识库录入:库存闸条、手工录入与切块预览、材料清单、挖知识暂存、向量化补 pending、检索自测、最近入库
- `/admin` 后台聚合首页:四张卡,某块依赖不可用只让自己显示「读数失败」

### 验收

1. **换说法召回**:`make eval-retrieval` 5/5;`/kb` ⑤ 检索自测「邮费是多少」Top-1 为「运费怎么算」;聊天页问同一句能看到工具徽章与包邮规则。
2. **中断重跑**:`tests/test_dualwrite.py::test_vectorize_resumes_after_crash`(注入第 2 批嵌入失败 → 重跑只捡 pending);页面 ④ 区「向量化待补块」与闸条三数对齐。
3. **建库全链**:`/kb` ① 贴一段正文 → 切块预览 → 入库并向量化 → 重录同一段显示「跳过重复」;`kb-preview / kb-build / kb-mine / kb-vectorize` 在页面上按得动,日志尾可读。

### 测试

    uv run pytest -q   # 需 MySQL 在跑;conftest 从 ch02/ch03 DDL 重建 mewhelp_test,单测不打真实嵌入 API
