# MewHelp — 电商智能客服(Ch01:纯对话)

FastAPI + LangChain,模型上游直连。

## 启动

    cp .env.example .env   # 填入 CHAT_API_KEY(上游真实 key)
    bash scripts/dev.sh    # :8000 应用(Windows 未装 make,make dev 不可用;PORT=x 可换端口)

应用内预览走 `.claude/launch.json`(8002,带 `--reload`),与命令行入口等价,只是端口不同。

## 验收

> Git Bash / GBK 终端里直接 `curl -d '{"message":"中文"}'` 会被终端转码成非 UTF-8,服务返回 `400 There was an error parsing the body`。含中文的请求用下面的 `uv run python`(httpx 按 UTF-8 编码),或把 JSON 存成 UTF-8 文件再用 `curl --data-binary @body.json`。

```bash
# 1. 流式对话(逐 token)
uv run python - <<'PY'
import httpx
with httpx.stream("POST", "http://localhost:8000/api/chat",
                  json={"session_id": "s1", "message": "你们卖猫粮吗?"},
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
