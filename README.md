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
