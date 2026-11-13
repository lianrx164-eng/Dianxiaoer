#!/usr/bin/env bash
# ch02 验收演示:三条验收标准(走 /api/agent,一眼看工具轨迹)。
# 含中文的请求走 httpx:Git Bash/GBK 终端直接 curl 中文 body 会 400。
set -euo pipefail
BASE="${BASE_URL:-http://localhost:8000}"
export BASE_URL="$BASE"
export PYTHONUTF8=1

uv run python - <<'PY'
import json
import os
import sys

import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # GBK 控制台下含 emoji 不崩

base = os.environ["BASE_URL"]
demo = [
    ("验收1:订单物流(应选中 query_logistics)", "订单 1001 的物流到哪了"),
    ("验收2:退货政策(query_faq 命中)", "退货政策是什么"),
    ("验收3:换说法(query_faq 漏召回为预期)", "邮费是多少"),
]
for title, msg in demo:
    r = httpx.post(f"{base}/api/agent", json={"user_id": "demo", "message": msg}, timeout=120)
    body = r.json()
    print("==", title, "==")
    print("tools:", [tc["name"] for tc in body["tool_calls"]])
    print("answer:", body["answer"])
    print()
PY
