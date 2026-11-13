#!/usr/bin/env bash
# 验收:两轮流式对话,第二轮须接住第一轮上下文(ch02 起带工具链)。
# 用 httpx 发送,避开 Git Bash/GBK 终端对 curl -d 中文的转码问题(会 400)。
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="${BASE_URL:-http://localhost:8000}"
export PYTHONUTF8=1

uv run python - "$BASE" <<'PY'
import json
import sys
import uuid

import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

base = sys.argv[1]
user_id = f"demo-{uuid.uuid4().hex[:6]}"
conversation_id = None
turns = [
    "我叫王小明,昨天买了你们的智能猫砂盆",
    "还记得我叫什么、买了什么吗?",
]

for i, msg in enumerate(turns, 1):
    print(f"=== 第{i}轮: {msg} ===")
    body = {"user_id": user_id, "message": msg, "conversation_id": conversation_id}
    with httpx.stream("POST", f"{base}/api/chat", json=body, timeout=120) as r:
        r.raise_for_status()
        event = "message"
        tools = []
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[len("event: "):]
                continue
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event == "error":
                print(f"\n[错误] {ev.get('message')}")
                sys.exit(1)
            if ev.get("event") == "tool":
                tools.append(ev["name"])
            elif ev.get("event") == "done":
                conversation_id = ev["conversation_id"]
            elif isinstance(ev.get("delta"), str):
                print(ev["delta"], end="", flush=True)
    if tools:
        print(f"\n[本轮工具: {', '.join(tools)}]")
    print()
print(f"conversation_id={conversation_id}")
PY
