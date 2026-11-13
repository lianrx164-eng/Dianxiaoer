#!/usr/bin/env bash
# 验收:两轮流式对话,第二轮须接住第一轮上下文。
# 用 httpx 发送,避开 Git Bash/GBK 终端对 curl -d 中文的转码问题(会 400)。
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="${BASE_URL:-http://localhost:8000}"

uv run python - "$BASE" <<'PY'
import json
import sys
import uuid

import httpx

base = sys.argv[1]
sid = f"demo-{uuid.uuid4().hex[:6]}"
turns = [
    "我叫王小明,昨天买了你们的智能猫砂盆",
    "还记得我叫什么、买了什么吗?",
]

for i, msg in enumerate(turns, 1):
    print(f"=== 第{i}轮: {msg} ===")
    with httpx.stream(
        "POST",
        f"{base}/api/chat",
        json={"session_id": sid, "message": msg},
        timeout=120,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload).get("delta", "")
            except json.JSONDecodeError:
                continue
            print(delta, end="", flush=True)
    print()
PY
