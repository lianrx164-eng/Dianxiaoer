.PHONY: dev test eval seed eval-agent

dev:
	./scripts/dev.sh

test:
	uv run pytest -v

eval:
	uv run python scripts/eval_extract.py

# 幂等;走 DATABASE_URL,不改 .env 也适用于 docker 起的库
seed:
	uv run python scripts/seed_db.py

# 标注样例评估(需服务在跑;BASE_URL 改端口)
eval-agent:
	uv run python scripts/eval_agent.py
