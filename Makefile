.PHONY: dev test eval seed eval-agent kb-preview kb-build kb-vectorize kb-mine kb-reset eval-retrieval seed-conv eval-mining smoke-rag eval-rag eval-check judge-check

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

# ch03 知识库:文档切块 → pending;再向量化进 Milvus
kb-preview:
	PYTHONPATH=. uv run python scripts/preview_kb.py

kb-build:
	PYTHONPATH=. uv run python scripts/build_kb.py

kb-vectorize:
	PYTHONPATH=. uv run python scripts/vectorize_kb.py

kb-mine:
	PYTHONPATH=. uv run python scripts/mine_knowledge.py

# 清空知识库(TRUNCATE knowledge_chunks + drop Milvus 集合),重建前用
kb-reset:
	PYTHONPATH=. uv run python scripts/reset_kb.py

eval-retrieval:
	PYTHONPATH=. uv run python scripts/eval_retrieval.py

# 合成历史对话种子(喂挖知识 job)
seed-conv:
	uv run python scripts/seed_db.py sql/ch03-seed.sql

eval-mining:
	PYTHONPATH=. uv run python scripts/eval_mining.py

# ch04 fail-fast:Milvus Lite BM25 + 百炼 rerank
smoke-rag:
	PYTHONPATH=. uv run python scripts/smoke_milvus_bm25.py
	PYTHONPATH=. uv run python scripts/smoke_rerank.py

eval-rag:
	PYTHONPATH=. uv run python -u scripts/eval_ch04.py

eval-check:
	PYTHONPATH=. uv run python scripts/validate_eval_ch04.py

judge-check:
	PYTHONPATH=. uv run python scripts/judge_check.py
