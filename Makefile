.PHONY: dev test eval seed eval-agent kb-preview kb-build kb-vectorize kb-mine kb-reset eval-retrieval seed-conv eval-mining smoke-rag eval-rag eval-check judge-check eval-rewrite smoke-graph eval-intent eval-ch05 smoke-interrupt eval-coref eval-expand eval-ch06 eval-ch07 smoke-ch07 smoke-ch07-demo mcp-up mcp-down phoenix-up phoenix-down calibrate-confidence flywheel flywheel-samples eval-flywheel cost-report eval-ch09

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

eval-rewrite:
	PYTHONPATH=. uv run python scripts/eval_rewrite.py

# ch05 fail-fast:LangGraph 图 + checkpointer + 多模式流式
smoke-graph:
	PYTHONUTF8=1 uv run python -m scripts.smoke_langgraph

eval-intent:
	PYTHONUTF8=1 uv run python -m scripts.eval_intent

# 五验收端到端(需服务在跑;BASE_URL 改端口)
eval-ch05:
	PYTHONUTF8=1 uv run python -m scripts.eval_ch05

# ch06 fail-fast:interrupt/Command(resume)/astream 中断 surface
smoke-interrupt:
	PYTHONUTF8=1 uv run python -m scripts.smoke_interrupt

# ch06 纯 Prompt 评估集(需上游可用)
eval-coref:
	PYTHONUTF8=1 uv run python -m scripts.eval_coref

eval-expand:
	PYTHONUTF8=1 uv run python -m scripts.eval_expand

# ch06 四验收端到端(需服务在跑 + 已应用 sql/ch06-ticket-type.sql)
eval-ch06:
	PYTHONUTF8=1 uv run python -m scripts.eval_ch06

# ch08 验收样例端到端(需 make dev + make mcp-up)
eval-ch08:
	PYTHONUTF8=1 uv run python -m scripts.eval_ch08

# ch07 分段摘要 prompt 标注评估(需上游可用)
eval-ch07:
	PYTHONUTF8=1 uv run python -m scripts.eval_ch07

# ch07 端到端冒烟:默认 22 轮不压(独立 checkpointer,避免与预览服务抢 SQLite 锁)
smoke-ch07:
	CHECKPOINTER_DB_PATH=data/ch05_checkpoints_smoke.sqlite PYTHONUTF8=1 uv run python -m scripts.smoke_ch07 default

# ch07 演示配置 30 轮级联(层1降级 → 摘要落库;25 轮层2约 1515<1695,30 轮过阈值)
smoke-ch07-demo:
	MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5 CHECKPOINTER_DB_PATH=data/ch05_checkpoints_smoke.sqlite PYTHONUTF8=1 uv run python -m scripts.smoke_ch07 demo

# ch08: 两台业务 MCP Server 起停(独立进程,Streamable HTTP :8101/:8102)
mcp-up:
	@mkdir -p log data
	@nohup uv run python mcp_servers/logistics_server.py  > log/mcp-logistics.log 2>&1 & echo $$! > data/mcp-logistics.pid
	@nohup uv run python mcp_servers/aftersales_server.py > log/mcp-aftersales.log 2>&1 & echo $$! > data/mcp-aftersales.pid
	@sleep 1 && echo "MCP Servers 已拉起: logistics=:8101 aftersales=:8102"

mcp-down:
	-@kill `cat data/mcp-logistics.pid 2>/dev/null` 2>/dev/null; rm -f data/mcp-logistics.pid
	-@kill `cat data/mcp-aftersales.pid 2>/dev/null` 2>/dev/null; rm -f data/mcp-aftersales.pid
	@echo "MCP Servers 已停"

# ch09: Phoenix 本地观测服务(SQLite 存储,不需要 Docker)
PHOENIX_WORKING_DIR ?= data/phoenix

phoenix-up:
	@mkdir -p log $(PHOENIX_WORKING_DIR)
	@nohup env PHOENIX_WORKING_DIR=$(PHOENIX_WORKING_DIR) uv run phoenix serve > log/phoenix.log 2>&1 & echo $$! > data/phoenix.pid
	@sleep 2 && echo "Phoenix 起中: http://localhost:6006 (数据 $(PHOENIX_WORKING_DIR)/phoenix.db)"

phoenix-down:
	-@kill `cat data/phoenix.pid 2>/dev/null` 2>/dev/null; rm -f data/phoenix.pid
	@echo "Phoenix 已停"

# ch09 置信度阈值校准(需 Milvus + 上游可用 + 知识库已建;跑前先停 dev 释放 Milvus Lite)
calibrate-confidence:
	PYTHONPATH=. uv run python scripts/calibrate_confidence.py

# ch09 飞轮批处理:问题池 → 标准化查重 → 待审队列(需 mysql + 上游可用)
flywheel:
	PYTHONPATH=. uv run python scripts/flywheel_pipeline.py

# ch09 标准化查重 prompt 标注样例验证(通过率 ≥ 80%)
flywheel-samples:
	PYTHONPATH=. uv run python scripts/validate_flywheel_samples.py

# ch09 评估流水线:复用 ch04 评估集,落 eval_runs 连趋势(TRIGGER=manual|scheduled)
eval-flywheel:
	PYTHONPATH=. uv run python scripts/eval_flywheel.py --triggered-by $(or $(TRIGGER),manual)

# ch09 按意图 token 账(需 Phoenix 在跑;DAYS=窗口天数)
cost-report:
	PYTHONPATH=. uv run python scripts/cost_by_intent.py --days $(or $(DAYS),7)

# ch09 验收 2/3/4 端到端(需 make dev 在跑;BASE_URL 改端口)
eval-ch09:
	PYTHONPATH=. uv run python -m scripts.eval_ch09 --base-url $(or $(BASE_URL),http://127.0.0.1:8002)

# ch10: 主题分类器(数据→训练→评测→ONNX→旁路批量归类)
ch10-golden:  ## 预标 prompt 黄金样例验证(通过率 ≥ 80% 才放行批量预标)
	PYTHONPATH=. uv run python scripts/ch10/validate_golden.py

ch10-corpus:  ## 语料流水线:捞池→清洗→预标→模拟补足→导出人工抽审(需 mysql + 上游可用)
	PYTHONPATH=. uv run python scripts/ch10/build_corpus.py

ch10-dataset:  ## 分层划分 80/10/10 + 训练集增强(需上游可用)
	PYTHONPATH=. uv run python scripts/ch10/build_dataset.py

ch10-train:  ## RoBERTa-wwm-ext 全参微调(cuda→cpu 自适应,重依赖走 ml 组)
	HF_ENDPOINT=$(or $(HF_ENDPOINT),https://hf-mirror.com) PYTHONPATH=. uv run --group ml python scripts/ch10/train.py

ch10-eval:  ## 测试集评测:每类 P/R/F1 + 混淆矩阵 + 容错红线 + 判错样本导出
	PYTHONPATH=. uv run --group ml python scripts/ch10/evaluate.py

ch10-export:  ## 导出 ONNX 并校验与 torch 预测完全一致
	PYTHONPATH=. uv run --group ml python scripts/ch10/export_onnx.py

classifier-up:  ## ch10 推理服务 :8110(ONNX 轻运行时)
	@mkdir -p log data
	@PYTHONPATH=. nohup uv run --group serve python scripts/ch10/serve.py > log/classifier.log 2>&1 &
	@sleep 4 && curl -sf http://127.0.0.1:8110/healthz >/dev/null && echo "分类器服务已拉起: :8110(pid 见 data/classifier.pid)" || echo "启动失败,看 log/classifier.log"

classifier-down:
	-@PID=`cat data/classifier.pid 2>/dev/null`; [ -n "$$PID" ] && kill $$PID 2>/dev/null
	-@PID=`cat data/classifier.pid 2>/dev/null`; [ -n "$$PID" ] && taskkill //PID $$PID //T //F >/dev/null 2>&1
	@rm -f data/classifier.pid
	@echo "分类器服务已停"

classify-pool:  ## ch10 旁路批量归类:攒够一批归一次,写 topic_classifications(需 :8110)
	PYTHONPATH=. uv run python scripts/ch10/classify_pool.py $(if $(FORCE),--force,)

ch10-threshold-scan:  ## ch10 阈值扫描重演:九候选线各算一遍 micro-F1(需 :8110)
	PYTHONPATH=. uv run python scripts/ch10/scan_threshold_replay.py
