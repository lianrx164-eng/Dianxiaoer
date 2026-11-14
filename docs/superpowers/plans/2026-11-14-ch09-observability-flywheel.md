# Ch09 可观测性与数据飞轮 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给客服系统接 Langfuse 全链路观测 + 按意图 token 账,并建成"三入口低置信度问题池 → 标准化查重待审队列 → 人工审核写回知识库"的数据飞轮闭环,配自动化评估趋势流水线。

**Architecture:** 方案 A"回调为主、脚本为辅"——观测靠 `compile().with_config({"callbacks":[CallbackHandler()]})` 编译时挂一次吃全图;飞轮/评估/成本统计是 make 驱动的批处理脚本;审核后台 = 静态单页 + REST。

**Tech Stack:** Langfuse v3 自部署(独立 docker compose)、`langfuse>=3.12,<4`(`langfuse.langchain.CallbackHandler`)、LangGraph、SQLAlchemy 2 async、FastAPI、复用 ch03 dualwrite / ch04 评估集 / ch05 证据管线 / ch08 工具系统。

**Spec:** `docs/superpowers/specs/2026-11-14-ch09-observability-flywheel-design.md`(§14 是仓库基线校正,与本计划冲突时以 spec §14 + 本计划为准)。

## Global Constraints

- **技术选型定死**:Langfuse 自部署(不用 LangSmith/云版);服务器镜像锁 v3,SDK 锁 `langfuse>=3.12,<4`。走不通停下来问用户。
- **API 用法先查证**:Langfuse/LangGraph/LangChain/FastAPI/SQLAlchemy 的具体 API,动手前用 Context7 核对;与实际安装版本不符时以实测为准并记 dev-notes。
- **观测可选降级**:三配置(public/secret/base_url)不齐时不挂回调、不初始化 client、所有 langfuse 调用点 guard;测试环境完全不依赖 Langfuse。
- **DDL 已定稿**:`sql/ch09-ddl.sql` 从桌面 `ch_09.sql` 拷入,表结构不改。`review_queue.review_status`、`eval_runs.triggered_by` 是中文值 ENUM。
- **查重语义**:候选 = `review_queue` 全部行(截 200);命中任何状态只 `occurrence_count+1`、状态不变;未命中新建"待审"。
- **飞轮是定时批处理**:脚本 + make,不引 APScheduler。
- **快照语义**:走了知识检索就写 state 的 `retrieved_snapshot`(精排 Top3);落池随条写 `retrieved_chunks`;没走检索为 NULL。
- **阈值不拍脑袋**:`evidence_confidence_threshold` 默认值来自校准脚本在 ch04 评估集上的实跑输出。
- **中文 JSON 不转义**:面向人看的输出一律 `ensure_ascii=False`。
- **测试命令**:`uv run pytest -q`(全量);每任务结束全量零回归再 commit。
- **Git**:commit message 简洁英文;不 push;不删除既有文件/历史。
- **过程留痕**:每完成一个阶段立即追记 `dev-notes/ch09.md`(用户原话/关键产出/纠偏/翻车),不许收尾补记。
- **前端例外**:Task 12/13 的页面走 Vibe Coding,不套 TDD,浏览器人工验收。

## Review Focus

- 无 Langfuse 配置/服务不可达时,聊天主流程必须完全正常(挂载、打标、成本脚本三处均不得抛错或阻塞)。
- 审核通过但向量化失败时,不得留下"状态已通过、知识库没写进"的半截状态(验收 3 会假绿)。
- 飞轮处理中模型返回幻觉 `matched_question_id`、坏 JSON、或同批同义问题时,不得重复建行、不得复活已驳回行、不得推进游标。
- 👎 快照回捞:被评价的问题与 checkpointer 最近一轮不一致时必须空着,不许把别轮快照硬塞进去。
- 置信度校准:两桶分布若严重重叠,不许硬选阈值,必须停下来把分布交给用户。

---

### Task 1: 飞轮地基(DDL + ORM + repository)

**Files:**
- Add: `sql/ch09-ddl.sql`(已从桌面拷入,确认内容)
- Modify: `app/db/models.py`(`ReviewQueue`、`EvalRun`;`LowConfidenceQuestion` 加两列)
- Modify: `app/db/repository.py`(`insert_low_confidence` 加快照参数 + 10 个方法)
- Modify: `tests/conftest.py`(`_DDL_FILES` 加 ch09;`_TABLES` 加两表)
- Test: `tests/db/test_ch09_flywheel_tables.py`

**Interfaces:**
- Consumes: `sql/ch09-ddl.sql`;`app.db.base` 既有 `Base/async_session`。
- Produces:
  - `class ReviewQueue`:`id, normalized_question, ai_suggested_answer, occurrence_count, review_status('待审'|'通过'|'驳回'), approved_answer, created_at, updated_at`
  - `class EvalRun`:`id, triggered_by('手动'|'定时'), dataset_size, metrics: dict, created_at`
  - `LowConfidenceQuestion.retrieved_chunks: list|None`(JSON)、`.matched_review_id: int|None`
  - `repository.insert_low_confidence(conversation_id, raw_question, source, reason, retrieved_chunks=None) -> int`
  - `fetch_unmatched_low_conf(limit) -> list[LowConfidenceQuestion]`(matched_review_id IS NULL,id 升序)
  - `list_review_candidates(limit=200) -> list[dict]`(`[{"id","normalized_question"}]`,updated_at 倒序)
  - `insert_review_item(normalized_question, ai_suggested_answer) -> int`
  - `increment_occurrence(review_id) -> None`
  - `set_matched_review(lcq_id, review_id) -> None`
  - `list_review_queue(status: str|None) -> list[ReviewQueue]`(occurrence_count 降序,同频 id 降序)
  - `get_review_detail(review_id) -> tuple[ReviewQueue, list[LowConfidenceQuestion]]|None`
  - `update_review_status(review_id, status, approved_answer=None) -> bool`(仅"待审"可流转)
  - `insert_eval_run(triggered_by, dataset_size, metrics) -> int`
  - `list_eval_runs(limit=10) -> list[EvalRun]`(id 倒序)

- [ ] **Step 1: conftest 挂 ch09 DDL**
  `_DDL_FILES` 追加 `sql/ch09-ddl.sql`;`_TABLES` 前部加 `"review_queue", "eval_runs"`(FK 检查已关,顺序只需保证重建时两表都在列表里)。

- [ ] **Step 2: 写失败测试** `tests/db/test_ch09_flywheel_tables.py`
  断言:①带/不带快照落池,`retrieved_chunks` 分别为 list / None,`matched_review_id` 初始为 None;②`insert_review_item` 后候选列表形状;③`increment_occurrence` 三次后 `occurrence_count==3`;④`set_matched_review` 后原话出现在详情、`fetch_unmatched_low_conf` 不再返回;⑤`update_review_status` 仅待审可流转,二次调用返回 False;⑥`list_review_queue` 按次数降序、状态筛选、None 全量;⑦`eval_runs` 往返、最新在前。

- [ ] **Step 3: 跑测试确认失败**
  `uv run pytest tests/db/test_ch09_flywheel_tables.py -q` → FAIL(表/类/方法不存在)。

- [ ] **Step 4: 实现 ORM + repository**
  按 `sql/ch09-ddl.sql` 映射(中文 ENUM 用 `Enum("待审","通过","驳回")`);repository 用 `async with db.async_session() as s` 既有风格;`insert_low_confidence` 新参数带默认值,保持既有调用点兼容。

- [ ] **Step 5: 跑测试 + 全量零回归 + 应用 DDL + Commit**
  `uv run pytest tests/db/test_ch09_flywheel_tables.py -q` → PASS;`uv run pytest -q` → 零回归。
  开发库:`"C:/Program Files/MySQL/MySQL Server 8.0/bin/mysql.exe" --default-character-set=utf8mb4 -uroot -p123456 mewhelp_ch02 < sql/ch09-ddl.sql`。
  ```bash
  git add sql/ch09-ddl.sql app/db/models.py app/db/repository.py tests/conftest.py tests/db/test_ch09_flywheel_tables.py
  git commit -m "feat(ch09): flywheel foundation - review_queue/eval_runs ORM and repository methods"
  ```

---

### Task 2: Langfuse 部署物 + 观测模块 + 编译时挂回调

**Files:**
- Create: `docker-compose.langfuse.yml`
- Create: `app/core/observability.py`
- Modify: `app/config.py`(3 个配置)
- Modify: `app/graph/runtime.py`(`attach_observability` + 三处 config 加 session metadata)
- Modify: `Makefile`(`langfuse-up/down`)
- Modify: `pyproject.toml`(`uv add "langfuse>=3.12,<4"`)
- Test: `tests/core/test_observability.py`

**Interfaces:**
- Produces:
  - `observability.langfuse_enabled() -> bool`
  - `observability.attach_observability(graph) -> graph`(未启用原样返回)
  - `observability.tag_intent(intent: str, confidence: float) -> None`(任何异常静默)
  - `observability.get_langfuse()`(启用返回 client,否则 None)
  - runtime 的 `run_turn/resume_turn/stream_turn/stream_resume` 的 config 含 `"metadata": {"langfuse_session_id": str(cid)}`

- [ ] **Step 1: Context7 核对 v3 SDK**
  确认 `Langfuse(public_key=..., secret_key=..., host=...)`、`from langfuse.langchain import CallbackHandler`、`langfuse.get_client().update_current_trace(metadata=..., tags=...)` 或等价 API;确认 LangChain config metadata 的 session 键(`langfuse_session_id`)。安装后用 `python -c` 反射核对签名,偏差记 dev-notes。

- [ ] **Step 2: 写失败测试** `tests/core/test_observability.py`
  ①三键缺一 → `langfuse_enabled()` False;②未启用 `attach_observability(sentinel)` 原样返回;③未启用 `tag_intent` 不抛;④启用时 `attach_observability` 调 `with_config({"callbacks":[handler]})`(monkeypatch `_init_client/_make_handler`)。

- [ ] **Step 3: 实现 `observability.py` + settings + runtime**
  全模块 guard;`_init_client` 惰性单例;`attach_observability` 不 import langfuse 当未启用。runtime 三处 config 补 metadata(未启用时多余 metadata 无害)。

- [ ] **Step 4: 测试 + 全量零回归**

- [ ] **Step 5: 部署物**
  `docker-compose.langfuse.yml` 以官方 v3 compose 为底(镜像锁 `langfuse/langfuse:3` / `langfuse/langfuse-worker:3`,端口按本机情况只改宿主侧),加 headless 初始化变量(`LANGFUSE_INIT_ORG_ID/PROJECT_PUBLIC_KEY/SECRET_KEY/USER_*`,参考 Context7 self-hosting 文档);`make langfuse-up/down`。
  `.env` 写 `LANGFUSE_PUBLIC_KEY/SECRET_KEY/BASE_URL`(值对齐 headless 预置)。

- [ ] **Step 6: 冒烟 + Commit**
  `make langfuse-up` → `curl -sf http://localhost:3000/api/public/health`;起服务发一条消息,浏览器确认 trace 树出现(验收 1 首点亮),截图记 dev-notes。
  ```bash
  git add docker-compose.langfuse.yml app/core/observability.py app/config.py app/graph/runtime.py Makefile pyproject.toml uv.lock tests/core/test_observability.py
  git commit -m "feat(ch09): Langfuse self-hosted stack and compile-time callback"
  ```

---

### Task 3: token usage 收口 + 意图打进 trace

**Files:**
- Modify: `app/core/llm.py`(`stream_usage=True`)
- Modify: `app/graph/nodes.py`(`main_agent` 去重复 bind;`classify_intent` 调 `tag_intent`)
- Test: `tests/core/test_llm_usage.py`

**Interfaces:**
- Consumes: `observability.tag_intent`。
- Produces: `get_chat_model(...)` 构造带 `stream_usage=True`;`classify_intent` 行为不变。

- [ ] **Step 1: Context7 核对 ChatOpenAI `stream_usage`**
  确认 `langchain_openai.ChatOpenAI(stream_usage=True)` 存在且等效 `stream_options.include_usage`。不存在则保留 main_agent 原 bind 并记 dev-notes。

- [ ] **Step 2: 失败测试** `tests/core/test_llm_usage.py`:流式与非流式模型 `stream_usage is True`。

- [ ] **Step 3: 实现**
  `get_chat_model` 加 `stream_usage=True`;`main_agent` 删 `.bind(stream_options=...)`;`classify_intent` 在两条 return 路径(粘滞分支与正常分支)都调用 `tag_intent(intent, conf)`。

- [ ] **Step 4: 测试 + 冒烟 + Commit**
  冒烟:一条消息的 trace metadata 有 intent、tags 有 `intent:xxx`、generation span 有 usage。
  ```bash
  git commit -m "feat(ch09): collect stream usage centrally and tag intent into trace"
  ```

---

### Task 4: `evidence_confidence` 纯函数

**Files:**
- Create: `app/core/confidence.py`
- Modify: `app/config.py`(阈值占位 0.5)
- Test: `tests/core/test_confidence.py`

**Interfaces:**
- Produces:
  - `@dataclass EvidenceConfidence: score: float; signals: dict`
  - `compute_evidence_confidence(hits: list[dict]) -> EvidenceConfidence`
  - `snapshot_from_hits(hits, top_n=3) -> list[dict]`(`question/answer/rerank_score/section_path`)
  - `settings.evidence_confidence_threshold`

- [ ] **Step 1: 失败测试** `tests/core/test_confidence.py`
  空召回 0 分且四信号为 0/False;强证据(0.95/0.40/0.35)分 > 0.7 且 margin≈0.55、key_clause_hit True;弱平证据(0.22/0.21)分 < 0.4 且 valid_count 0;单条 margin=top1;分数对 top1 单调;总分界 [0,1];`snapshot_from_hits` 取前 3 且字段齐全、空召回为空列表。

- [ ] **Step 2: 实现**
  权重常量 `W_TOP1=0.5, W_VALID=0.2, W_MARGIN=0.2, W_KEY=0.1`;`VALID_SCORE_FLOOR=0.3`、`VALID_COUNT_CAP=3`;关键条款用模块内词表(退货/退款/保修/运费/发票/售后/维修/期限/赔付 等)对 Top3 的 question+answer 做子串匹配。**不 import 不存在的 `_KEY_TERMS`**,文档注明与 ch03 `【关键】` 入库标记的口径差异。

- [ ] **Step 3: 测试 + Commit**

---

### Task 5: 置信度闸接线(state + evidence.gather_evidence + forced_rag + fallback_reply)

**Files:**
- Modify: `app/core/evidence.py`(抽 `retrieve_hits`;闸升级;返回 confidence/signals/snapshot)
- Modify: `app/graph/state.py`(3 字段)
- Modify: `app/graph/runtime.py`(`_graph_input` 重置)
- Modify: `app/graph/nodes.py`(`forced_rag` 写新字段;`fallback_reply` 用 source + 快照)
- Test: `tests/graph/test_ch09_confidence_gate.py`、`tests/core/test_evidence.py`(若无则新建)

**Interfaces:**
- Consumes: Task 4 函数;Task 1 `insert_low_confidence(..., retrieved_chunks=)`。
- Produces:
  - `evidence.retrieve_hits(query_raw, category=None) -> list[dict]`(理解 → hybrid_rerank)
  - `gather_evidence(...)` 返回追加 `confidence: float, signals: dict, snapshot: list`
  - state:`evidence_confidence: float`、`fallback_source: str`、`retrieved_snapshot: list`

- [ ] **Step 1: 失败测试**
  `tests/core/test_evidence.py`:弱证据 → `sufficient=False, source="retrieval_low_conf"`,带 confidence/signals/snapshot;强证据但 selfcheck 不过 → `source="self_check"`;强证据且 selfcheck 过 → snapshot 仍返回。
  `tests/graph/test_ch09_confidence_gate.py`:monkeypatch `evidence.gather_evidence` 返回不同组合,断言 `forced_rag` 把 `evidence_confidence/fallback_source/retrieved_snapshot` 写进 state;`fallback_reply` 落池带对 source 与 chunks,无快照存 None,缺 `fallback_source` 回落 `trace.gate` 再回落 `retrieval_low_conf`。

- [ ] **Step 2: 实现**
  `gather_evidence` 内先 `hits = await retrieve_hits(...)`,`conf = compute_evidence_confidence(hits)`,`snapshot = snapshot_from_hits(hits)`;机械闸换成 `conf.score < settings.evidence_confidence_threshold`;selfcheck 顺序不变;三条返回路径都带 `confidence/signals/snapshot`。
  `forced_rag`:不够时写 `fallback_source=r["source"]`、trace 加 `confidence_signals`;够时也写 `retrieved_snapshot`、`evidence_confidence`。
  `fallback_reply`:source 优先 `state.fallback_source`,再 `trace.gate`,再默认;reason 带分数与 signals JSON;`retrieved_chunks=state.retrieved_snapshot or None`。

- [ ] **Step 3: 测试 + 全量 + Commit**

---

### Task 6: 阈值校准脚本(评估集实跑)

**Files:**
- Create: `scripts/calibrate_confidence.py`
- Modify: `Makefile`(`calibrate-confidence`)
- Modify: `app/config.py`(回填实测值)
- Output: `data/ch09/reports/confidence_calibration.{txt,json}`(Task 13 要读)

**Interfaces:**
- Consumes: `tests/data/eval_ch04.jsonl`(A/B/C/E 可答,D 应拒);`evidence.retrieve_hits`;`compute_evidence_confidence`。

- [ ] **Step 1: 写脚本**
  逐条(并发 8)跑 `retrieve_hits(sample["query"])` → conf;打印两桶分布(min/p25/p50/p75/max);扫 t=0.05..0.95,算可答通过率、应拒放行率、Youden J,取 J 最大点;txt/json 双产物(json 含分布、扫描曲线、推荐阈值、样本数)。

- [ ] **Step 2: 实跑**
  `make calibrate-confidence`。Expected:两桶 p50 明显分离,J 有可解释的最优点。若 J < 0.3,停下,把分布表交给用户共同定夺(不许硬选)。

- [ ] **Step 3: 回填 + 测试 + Commit**
  `settings.evidence_confidence_threshold = <实测值>`,注释写校准来源;全量零回归。
  ```bash
  git commit -m "feat(ch09): calibrate evidence confidence threshold on ch04 eval set"
  ```

---

### Task 7: 👎 反馈入口(API + 快照回捞 + 前端轻接)

**Files:**
- Create: `app/api/feedback.py`、`app/schemas/feedback.py`
- Modify: `app/graph/runtime.py`(`get_turn_snapshot`)
- Modify: `app/main.py`(挂 router)
- Modify: `app/static/index.html`(👎/👍 发请求,Vibe)
- Test: `tests/test_feedback_api.py`

**Interfaces:**
- Produces:
  - `POST /api/feedback` body `{"conversation_id": int, "rating": "up"|"down", "question": str}` → `{"ok": true, "pooled": bool}`
  - `runtime.get_turn_snapshot(conversation_id) -> {"question": str, "snapshot": list}`

- [ ] **Step 1: 失败测试** `tests/test_feedback_api.py`
  up 不落库;down 落池 source=user_feedback,问题匹配时带快照;不匹配/回捞抛错时 chunks=None 且仍 200;rating 非法 422。

- [ ] **Step 2: 实现 runtime 回捞**
  `aget_state` 取 `messages` 最后一条 HumanMessage 与 `retrieved_snapshot`;空白归一化后比对。

- [ ] **Step 3: 实现 schema + API + 挂载**

- [ ] **Step 4: 前端(Vibe)**
  `addFeedbackBar` 的 give 里按 chosen 发 `/api/feedback`,带上向上最近一条 user 气泡文本与页面会话 id;失败静默。浏览器验收后按用户描述改。

- [ ] **Step 5: 测试 + Commit**
  ```bash
  git commit -m "feat(ch09): wire thumbs-down feedback to low-confidence pool"
  ```

---

### Task 8: 飞轮流水线(标准化+查重 + 批处理 + 样例验证)

**Files:**
- Create: `app/core/flywheel.py`、`scripts/flywheel_pipeline.py`、`scripts/validate_flywheel_samples.py`、`tests/data/flywheel_samples.json`
- Modify: `app/core/prompts.py`(`FLYWHEEL_NORMALIZE_*`)
- Modify: `Makefile`(`flywheel`、`flywheel-samples`)
- Test: `tests/core/test_flywheel.py`

**Interfaces:**
- Produces:
  - `NormalizeResult(normalized_question: str, matched_question_id: int|None, ai_suggested_answer: str)`
  - `normalize_and_match(raw_question, candidates) -> NormalizeResult`
  - `process_pending(limit=50) -> {"processed","merged","created","skipped"}`

- [ ] **Step 1: 写 prompt**
  一次输出三键,`matched_question_id` 只能取候选内 id、宁愿 null;示例答案不臆造数字。用 `structured_model(NormalizeResult)`(仓库既有 structured 入口)或 `get_chat_model().with_structured_output(...)`,执行时按上游稳定性选一种并记 dev-notes。

- [ ] **Step 2: 失败测试** `tests/core/test_flywheel.py`
  新建缺口;命中已驳回只累加不复活;幻觉 id 跳过且游标不动;LLM 失败跳过该条;同批同义串行归并(第二条候选含第一条新建行)。

- [ ] **Step 3: 实现 `process_pending`**
  逐条现拉候选(200),串行;命中先校验 id 存在;失败/幻觉 → skipped,不 `set_matched_review`;成功回写归并落点;候选截断时日志说明。

- [ ] **Step 4: 测试 + 脚本 + make**

- [ ] **Step 5: 标注样例验证**
  `tests/data/flywheel_samples.json` 12 条(口语去噪/情绪剥离/同义命中/近似不同义/空候选/多诉求);判分 `matched_question_id` 相等 + `expect_keywords` 全含;≥ 80% 视为可用。不到线改 prompt 重跑,连续两轮不到线停下来问用户。

- [ ] **Step 6: Commit**
  ```bash
  git commit -m "feat(ch09): flywheel pipeline - normalize, dedup, review queue"
  ```

---

### Task 9: 审核 API + 通过写回知识库

**Files:**
- Create: `app/api/review.py`、`app/schemas/review.py`
- Modify: `app/main.py`(router + `/review` 占位路由)
- Test: `tests/test_review_api.py`

**Interfaces:**
- Produces:
  - `GET /api/review/queue?status=待审` → `{"items":[{id,normalized_question,ai_suggested_answer,occurrence_count,review_status,created_at}]}`
  - `GET /api/review/{id}` → 行 + `raws:[{raw_question,source,reason,created_at,retrieved_chunks}]`;404
  - `POST /api/review/{id}/approve {"approved_answer": str}` → `{"ok":true,"chunk_ids":[...]}`;非待审 409;写库/向量化失败 502 且状态仍待审
  - `POST /api/review/{id}/reject` → 200;非待审 409

- [ ] **Step 1: 失败测试**(monkeypatch `_write_chunks/_vectorize`;断言 chunk 形状 `category="飞轮沉淀"`、`content_type="faq"`、questions/answer;失败回滚状态)

- [ ] **Step 2: 实现**
  approve:取行 → 校验待审 → 构造 `Chunk` → `dualwrite.write_pending` → `dualwrite.vectorize_pending(milvus_client.get_client())`(仓库同步直调风格)→ 成功后 `update_review_status(通过)`;任一失败 502。
  reject:直接 `update_review_status`,False 时区分 404/409。

- [ ] **Step 3: 测试 + Commit**

**Async 注意:** `Chunk` 是 dataclass,`write_pending` 返回 ids;`vectorize_pending` 需显式 client 参数。

---

### Task 10: 自动化评估流水线

**Files:**
- Create: `scripts/eval_flywheel.py`
- Modify: `Makefile`(`eval-flywheel`)
- Output: `data/ch09/reports/eval_trend.txt`

**Interfaces:**
- Consumes: `scripts/eval_ch04.py` 的 `_load/_mean/_try/_evidence`(复用,不复制);`repository.insert_eval_run/list_eval_runs`。
- Produces: `make eval-flywheel [TRIGGER=手动|定时]`;metrics = `{"recall_at_10","mrr","faithfulness","refusal_rate"}`;趋势表(↑/↓/→,下滑 ⚠)。

- [ ] **Step 1: 写脚本**
  可答桶 = A/B/C/E,D 桶拒答;**检索指标**:沿用 eval_ch04 的 `_retrieval_scores` 口径(Recall@10/MRR,expect_section 命中);**生成段**:RAG 答案 + Faithfulness 判官(复用 `FAITHFULNESS_PROMPT`、`RAG_ANSWER_PROMPT`);拒答率复用 `_refusal` 或按 D 桶逐条判定。跑完 `insert_eval_run` 并打印趋势。若 eval_ch04 的 helper 与所需签名不符,抽公共函数而非复制。

- [ ] **Step 2: 实跑两轮**
  `make eval-flywheel` → `make eval-flywheel TRIGGER=定时`;Expected:两行趋势、`eval_runs` 两行。

- [ ] **Step 3: Commit**

---

### Task 11: Cost Control(按意图 token 账)

**Files:**
- Create: `scripts/cost_by_intent.py`
- Modify: `Makefile`(`cost-report`)
- Output: `data/ch09/reports/cost_by_intent.{txt,json}`

**Interfaces:**
- Consumes: `observability.get_langfuse()`;Langfuse v3 SDK Metrics/Legacy API。
- Produces: `make cost-report [DAYS=7]`;json 含 rows(意图/请求数/总tokens/平均/占比)、窗口、最烧钱意图。

- [ ] **Step 1: 反射核对已装 SDK**
  `python -c` 看 `client.api` 有哪些 metrics/legacy 资源;以本地 v3.15 实装为准。spec 已记:metadata 不可分组,v1 Metrics 仅 observations/scores 视图;route = `api.legacy.metrics_v1` observations 视图,按 tags+traceId 聚合,客户端求和。若实际资源名不同,按实装调整并记 dev-notes。

- [ ] **Step 2: 写脚本**
  按 `intent:xxx` tag 过滤/拉取 observations,汇总 totalTokens/count;排序、算平均与占比;txt 表格 + json;无 trace 时 `present=false` 提示。

- [ ] **Step 3: 实跑 + Commit**
  先各意图聊几句,再 `make cost-report`;与 Langfuse 界面数量级对得上。

---

### Task 12: 审核后台页(Vibe Coding)

**Files:**
- Create: `app/static/review.html`
- Modify: `app/static/admin.js`(导航加「飞轮待审」)

**页面清单:**顶栏标题 + 状态筛选(待审/通过/驳回/全部)+ 刷新;三道闸提示条(垃圾/时效/频次);表格(标准化问题/出现次数徽标/示例答案摘要/状态/操作);行详情 toggle(归并原话 + source 三色徽标 + 召回快照卡 question/answer/rerank_score 条/section_path,null 显示"未走检索");通过 modal(预填 ai_suggested_answer,成功 toast,502 红 toast 可重试);驳回直接调;引 `admin.js` 的 `mountAdminNav("/review")`;样式复用 index.html 的猫咪 CSS 变量。

- [ ] **Step 1: 实现页面**
- [ ] **Step 2: 浏览器验收 + 按用户描述改到满意**
- [ ] **Step 3: Commit**

---

### Task 13: 观测与成本页 + 只读 API + 作业注册

**Files:**
- Create: `app/api/observability.py`、`app/static/observability.html`、`tests/test_observability_api.py`
- Modify: `app/core/jobs.py`(注册 `cost-report`/`eval-flywheel`/`calibrate-confidence`,后两个 heavy)
- Modify: `app/api/admin.py`(多一张卡)、`app/static/admin.js`(导航多一格)、`app/main.py`(路由 + include)、`app/static/acceptance.js`(如需)
- Modify: `tests/test_admin_api.py`、`tests/test_static.py`

**Interfaces:**
- Produces: `GET /api/observability/overview` 三块:
  - `cost`:读 `data/ch09/reports/cost_by_intent.json`,`present`/`rows`/`missing_hint`
  - `eval`:读 `repository.list_eval_runs(10)`,失败只这块 `status="error"`
  - `calibration`:读 `confidence_calibration.json` + 在用阈值,`in_sync` 标记
- 三块互不连坐;产物缺失 200 + `present=false`;坏 json 按没跑过处理。

- [ ] **Step 1: 失败测试** 七种情形(全缺/成本透出/窗口无 trace/趋势倒序/阈值不同步/坏 json/趋势读库抛错)
- [ ] **Step 2: 实现 API**
- [ ] **Step 3: 作业注册 + 页面**(三 panel;成本占比条+明细;趋势最近十轮+涨跌箭头;校准分布+扫描曲线;重跑按钮走作业运行器)
- [ ] **Step 4: 挂后台外壳 + 测试 + 浏览器验收**
- [ ] **Step 5: Commit**

---

### Task 14: 端到端验收 + review + finish

- [ ] **Step 1: 全量测试** `uv run pytest -q` 全绿;`make flywheel-samples` ≥ 80%。
- [ ] **Step 2: 六条验收逐条实跑**(前置:langfuse-up、mysql、milvus、MCP、dev 全起)
  1. trace 树完整 (localhost:3000)
  2. 答不上 → 兜底 + `make flywheel` → /review 详情见原话与快照
  3. /review 通过填核准答案 → 再问同一问题答对
  4. 👎 → 落池 → flywheel → 待审队列
  5. 各意图聊几句 → `make cost-report` → /observability 同源
  6. `make eval-flywheel` 两轮 → 趋势对比,页面同源
  命令与关键输出记 `dev-notes/ch09.md`;1/2/3/7 截图。
- [ ] **Step 3: code review**(两轴)修复必要与建议项,结论记 dev-notes。
- [ ] **Step 4: finish**(留分支,等用户明说再合并 main)。

## Self-Review

- **Spec 覆盖**:§3 → Task 2;§4 → Task 11/13;§5 → Task 4/5/6;§6 → Task 7;§7 → Task 1/8;§8 → Task 9/12;§9 → Task 10/13;§10 → Task 7/12/13。无缺口。
- **占位符**:Task 6 的 `<实测值>` 是校准产出,是需求本身(不许拍脑袋);Task 11 Step 1 的 API 形状核对是工作要求 3 的显式步骤。
- **类型一致**:`retrieved_snapshot`(state)与 `retrieved_chunks`(DB 列)分层命名,各任务一致;`NormalizeResult` 三键与 prompt 输出一致;`get_review_detail` 返回 tuple,Task 9 解包一致。
- **Review Focus** 五条分别落在 Task 2/9/8/7/6。
