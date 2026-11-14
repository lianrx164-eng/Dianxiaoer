# Ch05 Workflow + Agent 混合架构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 LangGraph 把客服后端重构成「确定性 Workflow 骨架 + 主力 Agent 核心节点」的混合架构,两个 API 入口都走图,前端加转人工/建工单两个按钮。

**Architecture:** `StateGraph` 编排六段骨架(指代消解→意图识别→分流→主力 Agent→置信度兜底→日志);知识类强制先检索、证据弱生成前短路兜底;主力 Agent 是 `agent_llm ↔ agent_tools` 的 ReAct 环,复用 ch02 五工具;会话状态用 State 贯穿、AsyncSqliteSaver 持久化;转人工/建工单由后端产出可选项、前端按钮触发。

**Tech Stack:** LangGraph + langgraph-checkpoint-sqlite + aiosqlite;FastAPI(既有);LangChain ChatOpenAI 直连上游(既有);MySQL/SQLAlchemy(既有,存工单/低置信池/消息审计);原生 HTML/CSS/JS 前端(既有,加按钮)。

**Spec:** `docs/superpowers/specs/2026-11-14-ch05-workflow-agent-design.md`

## Global Constraints

- 上游选型定死:LangGraph 图编排 + State + checkpointer;checkpointer = **AsyncSqliteSaver**;前端 = 原生 HTML/CSS/JS,仅加按钮。发现矛盾停下问用户,不自行换方案。
- 本仓库无 Docker:Milvus 用 Lite,所有命令走 `uv run`(Windows 无 `.venv/bin/python`)。
- 复用 ch02 五工具,本章不新写业务工具;D7 只**修改** `query_logistics` 入参与 `query_order` 返回字段。
- 意图识别单标签七类(物流/订单/商品咨询/退款退货/售后/投诉/闲聊),指代消解原样透传。
- 转人工、建工单是两件事,分开;后端不自动执行,只产出可选项;建工单写库只在按钮端点、点了才写;转人工纯前端模拟。
- 涉及 LangGraph/FastAPI/SQLAlchemy 具体 API,动手前用 Context7 核实(已核实两处,见 spec §8.1)。
- 不可单测的产出(纯 Prompt/数据)用标注样例/评估集验证代替 TDD;其余走 TDD。
- 端到端真实验收必须跑通(非纸面);模型抖动如实记录。
- 每个任务末尾 commit;`dev-notes/ch05.md` 每完成一阶段追记。
- 评审分级按用户 memory:必要 / 建议 / 可选。

## Review Focus

- **同一 conversation 跨轮**:第二轮进来时上一轮的 `trace`、`evidence`、`suggested_actions` 必须已重置,否则日志会假报 `forced_rag=True`、按钮会重复渲染。
- **步数封顶时最后一条 AIMessage 带未执行 tool_calls**:下一轮上游会 400;必须补合成 ToolMessage。
- **确定性出口(chitchat/complaint/fallback/cap)之后继续对话**:checkpointer 历史里必须有那条 assistant 回合,否则模型不知道自己刚说过什么。
- **知识路 Agent 又调 query_faq**:会叠加分钟级延迟且可能得到与已注入证据冲突的结果;系统提示必须明确禁止,评估集覆盖。
- **SSE 首次帧前的错误**:图在 `stream_turn` 第一次 yield 前抛错时,前端只能收到 error 帧;`ConversationNotFound` 不能被吞成通用上游错误。

---

## 文件结构

**新建:**
- `app/graph/__init__.py` — 包
- `app/graph/state.py` — `ConversationState`
- `app/graph/routing.py` — `route_by_intent` / `confidence_gate` / `should_continue`
- `app/graph/nodes.py` — 各节点函数
- `app/graph/build.py` — `build_graph()` 组装 StateGraph
- `app/graph/runtime.py` — checkpointer 生命周期 + 图单例 + `run_turn`/`stream_turn`
- `app/core/intent.py` — 意图分类器(prompt + 结构化输出)
- `app/api/actions.py` / `app/schemas/actions.py` — 建工单按钮端点
- `scripts/bare_agent_loop.py` — 祛魅裸循环
- `scripts/smoke_langgraph.py` — 红线冒烟
- `scripts/eval_intent.py` — 意图分类标注评估
- `scripts/eval_ch05.py` — 五验收端到端评估
- `tests/graph/` — 图相关单测

**修改:**
- `pyproject.toml` / `.gitignore` — 依赖与 sqlite 忽略
- `app/config.py` — ch05 配置
- `app/core/prompts.py` — 意图/兜底/安抚/闲聊话术
- `app/tools/business.py` — D7:query_order 增 tracking_no、query_logistics 改吃 tracking_no
- `app/schemas/agent.py` — `AgentResponse.suggested_actions`
- `app/api/chat.py` / `app/api/agent.py` / `app/api/errors.py` — 改走图
- `app/main.py` — lifespan 起图 + 挂 actions 路由
- `app/static/index.html` — actions 帧 + 两按钮 + 工单表单弹窗
- `Makefile` — `smoke-graph` / `eval-intent` / `eval-ch05`
- `tests/test_tools_mock.py`、`tests/test_agent_api.py`、`tests/test_chat_api.py`、`tests/test_agent_citations.py`、`tests/test_agent_orchestration.py`、`tests/test_agent_stream.py` — 跟 D7 与图改造

**退役:** `app/core/agent.py` 的 `run_agent_turn`/`stream_agent_turn` 两次调用编排;`ConversationNotFound` 迁到 `app/graph/runtime.py`。

---

### Task 1: 依赖 + 配置 + LangGraph/checkpointer/流式 fail-fast 冒烟(红线)

**Files:**
- Modify: `pyproject.toml`、`.gitignore`、`app/config.py`
- Create: `scripts/smoke_langgraph.py`

**Interfaces:**
- Produces: `settings.max_agent_steps:int`、`settings.checkpointer_db_path:str`

- [ ] **Step 1: 加依赖与忽略**

Run: `uv add langgraph langgraph-checkpoint-sqlite aiosqlite`
`.gitignore` 的 data 段追加 `data/*.sqlite*`。

- [ ] **Step 2: 加配置**

`app/config.py` 的 `Settings` 增加 `max_agent_steps: int = 6`、`checkpointer_db_path: str = "data/ch05_checkpoints.sqlite"`。

- [ ] **Step 3: 写冒烟脚本**

`scripts/smoke_langgraph.py`:建一个只有 `call_model` 节点的图,用 `AsyncSqliteSaver.from_conn_string(":memory:")` + `await cp.setup()`,以 `stream_mode=["messages","updates"]` 跑一次模型调用,打印每个 `(mode, chunk)` 的形状与 `meta["langgraph_node"]`。

- [ ] **Step 4: 跑冒烟**

Run: `uv run python -m scripts.smoke_langgraph`
Expected: 打印 `MSG node= call_model ...` 与 `UPD ['call_model']`,`OK modes={'messages','updates'}`。形状不符则先按真实形状修正本计划后续解析代码,再继续;图跑不通则停下问用户。

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .gitignore app/config.py scripts/smoke_langgraph.py
git commit -m "feat(ch05): add langgraph deps and smoke-test streaming shape"
```

---

### Task 2: 祛魅热身——手写裸 Agent 循环(不用框架)

**Files:**
- Create: `scripts/bare_agent_loop.py`
- Modify: `dev-notes/ch05.md`(新建,阶段 2 追记)

**Interfaces:**
- Produces: `async def run_agent(query: str, max_turns: int = 6) -> str`

- [ ] **Step 1: 写脚本**

复用 `get_chat_model().bind_tools(get_all_tools())`、`AGENT_SYSTEM`、`execute_tool_call`;`for step in range(...)` 调模型、有 tool_calls 就执行并回灌、无则收敛;每步打印工具名与参数;封顶打印未收敛。

- [ ] **Step 2: 跑简单问 + 复杂问验证**

Run:
```
uv run python -m scripts.bare_agent_loop "退货政策是什么"
uv run python -m scripts.bare_agent_loop "订单1001的物流到哪了"
```
Expected: 简单问 1 步收敛;复杂问看到先 query_order 再 query_logistics 后收敛(D7 完成后)。D7 在本任务前未完成时,复杂问可能一步并行调两工具——如实记录,Task 8 完成后再跑一次。

- [ ] **Step 3: 写 dev-notes 阶段 2 + Commit**

```bash
git add scripts/bare_agent_loop.py dev-notes/ch05.md
git commit -m "feat(ch05): add bare agent loop demystification script"
```

---

### Task 3: 会话 State 定义

**Files:**
- Create: `app/graph/__init__.py`(空)、`app/graph/state.py`
- Test: `tests/graph/__init__.py`(空)、`tests/graph/test_state.py`

**Interfaces:**
- Produces: `ConversationState`(TypedDict,`total=False`),字段见 spec §5;`def trace_of(state, **updates) -> dict`(合并本轮 trace)

- [ ] **Step 1: 写失败测试**

`tests/graph/test_state.py`:
```python
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from app.graph.state import ConversationState, trace_of


def test_add_messages_reducer_appends():
    merged = add_messages([HumanMessage("hi")], [AIMessage("yo")])
    assert [m.content for m in merged] == ["hi", "yo"]


def test_trace_of_merges_without_mutating():
    old = {"intent": "订单"}
    out = trace_of({"trace": old}, route="business")
    assert out == {"intent": "订单", "route": "business"}
    assert old == {"intent": "订单"}


def test_state_has_required_keys():
    for k in ["messages", "user_id", "conversation_id", "intent", "route",
              "evidence", "citations", "evidence_strong", "answer",
              "steps", "tokens_used", "suggested_actions", "trace"]:
        assert k in ConversationState.__annotations__
```

- [ ] **Step 2: 跑验证失败**

Run: `uv run pytest tests/graph/test_state.py -v`
Expected: FAIL(`app.graph.state` 不存在)

- [ ] **Step 3: 实现 State**

`messages: Annotated[list[AnyMessage], add_messages]`,`trace: dict`(普通 dict,不挂 reducer),其余按 spec §5 类型;`trace_of` 返回 `{**(state.get("trace") or {}), **updates}`。

- [ ] **Step 4: 跑验证通过**

Run: `uv run pytest tests/graph/test_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/__init__.py app/graph/state.py tests/graph/
git commit -m "feat(ch05): add conversation state for workflow graph"
```

---

### Task 4: 意图分类器(Prompt/数据 → 评估集验证)

**Files:**
- Modify: `app/core/prompts.py`
- Create: `app/core/intent.py`、`scripts/eval_intent.py`
- Test: `tests/test_intent.py`

**Interfaces:**
- Produces: `INTENTS: tuple[str, ...]`;`async def classify(query: str) -> str`(异常/越界回退 `"闲聊"`)

- [ ] **Step 1: 写分类 Prompt**

`INTENT_CLASSIFY_SYSTEM`(七类定义 + 判不准选最接近 + 不许造类)与 `INTENT_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([...])`。

- [ ] **Step 2: 写失败测试(只测确定性部分)**

`tests/test_intent.py`:用假模型断言 `classify` 返回 schema 值;让调用抛异常时返回 `"闲聊"`。真实判对率由 Step 4 的标注评估覆盖。

- [ ] **Step 3: 实现分类器并跑测试**

`_Intent(BaseModel)` 单字段 `intent: Literal[...]`;用 `structured_model(_Intent)`(不是裸 `with_structured_output`,DeepSeek 思考模式需要 `function_calling` + disabled)。Run: `uv run pytest tests/test_intent.py -v` → PASS。

- [ ] **Step 4: 标注评估**

`scripts/eval_intent.py`:七类各 2 条标注样例。Run: `uv run python -m scripts.eval_intent`
Expected: 判对率 ≥ 12/14;明显偏差调 Prompt 重跑,如实记 dev-notes。

- [ ] **Step 5: Commit**

```bash
git add app/core/prompts.py app/core/intent.py scripts/eval_intent.py tests/test_intent.py
git commit -m "feat(ch05): add seven-way intent classifier with labelled eval"
```

---

### Task 5: 路由与停止条件(确定性 → TDD)

**Files:**
- Create: `app/graph/routing.py`
- Test: `tests/graph/test_routing.py`

**Interfaces:**
- Produces:
  - `INTENT_TO_ROUTE: dict[str, str]`
  - `route_by_intent(state) -> str`(`knowledge|business|complaint|chitchat`,未知→`business`)
  - `confidence_gate(state) -> str`(`strong|weak`,缺省 weak)
  - `should_continue(state) -> str`(`continue|stop|cap`;无 tool_calls→stop;步数≥max→cap)

- [ ] **Step 1: 写失败测试**

覆盖:七类→四出口映射、未知意图→business、证据闸三态、无工具→stop、有工具未封顶→continue、到 `settings.max_agent_steps` 且有工具→cap、token 不参与停止判断。

- [ ] **Step 2: 跑验证失败**

Run: `uv run pytest tests/graph/test_routing.py -v` → FAIL

- [ ] **Step 3: 实现路由**

按 spec §3.1 写死映射;`should_continue` 用 `isinstance(last, AIMessage) and bool(last.tool_calls)` 判断。

- [ ] **Step 4: 跑验证通过**

Run: `uv run pytest tests/graph/test_routing.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/routing.py tests/graph/test_routing.py
git commit -m "feat(ch05): add intent routing and react stop conditions"
```

---

### Task 6: 确定性出口节点(chitchat / complaint / fallback / cap + 话术)

**Files:**
- Modify: `app/core/prompts.py`、`app/graph/nodes.py`(新建)
- Test: `tests/graph/test_reply_nodes.py`

**Interfaces:**
- Produces: `CHITCHAT_REPLY` / `COMPLAINT_REPLY` / `FALLBACK_REPLY` 常量;`chitchat_reply`、`complaint_reply`、`fallback_reply`、`cap_reply` 节点;`_user_text(state) -> str`。

- [ ] **Step 1: 写失败测试**

断言:chitchat 固定话术且无 actions;complaint 两个 action(`transfer_human` + `create_ticket`,draft 为原话/投诉);fallback 调 `repository.insert_low_confidence(..., source="retrieval_low_conf")` 且 answer 为兜底话术;cap 为最后一条带 tool_calls 的 AIMessage 补 ToolMessage、answer 为兜底话术、不写低置信池。

- [ ] **Step 2: 跑验证失败**

Run: `uv run pytest tests/graph/test_reply_nodes.py -v` → FAIL

- [ ] **Step 3: 加话术 + 实现节点**

`fallback_reply` 从 `state["trace"].get("gate")` 取 `retrieval_low_conf` / `self_check`;`cap_reply` 遍历最后一条 AIMessage 的 `tool_calls` 生成 `ToolMessage(content="已中止:达到最大步数", tool_call_id=..., name=...)` 并返回 `{"answer": FALLBACK_REPLY, "messages": [...], "trace": {"cap": True}}`。

- [ ] **Step 4: 跑验证通过**

Run: `uv run pytest tests/graph/test_reply_nodes.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/prompts.py app/graph/nodes.py tests/graph/test_reply_nodes.py
git commit -m "feat(ch05): add deterministic reply nodes"
```

---

### Task 7: 强制检索 + 意图/指代节点

**Files:**
- Modify: `app/graph/nodes.py`
- Test: `tests/graph/test_forced_rag.py`

**Interfaces:**
- Consumes: `query_understanding.understand`、`retrieval.search_knowledge(query, strategy="hybrid_rerank", bm25_query=...)`、`retrieval.arrange_head_tail`、`selfcheck.check_sufficient`、`settings.rerank_min_score`
- Produces: `coref`、`classify_intent`、`forced_rag`、`confidence_check` 节点;`forced_rag` 输出 `evidence_strong/evidence/citations/trace`。

- [ ] **Step 1: 写失败测试**

强证据:mock 返回 1 条 0.9 分命中,断言 `evidence_strong=True`、`[1]` 在 evidence、citations[0]["n"]==1、`trace["forced_rag"] is True`。
弱证据:空命中→False 且 trace 带 `gate="retrieval_low_conf"`。
自评失败:命中但 `check_sufficient` 返回 False→False 且 `gate="self_check"`。
`confidence_check` 在 True/False 下分别留痕 strong/weak。

- [ ] **Step 2: 跑验证失败** → FAIL

- [ ] **Step 3: 实现节点**

`forced_rag` 复用 `query_faq` 的两道闸;BM25 查询词 = standard + expanded。`confidence_check` 只留痕(条件边另判)。

- [ ] **Step 4: 跑验证通过** → PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/nodes.py tests/graph/test_forced_rag.py
git commit -m "feat(ch05): add forced rag and intent nodes"
```

---

### Task 8: 主力 Agent 节点 + D7 工具依赖改造

**Files:**
- Modify: `app/tools/business.py`、`app/graph/nodes.py`、`tests/test_tools_mock.py`
- Test: `tests/graph/test_agent_nodes.py`

**Interfaces:**
- Consumes: `get_chat_model`、`get_all_tools`、`execute_tool_call`、`AGENT_SYSTEM`
- Produces: `_agent_messages(state)`、`agent_llm(state, config=None)`、`agent_tools(state)`;`query_order` 返回 `tracking_no`,`query_logistics(tracking_no)`。

- [ ] **Step 1: 先改工具的失败测试**

`tests/test_tools_mock.py`:断言 `query_order("1001")` 返回 `tracking_no` 且以 `SF` 开头;`query_logistics` 入参改为 `tracking_no`,返回里带 `tracking_no` 与 `status`。Run: 失败。

- [ ] **Step 2: 改工具实现**

`LogisticsInput.tracking_no`(描述:由 query_order 返回的物流单号);`query_order` 增加 `tracking_no = f"SF{rng.randint(10**9, 10**10-1)}"`(与其他随机字段同一 seed 实例)。

- [ ] **Step 3: Agent 节点失败测试**

断言:`_agent_messages` 知识路把 evidence 拼进 SystemMessage 且提及「不要再调用 query_faq」,business 路不拼;`agent_llm` 步数+1、token 按 usage_metadata 累加;`agent_tools` 普通工具经 `execute_tool_call`;create_ticket 被拦截(执行函数抛 AssertionError)、产出 suggested_actions 且回合成 ToolMessage。

- [ ] **Step 4: 实现 Agent 节点并跑测试**

`agent_llm` 用 `get_chat_model(streaming=True).bind_tools(get_all_tools())`;`agent_tools` 拦截 `create_ticket`,其余顺序执行(模型给的顺序有意义)。

Run: `uv run pytest tests/test_tools_mock.py tests/graph/test_agent_nodes.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add app/tools/business.py app/graph/nodes.py tests/test_tools_mock.py tests/graph/test_agent_nodes.py
git commit -m "feat(ch05): add react agent nodes and tracking_no dependency"
```

---

### Task 9: 日志节点 log

**Files:**
- Modify: `app/graph/nodes.py`
- Test: `tests/graph/test_log_node.py`

**Interfaces:**
- Produces: `resolve_answer(state) -> str`;`log_node(state) -> dict`

- [ ] **Step 1: 写失败测试**

`resolve_answer` 优先 `state["answer"]`,否则最后一条 AIMessage 文本;`log_node` 调 `repository.append_message(cid,"assistant",content=...)`;确定性出口(`answer` 非空)返回 `{"messages": [AIMessage(answer)]}` 以回填 checkpointer 历史,Agent 出口不重复回填。

- [ ] **Step 2: 跑失败** → FAIL

- [ ] **Step 3: 实现**

日志打印 conv/intent/route/trace;仅当 `conversation_id` 存在才写库。

- [ ] **Step 4: 跑通过** → PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/nodes.py tests/graph/test_log_node.py
git commit -m "feat(ch05): add log node with history backfill"
```

---

### Task 10: 图组装 build_graph

**Files:**
- Create: `app/graph/build.py`
- Test: `tests/graph/test_build.py`

**Interfaces:**
- Produces: `_builder()` → 未编译 StateGraph;`build_graph(checkpointer=None)`。

- [ ] **Step 1: 写失败测试**

断言全部节点名存在;`compile(checkpointer=InMemorySaver())` 成功;条件边映射包含 `cap → cap_reply`。

- [ ] **Step 2: 跑失败** → FAIL

- [ ] **Step 3: 实现**

边:`START→coref→classify_intent`;`route_by_intent` 四出口;`forced_rag→confidence_check`,强→agent_llm,弱→fallback_reply;`agent_llm` 的 `should_continue` continue→agent_tools、stop→log、cap→cap_reply;`agent_tools→agent_llm`;四个确定性出口→log;log→END。

- [ ] **Step 4: 跑通过** → PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/build.py tests/graph/test_build.py
git commit -m "feat(ch05): assemble workflow graph"
```

---

### Task 11: 运行时——checkpointer 生命周期 + run_turn/stream_turn

**Files:**
- Create: `app/graph/runtime.py`
- Test: `tests/graph/test_runtime.py`

**Interfaces:**
- Produces: `init_graph()`、`close_graph()`、`get_graph()`、`ConversationNotFound`、`run_turn(user_id,message,conversation_id)->{"conversation_id","state"}`、`stream_turn(...)` 产出 `tool|citations|delta|actions|done` 事件。

- [ ] **Step 1: 写失败测试**

新建会话/会话不存在;事件映射:回答节点的 token 成 delta、classify_intent 的 token 被过滤、确定性出口 answer 成 delta、agent_tools 出 tool、forced_rag 出 citations、actions 在 done 前、done 带 cid;每轮输入重置 trace/evidence/answer/suggested_actions。

- [ ] **Step 2: 跑失败** → FAIL

- [ ] **Step 3: 实现**

`AsyncSqliteSaver.from_conn_string(...)` 存模块级 `_cm`,lifespan 里 `__aenter__` + `setup()`;`ANSWER_NODES={"agent_llm"}`,`DETERMINISTIC_ANSWER_NODES={"chitchat_reply","complaint_reply","fallback_reply","cap_reply"}`;输入用 `_graph_input` 显式重置本轮字段、`steps=0`、`tokens_used=0`。

- [ ] **Step 4: 跑通过** → PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/runtime.py tests/graph/test_runtime.py
git commit -m "feat(ch05): add graph runtime and event mapping"
```

---

### Task 12: /api/agent 接图 + suggested_actions

**Files:**
- Modify: `app/schemas/agent.py`、`app/api/agent.py`
- Test: `tests/graph/test_agent_view.py`、`tests/test_agent_api.py`

**Interfaces:**
- Consumes: `runtime.run_turn`
- Produces: `AgentResponse.suggested_actions: list = []`;`_views_from_state(state) -> (calls, results)`

- [ ] **Step 1: 写失败测试**

从 messages 重建 tool_calls/tool_results;API 层 monkeypatch `runtime.run_turn` 返回带 suggested_actions 的终态,断言 200 与字段;保留 404/DB 错/上游错错误路径(monkeypatch 抛 `ConversationNotFound`/`SQLAlchemyError`/`RuntimeError`)。

- [ ] **Step 2: 跑失败** → FAIL

- [ ] **Step 3: 实现**

`app/api/agent.py` 改调 `runtime.run_turn`;`resolve_answer` 从 `app.graph.nodes` 导入;错误映射保留 `errors.classify`(其 import 在 Task 15 一并改)。

- [ ] **Step 4: 跑通过** → PASS

- [ ] **Step 5: Commit**

```bash
git add app/schemas/agent.py app/api/agent.py tests/graph/test_agent_view.py tests/test_agent_api.py
git commit -m "feat(ch05): route agent api through graph"
```

---

### Task 13: /api/chat 接图 + actions SSE 帧

**Files:**
- Modify: `app/api/chat.py`
- Test: `tests/test_chat_api.py`

**Interfaces:**
- Consumes: `runtime.stream_turn`
- Produces: SSE 帧多一种 `{"event":"actions","items":[...]}`

- [ ] **Step 1: 改造失败测试**

monkeypatch `runtime.stream_turn` 产出 delta/actions/done,断言响应含 `"event": "actions"`、delta、`[DONE]`;error 帧路径保留。

- [ ] **Step 2: 跑失败** → FAIL

- [ ] **Step 3: 实现**

按事件类型映射;错误继续走 `errors.classify`。

- [ ] **Step 4: 跑通过** → PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/chat.py tests/test_chat_api.py
git commit -m "feat(ch05): route chat sse through graph with actions frame"
```

---

### Task 14: 建工单按钮端点

**Files:**
- Create: `app/api/actions.py`、`app/schemas/actions.py`
- Test: `tests/test_actions_api.py`

**Interfaces:**
- Produces: `POST /api/actions/create-ticket`,`{conversation_id:int, description:str, ticket_type:Literal["售后","投诉","咨询"]}` → `{ticket_no, status}`

- [ ] **Step 1: 写失败测试**

正常写入(monkeypatch `repository.create_ticket`);非法 ticket_type → 422;DB 错 → 503。

- [ ] **Step 2: 跑失败** → FAIL

- [ ] **Step 3: 实现**

按 schema 校验 + repository 调用。

- [ ] **Step 4: 跑通过** → PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/actions.py app/schemas/actions.py tests/test_actions_api.py
git commit -m "feat(ch05): add create-ticket action endpoint"
```

---

### Task 15: main.py 接线 + errors 迁移

**Files:**
- Modify: `app/main.py`、`app/api/errors.py`
- Test: 既有 `tests/test_static.py`(回归)

- [ ] **Step 1: 改接线**

`lifespan`(asynccontextmanager)里 `await runtime.init_graph()`,`yield` 后 `await runtime.close_graph()`;挂 `actions_router`;`errors.py` 改 `from app.graph.runtime import ConversationNotFound`。

- [ ] **Step 2: 验证**

Run: `uv run pytest tests/test_static.py -v`;`PORT=8002 ./scripts/dev.sh` 起服务,`curl -s localhost:8002/openapi.json | grep create-ticket`。
Expected: PASS + 路由存在、应用能起。

- [ ] **Step 3: Commit**

```bash
git add app/main.py app/api/errors.py
git commit -m "feat(ch05): start graph in lifespan and mount actions route"
```

---

### Task 16: 前端——actions 帧 + 两按钮 + 工单表单弹窗

**Files:**
- Modify: `app/static/index.html`

- [ ] **Step 1: 解析 actions 帧**

`sendMessage` 的 SSE 分发里收集 `payload.event === "actions"` 到 `actions` 数组;流结束后 `renderActions(bubble, actions)`。

- [ ] **Step 2: 渲染与交互**

`renderActions`:每个 action 一个独立按钮。转人工:置灰自身,追加两条 bot 气泡「已转接人工客服」「您好,我是客服小猫,请问有什么可以帮您的」。建工单:打开表单弹窗(类别下拉 售后/投诉/咨询、不预选;描述文本域、必填),校验后 POST `/api/actions/create-ticket`,成功显示「工单已创建 T…」并置灰该按钮,失败提示且按钮可重开;取消/遮罩/Esc 关闭不建单。

- [ ] **Step 3: 加样式**

`.action-bar` / `.action-btn` / 弹窗遮罩与表单样式,沿用现有配色。

- [ ] **Step 4: 浏览器验收**

起服务后:投诉问出两按钮;点转人工见两条气泡;重新投诉点建工单,空提交被拦、填写后成功;查 `SELECT * FROM tickets ORDER BY created_at DESC LIMIT 1`;不点按钮继续发消息正常。截图存 `dev-notes/ch05-acceptance3-*.png`。

- [ ] **Step 5: Commit**

```bash
git add app/static/index.html && git add -f dev-notes/ch05-acceptance3-*.png
git commit -m "feat(ch05): render action buttons and ticket form in chat"
```

---

### Task 17: 五验收端到端评估 + 真实验收

**Files:**
- Create: `scripts/eval_ch05.py`
- Modify: `Makefile`、`README.md`

- [ ] **Step 1: 写脚本**

走 `/api/agent`:验收 2(query_logistics 命中)、3(两 action)、4(闲聊不调工具)、5(tool_calls ≥2 且顺序为 query_order→query_logistics);验收 1 提示看日志。`BASE_URL` 可覆盖端口,`trust_env=False`。

- [ ] **Step 2: Makefile 目标**

`smoke-graph`、`eval-intent`、`eval-ch05`。

- [ ] **Step 3: 真实验收**

起服务跑 `make eval-ch05`;验收 1 用 `curl` 问政策类问题后看服务日志 `route=knowledge ... 'forced_rag': True`;浏览器完成验收 3 前端部分。抖动如实记录。

- [ ] **Step 4: 全量回归**

Run: `uv run pytest -q`
Expected: 全绿。

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_ch05.py Makefile README.md dev-notes/ch05.md
git commit -m "test(ch05): add end-to-end acceptance eval"
```

---

### Task 18: 收尾——退役旧编排 + README/dev-notes

**Files:**
- Modify: `app/core/agent.py`、旧测试、`README.md`、`dev-notes/ch05.md`

- [ ] **Step 1: 确认无引用**

Run: `grep -rn "run_agent_turn\|stream_agent_turn\|core.agent" app tests scripts --include=*.py`。仅旧测试引用则删除两个函数与不再复用的 helper;`scripts/eval_agent.py` 保持走 `/api/agent`。

- [ ] **Step 2: 改写旧测试**

`test_agent_orchestration.py` / `test_agent_stream.py` / `test_agent_citations.py` 改写为图节点/runtime 测试或删除与 `tests/graph` 重复的用例(文件保留,不删文件)。

- [ ] **Step 3: 全量回归**

Run: `uv run pytest -q` → 全绿。

- [ ] **Step 4: README + dev-notes finish 段**

补 ch05 演示命令、五验收结论、翻车汇总。

- [ ] **Step 5: Commit**

```bash
git add app/core/agent.py app/api tests README.md dev-notes/ch05.md
git commit -m "chore(ch05): retire legacy orchestration and document acceptance"
```

---

## Self-Review

**Spec 覆盖:** §2 数据流→Task 10;§3 节点→Task 6/7/8/9;§3.1 分流→Task 5;§4 ReAct 停止/token→Task 5/8;§5 State+checkpointer+重置→Task 3/11;§6 动作机制→Task 8/14/16;§7 两入口→Task 12/13;§8 依赖与验证→Task 1/4/17;§9 无新表→复用;§11 D1–D10→各任务;§12 R1–R4→Task 1/5/8/11/17;§13 五验收→Task 16/17。

**类型一致:** `route_by_intent` 返回四出口键与 build 映射一致;`should_continue` 三值 `continue/stop/cap` 与 build 一致;事件 dict 五型与 chat 映射一致;`suggested_actions` 结构 `{type,draft?}` 在 6/8/12/16 一致;`run_turn` 返回 `{conversation_id,state}` 与 agent.py 消费一致。

**Review Focus 对齐:** 跨轮重置→Task 11 测试;悬空 tool_calls→Task 6(cap)与 Task 5;确定性出口历史回填→Task 9;知识路禁再检索→Task 8 测试;首帧前错误→Task 11/13。
