# Ch06 意图识别与对话管理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch05 骨架里占位的「分流器前段」做成正式版:指代消解 + Query 改写、意图四件套(8 类含「其他」+ confidence)、五出口路由、退款退货/售后的确定性子流程(interrupt 弹订单选择器 → Query 扩写强制检索政策 → 主力 Agent 只判能不能退 → submit_refund 承载退款表单)。

**Architecture:** 沿用 ch05 `StateGraph`。前段两站做实:`coref`(LLM coref + 改写,写 `resolved_query`)→ `classify_intent`(四件套,写 `intent`/`intent_confidence`)。`route_by_intent` 由 4 出口扩成 5 出口(+`refund_flow`,+「其他」归 `fallback_script`)。新增确定性子流程 `fetch_order → retrieve_policy → agent_llm`:缺订单号用 LangGraph `interrupt` 暂停等前端点选、`Command(resume)` 续跑;`retrieve_policy` 做 Query 扩写多查去重合并强制检索政策证据;主力 Agent 判「能不能退」,能退调 `submit_refund` 被 `agent_tools` 拦成前端退款表单(仿 ch05 create_ticket)。置信度闸位置、主力 Agent ReAct 环、两入口走图,全部沿用 ch05 不动。

**Tech Stack:** LangGraph(interrupt/Command 是自带 HITL 能力,**无新增依赖**)+ AsyncSqliteSaver(既有);LangChain ChatOpenAI 直连上游(既有);ch04 混合检索 + 重排检索器(既有);MySQL/SQLAlchemy(既有,tickets 表加 `退款` 枚举值);原生 HTML/CSS/JS 前端(既有,加订单卡片 + 退款表单)。

**Spec:** `docs/superpowers/specs/2026-11-14-ch06-intent-recognition-design.md`

## Global Constraints

- 上游选型定死:LangGraph `interrupt`/`Command(resume)` 做订单选择器 HITL(**无新组件**);既有模型工厂;复用 ch04 检索器与 tickets 表。发现矛盾/死路停下问用户,不自行换方案。
- 意图识别默认走够格大模型(`settings.intent_model`,空则回落 `chat_model`)**先求准**;降级路(小模型判 + 置信度、低置信升级大模型)本章**只落配置种子,不实现运行时**,`intent_mode` 恒 `accuracy`。
- Query 扩写(严格 3 条 JSON `{"queries":[...]}`)**只在 refund_flow 的 `retrieve_policy`** 检索侧做;商品咨询 `knowledge` 路沿用 ch05 同义词轻扩;**库里知识只留一份**,不在入库侧拆存。
- 退款子流程只把「这一单能不能退」交主力 Agent;取单、检索政策是写死的确定性节点;**缺订单号不猜不编**,执行阶段 `interrupt` 弹订单选择器点选回填;退款原因提交时固定类目下拉。
- **前端例外走 Vibe Coding**(用户描述效果、我改):不套 brainstorm/TDD/code-review;功能必须落到前端在用入口,浏览器端到端验收。
- 不可单测的产出(纯 Prompt:coref/expand/意图四件套)用**标注评估集**代替 TDD;确定性代码走 **TDD**;`interrupt`/`resume`/`astream` 中断 surface 形状先用**红线冒烟**钉死(fail-fast 最前)。
- 涉及 LangGraph/FastAPI/SQLAlchemy 具体 API,动手前用 **Context7** 核实当前接口(已核实 interrupt/resume;其余沿用 ch05 已验证用法)。
- 沿用 ch05:置信度闸在 Agent 之前(生成前证据闸,ch05 D3);节点名 `coref`/`forced_rag`/`agent_llm` 不变;trace 观测键(`coref`/`forced_rag`)保留。
- 每个任务末尾 commit(分支 `ch06-intent-recognition`);`dev-notes/ch06.md` 每完成一阶段追记,**不许收尾时一次性补记**。
- 模型非确定性:eval 抖动如实重跑记录。
- 命令:测试 `uv run pytest -q`;脚本 `PYTHONUTF8=1 uv run python -m scripts.xxx`;服务 `make dev`(PORT 可覆盖)。

## 权威来源

- 本仓 spec:`docs/superpowers/specs/2026-11-14-ch06-intent-recognition-design.md`
- 桌面 Spec/Plan:`C:\Users\Jia Bao Yu\Desktop\智能客服\CH_06\Sepc.md`、`Plan.md`(基线差异见 spec §2)

---

## 文件结构

**新建:**
- `app/core/coref.py` — 指代消解 + 改写 `resolve(query, history) -> str`
- `scripts/smoke_interrupt.py` — interrupt/resume/astream 中断 surface 红线冒烟
- `scripts/eval_coref.py` — 指代消解标注评估(含已完整透传)
- `scripts/eval_expand.py` — Query 扩写标注评估(核心场景出 3 条 / JSON 稳定)
- `scripts/eval_ch06.py` — 四验收端到端评估
- `sql/ch06-ticket-type.sql` — tickets.ticket_type 枚举加 `退款`
- `tests/graph/test_ch06_nodes.py` — 新节点单测
- `tests/test_refund_api.py` — create-refund + resume 端点测试

**修改:**
- `app/graph/state.py` — 加 `resolved_query`/`intent_confidence`/`order_id`/`order_data`
- `app/graph/routing.py` — `INTENT_TO_ROUTE` 8→5 出口
- `app/graph/nodes.py` — `coref`/`classify_intent` 做实;新增 `fetch_order`/`retrieve_policy`/`script_reply`;`agent_tools` 加 submit_refund 拦截;`_agent_messages` refund 适配;退役 `chitchat_reply`
- `app/graph/build.py` — 5 出口条件边 + refund_flow 链 + `script_reply`
- `app/graph/runtime.py` — `resume_turn`/`stream_resume` + interrupt 事件 + 未决中断检测 + 节点名对齐
- `app/core/intent.py` — 七类单标签 → 八类 `{intent, confidence}` + 「其他」兜底
- `app/core/query_understanding.py` — 加 `expand_queries(query) -> list[str]`
- `app/core/prompts.py` — 加 `COREF_REWRITE_*`/`EXPAND_QUERIES_*`/重写 `INTENT_CLASSIFY_*`/`SCRIPT_REPLY_OTHER`/`REFUND_JUDGE_HINT`
- `app/core/llm.py` — `get_chat_model` 加 `model` 覆盖参
- `app/config.py` — 意图模型配置种子
- `app/tools/business.py` — 抽 `order_snapshot`;加 `list_user_orders`/`submit_refund`
- `app/tools/registry.py` — 注册 `submit_refund`
- `app/db/models.py` + `sql/ch02-ddl.sql` — `Ticket.ticket_type` 枚举加 `退款`
- `app/api/actions.py` — `POST /api/actions/create-refund` + `POST /api/actions/resume`(SSE)
- `app/api/chat.py` — `interrupt` 事件 → SSE `event: interrupt`
- `app/api/agent.py` + `app/schemas/agent.py` — `AgentResponse.interrupt`
- `app/schemas/actions.py` — `CreateRefundRequest/Response`、`ResumeRequest`
- `app/static/index.html` — 订单选择器卡片 + 退款表单(Vibe Coding)
- `Makefile` — `smoke-interrupt`/`eval-coref`/`eval-expand`/`eval-ch06`
- `tests/graph/test_routing.py` — 7→4 改 8→5
- `tests/graph/test_reply_nodes.py` — `chitchat_reply` 改 `script_reply`
- `tests/graph/test_forced_rag.py` — `classify` 返回 dict + refund_flow
- `tests/graph/test_build.py` — 节点清单
- `tests/graph/test_runtime.py` — interrupt/resume 用例
- `tests/test_chat_api.py` — interrupt 帧
- `tests/test_agent_api.py` — interrupt 字段
- `scripts/eval_intent.py` — 8 类 + 其他 + 历史
- `README.md` — ch06 验收命令

---

## Task 1: interrupt/resume/astream 中断 surface 红线冒烟(fail-fast)

**目的:** spec §8 / R1。把已实测的结论固化成可重跑脚本,后续 Task 7/13 直接引用。

**Files:**
- Create: `scripts/smoke_interrupt.py`
- Modify: `Makefile`

**Interfaces:**
- Produces:可重跑的冒烟脚本;A/B/C/C'/D 五段真实形状。

- [ ] **Step 1: 写冒烟脚本**

包含:A 非流式 `ainvoke` 的 `__interrupt__`;B `Command(resume)`;C `astream` 的中断 chunk;C' `aget_state` 的 `next/interrupts`;C'' 流式 resume;D 已暂停线程收到新输入的行为。

- [ ] **Step 2: 加 Makefile 目标并跑**

`smoke-interrupt:` → `PYTHONUTF8=1 uv run python -m scripts.smoke_interrupt`
Expected: 打印 `__interrupt__` 取法、updates chunk 键、`aget_state.interrupts`、resume 后节点重跑、D 段新输入再次触发中断。

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_interrupt.py Makefile
git commit -m "feat(ch06): interrupt/resume 中断 surface 冒烟脚本"
```

---

## Task 2: 会话 State 扩字段 + 入口清零

**Files:**
- Modify: `app/graph/state.py`
- Modify: `app/graph/runtime.py`(`_graph_input`)
- Test: `tests/graph/test_runtime.py`

**Interfaces:**
- Produces:`ConversationState` 加 `resolved_query:str`、`intent_confidence:float`、`order_id:str`、`order_data:dict`;`_graph_input` 清零。

- [ ] **Step 1: 扩测试断言**

在 `test_run_turn_creates_conversation_when_none` 的 fake ainvoke 里断言:
```python
assert inp["resolved_query"] == "" and inp["intent_confidence"] == 0.0
assert inp["order_id"] == "" and inp["order_data"] == {}
```
并在 `test_stream_turn_maps_events` 的 astream 断言里同样加。

- [ ] **Step 2: 跑失败**

Run: `uv run pytest tests/graph/test_runtime.py -q`
Expected: FAIL(KeyError)。

- [ ] **Step 3: 加字段 + 清零**

state.py 加四个字段;`_graph_input` 加:
```python
"resolved_query": "", "intent_confidence": 0.0,
"order_id": "", "order_data": {},
```

- [ ] **Step 4: 跑通过 + Commit**

Run: `uv run pytest tests/graph/test_runtime.py -q` → PASS
```bash
git add app/graph/state.py app/graph/runtime.py tests/graph/test_runtime.py
git commit -m "feat(ch06): State 加 resolved_query/intent_confidence/order_id/order_data"
```

---

## Task 3: 路由 5 出口

**Files:**
- Modify: `app/graph/routing.py`
- Test: `tests/graph/test_routing.py`

**Interfaces:**
- Produces:`INTENT_TO_ROUTE` 8 键 → 5 值(`escalate/fallback_script/knowledge/refund_flow/business`);未知 → business。

- [ ] **Step 1: 改测试**

替换旧 7→4 用例为参数化 8 类断言 + `INTENT_TO_ROUTE` 键集断言 + 未知归 business。

- [ ] **Step 2: 跑失败**

Run: `uv run pytest tests/graph/test_routing.py -q` → FAIL。

- [ ] **Step 3: 改路由表**

```python
INTENT_TO_ROUTE: dict[str, str] = {
    "投诉": "escalate", "闲聊": "fallback_script", "其他": "fallback_script",
    "商品咨询": "knowledge", "退款退货": "refund_flow", "售后": "refund_flow",
    "物流": "business", "订单": "business",
}
```

- [ ] **Step 4: 跑通过 + Commit**

Run: `uv run pytest tests/graph/test_routing.py -q` → PASS
```bash
git add app/graph/routing.py tests/graph/test_routing.py
git commit -m "feat(ch06): 路由扩成 5 出口"
```

---

## Task 4: 意图四件套 + 模型配置种子 + classify_intent 节点

**Files:**
- Modify: `app/core/prompts.py`(`INTENT_CLASSIFY_*` 重写)
- Modify: `app/core/intent.py`(全文)
- Modify: `app/core/llm.py`(`model` 覆盖)
- Modify: `app/config.py`(配置种子)
- Modify: `app/graph/nodes.py`(`classify_intent` + `_history_text`)
- Test: `tests/graph/test_ch06_nodes.py`
- Modify: `tests/graph/test_forced_rag.py`
- Modify: `scripts/eval_intent.py`

**Interfaces:**
- Produces:`intent.classify(query, history="") -> {"intent":str,"confidence":float}`;`get_chat_model(..., model:str|None=None)`;节点写 `intent/intent_confidence/route` + trace。

- [ ] **Step 1: 写失败测试**

`tests/graph/test_ch06_nodes.py`:
```python
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nodes


@pytest.mark.asyncio
async def test_classify_intent_writes_confidence_and_route(monkeypatch):
    async def fake_classify(query, history=""):
        return {"intent": "退款退货", "confidence": 0.83}
    monkeypatch.setattr(nodes.intent_mod, "classify", fake_classify)
    out = await nodes.classify_intent({
        "messages": [HumanMessage("这个能退吗")],
        "resolved_query": "蓝牙耳机还能申请退货吗"})
    assert out["intent"] == "退款退货"
    assert out["intent_confidence"] == 0.83
    assert out["route"] == "refund_flow"
    assert out["trace"]["route"] == "refund_flow"
    assert out["trace"]["intent_confidence"] == 0.83
    assert "confidence" not in out["trace"]      # 不占用证据闸的键


def test_get_chat_model_honors_model_override():
    from app.config import settings
    from app.core.llm import get_chat_model
    m = get_chat_model(model="glm-4-flash")
    assert m.model_name == "glm-4-flash"
    assert get_chat_model().model_name == settings.chat_model
```

- [ ] **Step 2: 跑失败**

Run: `uv run pytest tests/graph/test_ch06_nodes.py -q` → FAIL。

- [ ] **Step 3: llm.py 加 `model` 覆盖**

```python
def get_chat_model(streaming=False, extra_body=None, temperature=0.3, model=None):
    return ChatOpenAI(model=model or settings.chat_model, ...)
```

- [ ] **Step 4: config 加种子**

```python
intent_model: str = ""
intent_small_model: str = ""
intent_mode: str = "accuracy"
intent_conf_threshold: float = 0.6
```

- [ ] **Step 5: 重写意图 prompt(八类 + 边界 few-shot + JSON)**

system 列八类(含「其他」)、边界样例、输出 `{"intent":..., "confidence":...}`;human 带 `{history}` 与 `{query}`。

- [ ] **Step 6: 重写 intent.py**

```python
Intent = Literal["物流","订单","商品咨询","退款退货","售后","投诉","闲聊","其他"]
INTENTS = get_args(Intent)

class _Intent(BaseModel):
    intent: Intent
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

async def classify(query: str, history: str = "") -> dict:
    model = get_chat_model(model=settings.intent_model or None,
                           extra_body=thinking_body())
    try:
        r = await structured_model(_Intent, model=model).ainvoke(
            INTENT_CLASSIFY_PROMPT.format_messages(query=query, history=history or "(无)"))
    except Exception:
        logger.warning("意图分类失败,归其他", exc_info=True)
        return {"intent": "其他", "confidence": 0.0}
    intent = r.intent if r.intent in INTENTS else "其他"
    return {"intent": intent, "confidence": float(r.confidence)}
```

- [ ] **Step 7: `_history_text` + 改 `classify_intent`**

```python
def _history_text(state, max_messages: int = 6) -> str:
    msgs = state.get("messages", [])
    prior = msgs[:-1] if msgs else []
    lines = []
    for m in prior[-max_messages:]:
        role = "用户" if isinstance(m, HumanMessage) else "客服"
        text = m.content if isinstance(m.content, str) else ""
        if text:
            lines.append(f"{role}:{text}")
    return "\n".join(lines)

async def classify_intent(state) -> dict:
    query = state.get("resolved_query") or _user_text(state)
    r = await intent_mod.classify(query, _history_text(state))
    intent, conf = r["intent"], r["confidence"]
    route = INTENT_TO_ROUTE.get(intent, "business")
    return {"intent": intent, "intent_confidence": conf, "route": route,
            "trace": trace_of(state, intent=intent,
                              intent_confidence=conf, route=route)}
```

- [ ] **Step 8: 改 ch05 过时测试 + 跑通过**

`tests/graph/test_forced_rag.py::test_classify_intent_writes_state` 的 fake 改返回 dict;断言 `route == "business"`(物流仍 business)即可。跑:
`uv run pytest tests/graph/test_ch06_nodes.py tests/graph/test_forced_rag.py -q` → PASS。

- [ ] **Step 9: 扩 eval_intent.py**

16 条(8 类 + 其他 + 边界)+ 多轮历史用例;断言 intent 与 confidence 可解析。

- [ ] **Step 10: 跑 eval(需上游)并记录**

`PYTHONUTF8=1 uv run python -m scripts.eval_intent` → 记录判对率。

- [ ] **Step 11: Commit**

```bash
git add app/core/intent.py app/core/prompts.py app/core/llm.py app/config.py \
  app/graph/nodes.py tests/graph/test_ch06_nodes.py tests/graph/test_forced_rag.py \
  scripts/eval_intent.py
git commit -m "feat(ch06): 意图四件套(8类+confidence+其他兜底)+ 配置种子"
```

---

## Task 5: 指代消解 + Query 改写(`coref.resolve`)

**Files:**
- Create: `app/core/coref.py`
- Modify: `app/core/prompts.py`
- Modify: `app/graph/nodes.py`(`coref`)
- Create: `scripts/eval_coref.py`
- Test: `tests/graph/test_ch06_nodes.py`

**Interfaces:**
- Produces:`coref.resolve(query, history="") -> str`;节点写 `resolved_query`,trace `coref=rewrite|passthrough`。

- [ ] **Step 1: 失败测试**

```python
@pytest.mark.asyncio
async def test_resolve_reference_passthrough_when_complete(monkeypatch):
    async def fake_resolve(q, history=""):
        return q
    monkeypatch.setattr(nodes.coref_mod, "resolve", fake_resolve)
    out = await nodes.coref({"messages": [HumanMessage("蓝牙耳机的保修期多久")]})
    assert out["resolved_query"] == "蓝牙耳机的保修期多久"
    assert out["trace"]["coref"] == "passthrough"


@pytest.mark.asyncio
async def test_resolve_reference_rewrites_with_history(monkeypatch):
    async def fake_resolve(q, history=""):
        return "蓝牙耳机还能申请退货吗"
    monkeypatch.setattr(nodes.coref_mod, "resolve", fake_resolve)
    out = await nodes.coref({"messages": [
        HumanMessage("蓝牙耳机什么时候到"), AIMessage("预计明天"), HumanMessage("这个能退吗")]})
    assert out["resolved_query"] == "蓝牙耳机还能申请退货吗"
    assert out["trace"]["coref"] == "rewrite"
```

- [ ] **Step 2: 跑失败** → FAIL。

- [ ] **Step 3: 加 prompt + coref.py**

`COREF_REWRITE_SYSTEM` 含:指代补实体;指代某单时带订单号;已完整/通用规则问题原样透传;只输出一句。
`resolve` 失败返回原句。

- [ ] **Step 4: 改 coref 节点**

```python
from app.core import coref as coref_mod

async def coref(state) -> dict:
    query = _user_text(state)
    resolved = await coref_mod.resolve(query, _history_text(state))
    mode = "rewrite" if resolved != query else "passthrough"
    return {"resolved_query": resolved, "trace": trace_of(state, coref=mode)}
```

- [ ] **Step 5: 跑通过 + eval_coref**

写 `scripts/eval_coref.py`(2 补全 + 2 透传);跑并记录。

- [ ] **Step 6: Commit**

```bash
git add app/core/coref.py app/core/prompts.py app/graph/nodes.py \
  tests/graph/test_ch06_nodes.py scripts/eval_coref.py
git commit -m "feat(ch06): 指代消解+改写做实"
```

---

## Task 6: Query 扩写(`expand_queries`)

**Files:**
- Modify: `app/core/prompts.py`
- Modify: `app/core/query_understanding.py`
- Create: `scripts/eval_expand.py`

**Interfaces:**
- Produces:`expand_queries(query) -> list[str]`(严格前 3 条;全空回落 `[query]`)。

- [ ] **Step 1: 加 prompt + `_Expanded` + 函数**

```python
class _Expanded(BaseModel):
    queries: list[str] = Field(default_factory=list)

async def expand_queries(query: str) -> list[str]:
    try:
        r = await (EXPAND_QUERIES_PROMPT |
                   structured_model(_Expanded)).ainvoke({"query": query})
        qs = [q.strip() for q in (r.queries or []) if q and q.strip()]
    except Exception:
        qs = []
    return qs[:3] or [query]
```

- [ ] **Step 2: 写评估脚本 + 跑(需上游)**

3 条核心问;打印条数/内容。记录。

- [ ] **Step 3: Commit**

```bash
git add app/core/prompts.py app/core/query_understanding.py scripts/eval_expand.py
git commit -m "feat(ch06): Query 扩写 expand_queries"
```

---

## Task 7: fetch_order + list_user_orders + 缺单 interrupt

**Files:**
- Modify: `app/tools/business.py`
- Modify: `app/graph/nodes.py`
- Test: `tests/graph/test_ch06_nodes.py`
- Test: `tests/test_tools_mock.py`(回归)

**Interfaces:**
- Produces:`business.order_snapshot(order_id)`;`business.list_user_orders(user_id)`;`nodes._extract_order_id(text)`;`nodes.fetch_order(state)`。

- [ ] **Step 1: 失败测试**

```python
def test_extract_order_id():
    assert nodes._extract_order_id("订单1001的物流") == "1001"
    assert nodes._extract_order_id("尾号 20260701 那单") == "20260701"
    assert nodes._extract_order_id("我要退货") is None


@pytest.mark.asyncio
async def test_fetch_order_uses_id_in_query():
    out = await nodes.fetch_order({"resolved_query": "订单1001能退吗", "user_id": "u1"})
    assert out["order_id"] == "1001"
    assert out["order_data"]["order_id"] == "1001"
    assert out["trace"]["fetch_order"]["order_id"] == "1001"


@pytest.mark.asyncio
async def test_fetch_order_interrupts_when_missing(monkeypatch):
    monkeypatch.setattr(nodes.business, "list_user_orders",
                        lambda uid: [{"order_id": "1001", "product": "猫粮",
                                      "status": "已签收", "amount": 99}])
    from langgraph.graph import START, END, StateGraph
    from langgraph.checkpoint.memory import InMemorySaver
    from app.graph.state import ConversationState
    b = StateGraph(ConversationState)
    b.add_node("fetch_order", nodes.fetch_order)
    b.add_edge(START, "fetch_order")
    b.add_edge("fetch_order", END)
    graph = b.compile(checkpointer=InMemorySaver())
    out = await graph.ainvoke({"resolved_query": "我要退款", "user_id": "u1"},
                              {"configurable": {"thread_id": "t-fetch"}})
    payload = out["__interrupt__"][0].value
    assert payload["type"] == "select_order"
    assert payload["orders"][0]["order_id"] == "1001"


def test_list_user_orders_stable_and_queryable():
    from app.tools import business
    a = business.list_user_orders("u-42")
    b = business.list_user_orders("u-42")
    assert a == b and len(a) >= 2
    snap = business.order_snapshot(a[0]["order_id"])
    assert snap["order_id"] == a[0]["order_id"]
    assert snap["product"] == a[0]["product"]
```

- [ ] **Step 2: 跑失败** → FAIL。

- [ ] **Step 3: business.py 抽 `order_snapshot` + 加 `list_user_orders`**

`query_order` 改为 `return order_snapshot(order_id)`;`list_user_orders` 用 `random.Random(f"user_orders:{user_id}")` 生成 2-4 个 4 位 id,每项 `{order_id, product, status, amount}`。

- [ ] **Step 4: nodes.py 加 import + 节点**

```python
import re
from langgraph.types import interrupt
from app.tools import business

_ORDER_RE = re.compile(r"(?<!\d)(\d{4,})(?!\d)")

def _extract_order_id(text: str) -> str | None:
    m = _ORDER_RE.search(text or "")
    return m.group(1) if m else None

async def fetch_order(state) -> dict:
    oid = state.get("order_id") or _extract_order_id(
        state.get("resolved_query") or _user_text(state))
    if not oid:
        orders = business.list_user_orders(state.get("user_id", ""))
        oid = interrupt({"type": "select_order", "orders": orders})
    data = business.order_snapshot(oid)
    return {"order_id": oid, "order_data": data,
            "trace": trace_of(state, fetch_order={"order_id": oid})}
```

- [ ] **Step 5: 跑通过 + 工具回归**

`uv run pytest tests/graph/test_ch06_nodes.py tests/test_tools_mock.py -q` → PASS。

- [ ] **Step 6: Commit**

```bash
git add app/tools/business.py app/graph/nodes.py tests/graph/test_ch06_nodes.py
git commit -m "feat(ch06): fetch_order 抽单/缺单 interrupt + list_user_orders"
```

---

## Task 8: retrieve_policy(扩写多查 + 去重合并)

**Files:**
- Modify: `app/graph/nodes.py`
- Test: `tests/graph/test_ch06_nodes.py`

**Interfaces:**
- Produces:`retrieve_policy(state)` 写 `evidence`/`citations`,trace `retrieve_policy={queries,hits}`。

- [ ] **Step 1: 失败测试**

```python
@pytest.mark.asyncio
async def test_retrieve_policy_expands_dedups_merges(monkeypatch):
    async def fake_expand(q):
        return ["退货政策", "无理由退换货", "退货时限"]
    per_query = {
        "退货政策": [{"id": 1, "question": "退货", "answer": "7天无理由",
                     "rerank_score": 0.7, "section_path": "政策/退货",
                     "content_type": "policy"}],
        "无理由退换货": [
            {"id": 1, "question": "退货", "answer": "7天无理由",
             "rerank_score": 0.9, "section_path": "政策/退货", "content_type": "policy"},
            {"id": 2, "question": "运费", "answer": "质量问题商家承担",
             "rerank_score": 0.6, "section_path": "政策/运费", "content_type": "policy"}],
        "退货时限": [],
    }
    async def fake_search(q, **k):
        return per_query.get(q, [])
    monkeypatch.setattr(nodes.query_understanding, "expand_queries", fake_expand)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", fake_search)
    monkeypatch.setattr(nodes.retrieval, "arrange_head_tail", lambda h: h)

    out = await nodes.retrieve_policy({"resolved_query": "这个订单能退吗",
                                       "order_data": {"status": "已签收"}})
    assert [c["id"] for c in out["citations"]] == [1, 2]
    assert "[1]" in out["evidence"] and "[2]" in out["evidence"]
    assert out["trace"]["retrieve_policy"]["hits"] == 2
    assert out["trace"]["retrieve_policy"]["queries"] == ["退货政策", "无理由退换货", "退货时限"]
```

- [ ] **Step 2: 跑失败** → FAIL。

- [ ] **Step 3: 实现**

```python
from app.core import query_understanding, retrieval

async def retrieve_policy(state) -> dict:
    base = state.get("resolved_query") or _user_text(state)
    od = state.get("order_data") or {}
    seed = f"{base} {od.get('status', '')}".strip()
    queries = await query_understanding.expand_queries(seed)
    merged = {}
    for q in queries:
        for h in await retrieval.search_knowledge(q, strategy="hybrid_rerank", bm25_query=q):
            cur = merged.get(h["id"])
            if cur is None or h["rerank_score"] > cur["rerank_score"]:
                merged[h["id"]] = h
    ranked = sorted(merged.values(), key=lambda h: h["rerank_score"], reverse=True)
    arranged = retrieval.arrange_head_tail(ranked)
    citations = [{"n": i + 1, "id": h["id"], "chunk_id": h["id"],
                  "section_path": h["section_path"], "question": h["question"],
                  "answer": h["answer"], "content_type": h["content_type"]}
                 for i, h in enumerate(arranged)]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    return {"evidence": evidence, "citations": citations,
            "trace": trace_of(state, retrieve_policy={"queries": queries,
                                                      "hits": len(ranked)})}
```

- [ ] **Step 4: 跑通过 + Commit**

`uv run pytest tests/graph/test_ch06_nodes.py -k retrieve_policy -q` → PASS
```bash
git add app/graph/nodes.py tests/graph/test_ch06_nodes.py
git commit -m "feat(ch06): retrieve_policy 扩写多查+去重合并"
```

---

## Task 9: submit_refund + agent_tools 拦截 + _agent_messages 适配

**Files:**
- Modify: `app/tools/business.py`、`app/tools/registry.py`
- Modify: `app/core/prompts.py`(`REFUND_JUDGE_HINT`)
- Modify: `app/graph/nodes.py`
- Test: `tests/graph/test_ch06_nodes.py`、`tests/graph/test_agent_nodes.py`

**Interfaces:**
- Produces:`business.submit_refund(order_id, reason=None)`;`agent_tools` 拦截成 `{"type":"refund_form","draft":{...}}`;`_agent_messages` 有 evidence 即注入,refund 路追加 order_data + 判定指令。

- [ ] **Step 1: 失败测试**

```python
from langchain_core.messages import SystemMessage


@pytest.mark.asyncio
async def test_agent_tools_intercepts_submit_refund():
    ai = AIMessage(content="", tool_calls=[
        {"id": "r1", "name": "submit_refund", "args": {"order_id": "1001", "reason": None}}])
    out = await nodes.agent_tools({"messages": [ai], "order_id": "1001"})
    assert out["suggested_actions"] == [
        {"type": "refund_form", "draft": {"order_id": "1001", "reason": None}}]
    tm = out["messages"][0]
    assert tm.name == "submit_refund" and "退款" in tm.content


def test_agent_messages_injects_order_and_policy_on_refund():
    msgs = nodes._agent_messages({
        "route": "refund_flow",
        "order_data": {"order_id": "1001", "status": "已签收", "product": "猫粮 5kg"},
        "evidence": "[1] 退货: 7天无理由",
        "messages": [HumanMessage("这单能退吗")]})
    sys = msgs[0]
    assert isinstance(sys, SystemMessage)
    assert "7天无理由" in sys.content
    assert "猫粮 5kg" in sys.content and "submit_refund" in sys.content


def test_agent_messages_knowledge_path_still_injects_evidence():
    msgs = nodes._agent_messages({
        "route": "knowledge", "evidence": "[1] 运费: 满99包邮",
        "messages": [HumanMessage("运费多少")]})
    assert "满99包邮" in msgs[0].content
```

- [ ] **Step 2: 跑失败** → FAIL。

- [ ] **Step 3: 加工具 + 注册 + hint**

`RefundInput`、`@tool`、registry import/_ALL/NO_RETRY;`REFUND_JUDGE_HINT`(能退调 submit_refund;不能退说原因+引导人工;证据为空不臆造;订单数据随后)。

- [ ] **Step 4: 改 agent_tools + _agent_messages**

拦截分支:`order_id = tc["args"].get("order_id") or state.get("order_id", "")`;dedupe;合成 ToolMessage。
`_agent_messages`:有 evidence 就拼 `_KNOWLEDGE_EVIDENCE_HINT`;refund 路追加 hint + `json.dumps(order_data)`。

- [ ] **Step 5: 跑通过 + 回归**

`uv run pytest tests/graph/test_ch06_nodes.py tests/graph/test_agent_nodes.py -q` → PASS。

- [ ] **Step 6: Commit**

```bash
git add app/tools/business.py app/tools/registry.py app/core/prompts.py \
  app/graph/nodes.py tests/graph/test_ch06_nodes.py
git commit -m "feat(ch06): submit_refund 拦截为退款表单 + 证据/订单注入"
```

---

## Task 10: script_reply(闲聊/其他分文案)

**Files:**
- Modify: `app/core/prompts.py`
- Modify: `app/graph/nodes.py`
- Test: `tests/graph/test_ch06_nodes.py`

**Interfaces:**
- Produces:`script_reply(state)` 写 `answer` + trace `route=fallback_script`。

- [ ] **Step 1: 失败测试**

```python
@pytest.mark.asyncio
async def test_script_reply_by_intent():
    from app.core.prompts import CHITCHAT_REPLY_TEXT, SCRIPT_REPLY_OTHER
    chit = await nodes.script_reply({"intent": "闲聊"})
    assert chit["answer"] == CHITCHAT_REPLY_TEXT
    assert chit["trace"]["route"] == "fallback_script"
    other = await nodes.script_reply({"intent": "其他"})
    assert other["answer"] == SCRIPT_REPLY_OTHER
```

- [ ] **Step 2: 跑失败** → FAIL。

- [ ] **Step 3: 加文案 + 节点**

```python
SCRIPT_REPLY_OTHER = "抱歉,我不太确定您的意思。您可以把问题说得更具体些吗?比如您想咨询的商品、某个订单,或退款/售后问题。"

async def script_reply(state) -> dict:
    text = SCRIPT_REPLY_OTHER if state.get("intent") == "其他" else CHITCHAT_REPLY_TEXT
    return {"answer": text, "trace": trace_of(state, route="fallback_script")}
```

- [ ] **Step 4: 跑通过 + Commit**

```bash
git add app/core/prompts.py app/graph/nodes.py tests/graph/test_ch06_nodes.py
git commit -m "feat(ch06): script_reply 按意图分文案"
```

---

## Task 11: tickets 枚举加「退款」+ create-refund 端点

**Files:**
- Modify: `app/db/models.py`、`sql/ch02-ddl.sql`
- Create: `sql/ch06-ticket-type.sql`
- Modify: `app/schemas/actions.py`、`app/api/actions.py`
- Test: `tests/test_refund_api.py`

**Interfaces:**
- Produces:`POST /api/actions/create-refund` → `{ticket_no, status:"退款申请已提交"}`;reason 固定 5 类目。

- [ ] **Step 1: 失败测试**

```python
from fastapi.testclient import TestClient
from app.api import actions
from app.main import app


def test_create_refund_writes_ticket(monkeypatch):
    async def fake_create(cid, desc, ttype):
        assert cid == 7 and ttype == "退款"
        assert "1001" in desc and "质量问题" in desc
        return "T20260716001"
    monkeypatch.setattr(actions.repository, "create_ticket", fake_create)
    r = TestClient(app).post("/api/actions/create-refund", json={
        "conversation_id": 7, "order_id": "1001", "reason": "质量问题"})
    assert r.status_code == 200
    assert r.json() == {"ticket_no": "T20260716001", "status": "退款申请已提交"}


def test_create_refund_rejects_bad_reason():
    r = TestClient(app).post("/api/actions/create-refund", json={
        "conversation_id": 7, "order_id": "1001", "reason": "乱填"})
    assert r.status_code == 422
```

- [ ] **Step 2: 跑失败** → FAIL。

- [ ] **Step 3: 模型/DDL/迁移**

models.py `Enum("售后","投诉","咨询","退款")`;`sql/ch02-ddl.sql` 同步;`sql/ch06-ticket-type.sql` 用 `SET NAMES utf8mb4; ALTER TABLE tickets MODIFY COLUMN ticket_type ENUM('售后','投诉','咨询','退款') NOT NULL;`

- [ ] **Step 4: schema + 端点**

`CreateRefundRequest{conversation_id:int, order_id:str(min_length=1), reason:Literal[5类]}`、`CreateRefundResponse{ticket_no,status="退款申请已提交"}`;端点写 `tickets`。

- [ ] **Step 5: 跑通过**

`uv run pytest tests/test_refund_api.py -q` → PASS。

- [ ] **Step 6: 对本地库应用迁移**

用 SQLAlchemy/asyncmy 执行 `sql/ch06-ticket-type.sql`,确认 `SHOW COLUMNS FROM tickets LIKE 'ticket_type'` 含「退款」。

- [ ] **Step 7: Commit**

```bash
git add app/db/models.py sql/ch02-ddl.sql sql/ch06-ticket-type.sql \
  app/schemas/actions.py app/api/actions.py tests/test_refund_api.py
git commit -m "feat(ch06): tickets 枚举加退款 + create-refund 端点"
```

---

## Task 12: 图组装——5 出口 + refund_flow 链 + 退役 chitchat_reply

**Files:**
- Modify: `app/graph/build.py`、`app/graph/nodes.py`
- Test: `tests/graph/test_build.py`、`tests/graph/test_reply_nodes.py`

**Interfaces:**
- Produces:节点集含 `fetch_order`/`retrieve_policy`/`script_reply`,不含 `chitchat_reply`;五出口边 + `fetch_order→retrieve_policy→agent_llm`。

- [ ] **Step 1: 改测试**

test_build 节点清单加新节点、断言无 `chitchat_reply`;test_reply_nodes 把 chitchat 用例换成 script_reply。

- [ ] **Step 2: 跑失败** → FAIL。

- [ ] **Step 3: 改 build.py + 删 `chitchat_reply`**

节点、条件边映射、退款链、`script_reply→log`。删除 `nodes.chitchat_reply`;确认无引用。

- [ ] **Step 4: 跑通过 + Commit**

```bash
git add app/graph/build.py app/graph/nodes.py tests/graph/test_build.py tests/graph/test_reply_nodes.py
git commit -m "feat(ch06): 图组装 5 出口 + refund_flow 链"
```

---

## Task 13: 运行时——resume + interrupt 事件 + 未决中断

**Files:**
- Modify: `app/graph/runtime.py`
- Test: `tests/graph/test_runtime.py`

**Interfaces:**
- Produces:`run_turn` 带 `interrupt`;`resume_turn`;`stream_resume`;`stream_turn` 中断事件(含 conversation_id)+ 未决中断重吐。

- [ ] **Step 1: 失败测试**

- `test_stream_turn_emits_interrupt_event`:FakeGraph.astream 产 `("updates", {"__interrupt__": (Intr({...}),)})`;断言事件 `kind/orders/conversation_id`,且无 `done`。
- `test_resume_turn_drives_command`:FakeGraph.ainvoke 收到 `Command`,resume=1001;返回 conversation_id/state。
- `test_stream_turn_reemits_pending_interrupt`:`aget_state` 返回 `.interrupts=(Intr({...}),)`;FakeGraph.astream 若被调则失败;断言只吐 interrupt 事件。

- [ ] **Step 2: 跑失败** → FAIL。

- [ ] **Step 3: 实现**

- `DETERMINISTIC_ANSWER_NODES` 换 `script_reply`;`CITATION_NODES={"forced_rag","retrieve_policy"}`。
- `_stream_events(cid, source)`:中断检测、事件映射、actions 合并去重、done。
- `stream_turn`:pending 检测 → 重吐 interrupt;否则 `_stream_events`。
- `stream_resume(cid, value)`。
- `run_turn`:pending 检测 → 返回空 answer + interrupt;否则 ainvoke + turn_messages + interrupt。
- `resume_turn`。
- `_interrupt_orders` / `_pending_payload` helpers。

- [ ] **Step 4: 跑通过(含 ch05 回归)**

`uv run pytest tests/graph/test_runtime.py -q` → PASS。

- [ ] **Step 5: Commit**

```bash
git add app/graph/runtime.py tests/graph/test_runtime.py
git commit -m "feat(ch06): runtime resume/interrupt + 未决中断处理"
```

---

## Task 14: 两入口接线

**Files:**
- Modify: `app/api/chat.py`、`app/api/actions.py`、`app/api/agent.py`
- Modify: `app/schemas/actions.py`、`app/schemas/agent.py`
- Test: `tests/test_refund_api.py`、`tests/test_chat_api.py`、`tests/test_agent_api.py`

**Interfaces:**
- Produces:chat SSE `interrupt` 帧;`POST /api/actions/resume` SSE;`AgentResponse.interrupt`。

- [ ] **Step 1: 失败测试**

- resume 端点(monkeypatch `actions.runtime.stream_resume`)返回 delta/actions/[DONE]。
- chat interrupt 帧(fake_stream 产 `{"type":"interrupt",...}`)。
- agent interrupt 字段(fake_run 返回 `interrupt=[...]`)。

- [ ] **Step 2: 跑失败** → FAIL。

- [ ] **Step 3: 实现**

- chat: `elif ev["type"] == "interrupt": yield _sse({"event":"interrupt","kind":...,"orders":...,"conversation_id":...})`。
- actions: `ResumeRequest`;`/api/actions/resume` StreamingResponse,事件同 chat;错误帧。
- agent: `interrupt=out.get("interrupt")`;schema 加字段。

- [ ] **Step 4: 跑通过 + 回归**

`uv run pytest tests/test_refund_api.py tests/test_actions_api.py tests/test_agent_api.py tests/test_chat_api.py -q` → PASS。

- [ ] **Step 5: Commit**

```bash
git add app/api/chat.py app/api/actions.py app/api/agent.py \
  app/schemas/actions.py app/schemas/agent.py \
  tests/test_refund_api.py tests/test_chat_api.py tests/test_agent_api.py
git commit -m "feat(ch06): chat interrupt 帧 + resume SSE + agent interrupt 字段"
```

---

## Task 15: 前端——订单选择器卡片 + 退款表单(Vibe Coding)

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- Consumes:SSE `interrupt`(`kind/orders/conversation_id`)、`actions`(`refund_form`);`POST /api/actions/resume`、`POST /api/actions/create-refund`。

- [ ] **Step 1: 订单选择器**

`parseFrame` 后分发 `payload.event === "interrupt"`:保存 conversation_id;在当前 bot 气泡下渲染订单卡片(订单号/商品/状态/金额);点选 → 禁用整组 → POST resume → 用同一套 SSE 消费函数把续流渲染到新 bot 气泡。

- [ ] **Step 2: 退款表单**

`renderActions` 加 `refund_form`:按钮「提交退款工单」→ 弹窗(订单号只读 + 原因下拉:七天无理由/质量问题/发错货/不想要了/其他,不预选必选)→ POST create-refund → 显示退款单号并禁用按钮。

- [ ] **Step 3: 样式**

`.order-card`(可点、hover 高亮、选中/禁用灰态);复用 `.action-bar/.action-btn/.ticket-form`。

- [ ] **Step 4: 浏览器验收**

重启服务(PORT=8002);浏览器按验收 4 真跑:不带订单号问「我要退款」→ 弹卡片 → 点选 → 续流 → 能退出退款按钮 → 表单提交 → 显示退款单号;截图存 `dev-notes/ch06-acceptance4-*.png`。

- [ ] **Step 5: Commit**

```bash
git add app/static/index.html && git add -f dev-notes/ch06-acceptance4-*.png
git commit -m "feat(ch06): 前端订单选择器卡片 + 退款表单"
```

---

## Task 16: 四验收端到端评估 + 全量回归 + 收尾

**Files:**
- Create: `scripts/eval_ch06.py`
- Modify: `Makefile`、`README.md`、`dev-notes/ch06.md`

**Interfaces:**
- Consumes:`/api/agent`、`/api/actions/resume`。

- [ ] **Step 1: 写 eval_ch06.py**

验收 1 多轮(独立会话):物流→退款→物流,断言每轮 trace.route;验收 2 归入 eval_intent;验收 3 带订单号退款,断言 trace `route=refund_flow` 且 `retrieve_policy` 存在、resolved_query 含订单号;验收 4 新会话「我要退款」→ `interrupt` 非空 → POST resume SSE → 收到 delta/done。

- [ ] **Step 2: Makefile + README**

`eval-ch06`、`smoke-interrupt`、`eval-coref`、`eval-expand`;README 补演示命令。

- [ ] **Step 3: 真跑四验收 + 三个 prompt eval**

重启服务;`BASE_URL=http://localhost:8002 make eval-ch06`;`make eval-intent`/`eval-coref`/`eval-expand`;结果记 dev-notes。

- [ ] **Step 4: 全量回归**

`uv run pytest -q` → 全绿。

- [ ] **Step 5: Commit + dev-notes finish**

```bash
git add scripts/eval_ch06.py Makefile README.md && git add -f dev-notes/ch06.md
git commit -m "test(ch06): 四验收端到端评估 + README/dev-notes 收尾"
```

---

## Self-Review

**Spec 覆盖:** §1 目标 → 全任务;§3/§3.1 五出口 → Task 3/12;§4 coref → Task 5;§5 扩写 → Task 6/8;§6 意图四件套 → Task 4;§7 refund_flow → Task 7/8/9;§8 interrupt/resume → Task 1/7/13/14/15;§9 退款单 → Task 11;§10 State → Task 2;§11 前端 → Task 15;§12 验证分层 → 各任务;§13 D1–D12 → 对应任务;§14 R1–R5 → Task 1/7/4/11、eval;§15 验收 → Task 16。无遗漏。

**占位符扫描:** 无 TBD/TODO;代码步给签名与关键实现,测试步给真实断言。中断 surface 已实测,不再是活扣。

**类型一致性:** State 四字段 ↔ 节点读写;`classify -> dict` ↔ 节点/eval;`INTENT_TO_ROUTE` 五值 ↔ build 映射键;`refund_form` 动作 ↔ 前端;`order_snapshot` 三处同源;事件 `interrupt` dict ↔ chat/resume SSE;trace 键 `intent_confidence` 与 `confidence` 分离。
