# Ch08 即插即用工具系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把工具层从写死的内置清单升级为即插即用系统:统一 ToolSpec 注册中心(内置 + MCP)、JSON Schema 校验、读写权限、统一执行引擎(超时/重试/分诊/格式化/审计)、两台业务 MCP Server、create_ticket interrupt 确认流。

**Architecture:** 方案 A(spec §3):进程内 ToolSpec 注册表;内置工具放 `app/tools/builtin/` 包,服务启动时 `pkgutil` 扫描导入即注册;MCP 工具每轮用 `MultiServerMCPClient.get_tools()` 现拉、与内置合并;`engine.py` 六段管道(查工具 → 校验 → 权限门 → 执行 → 分诊 → 格式化 + 审计);`agent_tools` 顶置 interrupt 做建工单确认。

**Tech Stack:** MCP 官方 Python SDK v2(`MCPServer`,Streamable HTTP)、`langchain-mcp-adapters` `MultiServerMCPClient`、`jsonschema`(Draft 2020-12)、既有 FastAPI + SQLAlchemy 2 async + LangGraph。

**Spec:** `docs/superpowers/specs/2026-11-14-ch08-tool-system-design.md`

## Global Constraints

- **开工前先建分支**:`git checkout -b ch08-tool-system`(当前在 `main`;章节完成后分支保留,合并 main 等用户明说)。
- 技术选型定死(用户工作要求 4):MCP Server = 官方 Python SDK Streamable HTTP;Client = `langchain-mcp-adapters` `MultiServerMCPClient`。走不通停下来问,不许换方案。
- 库 API 已用 Context7 核对,实装以 `mcp 2.3.0` / `langchain-mcp-adapters 0.3.1` 为准;若装出来的版本不同,先记 dev-notes 再按实装调整。
- JSON 序列化一律 `ensure_ascii=False`;审计写失败不许反拦工具执行;写操作(`create_ticket`)恒不自动重试。
- 每任务完成即 commit,并追记 `dev-notes/ch08.md`(不许收尾补记);commit message 用简洁英文,结尾带 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。
- 红线:修改 `.env.example` 前单独向用户确认;不 push、不 rebase、不删分支。
- 测试命令 `uv run pytest <path> -v`,全量 `make test`;测试连真 MySQL 测试库(docker `dianxiaoer-mysql` 需在跑)。
- ch05 投诉按钮建单路径(`complaint_reply` + `POST /api/actions/create-ticket`)一行不动。
- 前端 Task 8 走 Vibe Coding,不套 TDD;prompt 改动用标注样例(Task 9)验证。
- ch08 会改 `AGENT_SYSTEM` 与工具清单,ch07 预算常量与 `tests/test_context_budget.py` 必须跟着重测(见 Task 2/5/6),ch07 历史演示数字不再复现。

## Review Focus

1. **MCP Server 未启动或挂起**:对话仍能用内置工具正常作答,不 500、不无限等待。测试:Task 5 `test_one_server_down_degrades_gracefully` + Task 4 集成测试。
2. **模型发缺参/错型工具调用**:回灌可读错误而不是抛异常;模型能追问或改参。测试:Task 3 `test_validation_blocks_and_feeds_back`。
3. **建单预览期间取消、重复点按钮、resume 值畸形**:不重复建单、不崩。测试:Task 6 `test_confirm_false_denied_and_audited` + `test_resume_value_not_dict_treated_as_cancel`。
4. **写操作超时或审计库故障**:用户得到如实告知;审计缺失不反拦执行。测试:Task 3 `test_write_confirmed_executes_and_not_retried` + `test_audit_failure_never_blocks_execution`。
5. **工具清单动态变化/重名**:Server 新增工具下一轮即见;重名内置优先、不出现二选一。测试:Task 2 `test_duplicate_register_is_dropped_with_warning` + Task 5 `test_get_all_specs_merges_builtin_wins`。

## File Structure

**新建**
- `app/tools/builtin/{__init__,orders,faq,tickets,refunds}.py` — 内置工具,模块尾部注册
- `app/tools/engine.py` — 统一执行引擎
- `app/tools/mcp_client.py` — MCP Client + 结果格式化
- `mcp_servers/{__init__,logistics_server,aftersales_server}.py` — 两台业务 MCP Server
- `sql/ch08-ddl.sql` — `tool_audit_logs`
- `tests/db/test_tool_audit.py`、`tests/tools/test_{registry,engine,mcp_client,mcp_servers}.py`
- `tests/graph/test_ch08_confirm_ticket.py`、`tests/graph/test_ch08_ticket_followup.py`
- `scripts/demo_ch08_promotions.py.txt`、`scripts/eval_ch08.py`

**修改**
- `app/tools/registry.py`(重造)、`app/tools/business.py`(瘦身)、`app/main.py`(启动扫描)
- `app/config.py`、`app/db/models.py`、`app/db/repository.py`、`tests/conftest.py`
- `app/graph/nodes.py`、`app/graph/runtime.py`、`app/graph/state.py`、`app/graph/routing.py`
- `app/core/prompts.py`、`app/core/intent.py`
- `app/api/sse.py`、`app/api/actions.py`、`app/schemas/actions.py`、`app/schemas/agent.py`
- `app/static/index.html`(Vibe Coding)
- `Makefile`、`scripts/dev.sh`、`scripts/eval_intent.py`、`scripts/eval_ch06.py`、`scripts/smoke_ch07.py`、`scripts/bare_agent_loop.py`

**删除**
- `app/tools/infra.py`(Task 5)、`tests/test_tools_infra.py`(Task 5,用例已迁 engine/registry)

---

### Task 1: 审计地基(依赖 + settings + ToolAuditLog + repository + 测试库接线)

**Files:**
- Modify: `pyproject.toml`, `uv.lock`(uv add)
- Modify: `app/config.py`(ch07 预算块后追加 ch08 块)
- Modify: `app/db/models.py`(文件尾加 `ToolAuditLog`)
- Modify: `app/db/repository.py`(文件尾加 `insert_tool_audit`)
- Modify: `tests/conftest.py`(`_DDL_FILES` / `_TABLES`)
- Create: `sql/ch08-ddl.sql`(桌面 `ch_08.sql` 原样复制)
- Create: `tests/db/__init__.py`、`tests/db/test_tool_audit.py`
- Modify: `.env.example`(先向用户确认)

**Interfaces:**
- Produces: `repository.insert_tool_audit(conversation_id: int | None, tool_call_id: str | None, tool_name: str, tool_source: str, mcp_server: str | None, arguments: dict | None, result_summary: str | None, status: str, error_message: str | None, retry_count: int, duration_ms: int | None) -> None`
- Produces settings:`mcp_logistics_url="http://127.0.0.1:8101/mcp"`、`mcp_aftersales_url="http://127.0.0.1:8102/mcp"`、`tool_default_timeout=5.0`、`mcp_tool_timeout=10.0`、`mcp_discovery_timeout=3.0`、`tool_max_retries=2`、`demo_ticket_delay_seconds=0.0`

- [ ] **Step 1: 装依赖并核对实装 API**

```bash
uv add mcp langchain-mcp-adapters jsonschema
uv run python -c "import mcp, langchain_mcp_adapters, jsonschema; print(mcp.__file__)"
uv run python -c "from mcp.server import MCPServer; print('MCPServer OK')"
uv run python -c "from langchain_mcp_adapters.client import MultiServerMCPClient; import inspect; print('handle_tool_errors' in inspect.signature(MultiServerMCPClient.__init__).parameters)"
```

Expected: 前两条通过;第三条打印 `True`。若 `MCPServer` 导入失败,改用 `from mcp.server.fastmcp import FastMCP as MCPServer`,并把版本差异记入 dev-notes。

- [ ] **Step 2: settings**

`app/config.py` 在 ch07 块之后追加:

```python
    # ch08 工具系统
    mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"
    mcp_aftersales_url: str = "http://127.0.0.1:8102/mcp"
    tool_default_timeout: float = 5.0
    mcp_tool_timeout: float = 10.0
    mcp_discovery_timeout: float = 3.0
    tool_max_retries: int = 2
    demo_ticket_delay_seconds: float = 0.0
```

- [ ] **Step 3: 写失败测试 `tests/db/test_tool_audit.py`**

```python
from sqlalchemy import select

from app.db import repository
from app.db.models import ToolAuditLog


async def test_insert_tool_audit_minimal(db_session_factory, db_clean):
    await repository.insert_tool_audit(
        conversation_id=None, tool_call_id=None, tool_name="query_order",
        tool_source="builtin", mcp_server=None, arguments={"order_id": "1001"},
        result_summary="{}", status="成功", error_message=None,
        retry_count=0, duration_ms=12)
    async with db_session_factory() as s:
        row = (await s.execute(select(ToolAuditLog))).scalars().one()
    assert row.tool_name == "query_order" and row.status == "成功"
    assert row.conversation_id is None
    assert row.arguments == {"order_id": "1001"}


async def test_insert_tool_audit_all_statuses(db_session_factory, db_clean):
    for st in ["成功", "失败", "超时", "校验拦下", "权限拒绝"]:
        await repository.insert_tool_audit(
            conversation_id=1, tool_call_id="tc-1", tool_name="create_ticket",
            tool_source="mcp", mcp_server="logistics", arguments=None,
            result_summary=None, status=st, error_message="原因",
            retry_count=2, duration_ms=None)
    async with db_session_factory() as s:
        rows = (await s.execute(select(ToolAuditLog))).scalars().all()
    assert {r.status for r in rows} == {"成功", "失败", "超时", "校验拦下", "权限拒绝"}
```

- [ ] **Step 4: 跑测试确认失败**

Run: `uv run pytest tests/db/test_tool_audit.py -v`
Expected: FAIL(`ToolAuditLog` / `insert_tool_audit` 不存在)。

- [ ] **Step 5: 实现 DDL、模型、repository、测试库接线**

`sql/ch08-ddl.sql` 原样复制桌面 `CH_08/ch_08.sql`(文件内无 ASCII `;` 注释,`split_statements` 可正常切分)。

`app/db/models.py` 尾部:

```python
class ToolAuditLog(Base):
    """ch08 工具调用审计:内置与 MCP 都记,被权限拒、被校验拦的调用同样落一条;不挂外键。"""

    __tablename__ = "tool_audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128))
    tool_source: Mapped[str] = mapped_column(Enum("builtin", "mcp"))
    mcp_server: Mapped[str | None] = mapped_column(String(64), nullable=True)
    arguments: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Enum("成功", "失败", "超时", "校验拦下", "权限拒绝"))
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, server_default="0")
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

`app/db/repository.py` 尾部:

```python
# ---------------- ch08 工具审计 ----------------


async def insert_tool_audit(
    conversation_id: int | None, tool_call_id: str | None, tool_name: str,
    tool_source: str, mcp_server: str | None, arguments: dict | None,
    result_summary: str | None, status: str, error_message: str | None,
    retry_count: int, duration_ms: int | None,
) -> None:
    """工具调用审计落一条;调用方(engine)自行 try/except——审计失败不许反拦工具执行。"""
    async with db.async_session() as s:
        s.add(ToolAuditLog(
            conversation_id=conversation_id, tool_call_id=tool_call_id,
            tool_name=tool_name, tool_source=tool_source, mcp_server=mcp_server,
            arguments=arguments, result_summary=result_summary, status=status,
            error_message=error_message, retry_count=retry_count, duration_ms=duration_ms))
        await s.commit()
```

`tests/conftest.py`:`_DDL_FILES` 加 `sql/ch08-ddl.sql`;`_TABLES` 首部加 `"tool_audit_logs"`(子表先删)。

- [ ] **Step 6: 跑测试 + dev 库应用 DDL**

Run: `uv run pytest tests/db/test_tool_audit.py -v`
Expected: 2 passed。

```bash
docker exec -i dianxiaoer-mysql mysql --default-character-set=utf8mb4 -uroot -proot dianxiaoer < sql/ch08-ddl.sql
docker exec -i dianxiaoer-mysql mysql -uroot -proot dianxiaoer -e "DESC tool_audit_logs;"
```

Expected: 13 列。

- [ ] **Step 7: 确认后改 `.env.example`**

先向用户确认(红线),确认后在尾部追加:

```
# ch08 工具系统
MCP_LOGISTICS_URL=http://127.0.0.1:8101/mcp
MCP_AFTERSALES_URL=http://127.0.0.1:8102/mcp
DEMO_TICKET_DELAY_SECONDS=0
```

- [ ] **Step 8: Commit + dev-notes**

```bash
git add pyproject.toml uv.lock app/config.py app/db/models.py app/db/repository.py tests/conftest.py tests/db sql/ch08-ddl.sql .env.example
git commit -m "feat(ch08): audit log foundation and tool-system settings"
```

追记 `dev-notes/ch08.md`:Task 1 完成、实际依赖版本、`.env.example` 确认结果。

---

### Task 2: ToolSpec 注册中心 + builtin 包(注册即定义,query_logistics 下线)

**Files:**
- Rewrite: `app/tools/registry.py`
- Create: `app/tools/builtin/{__init__,orders,faq,tickets,refunds}.py`
- Modify: `app/tools/business.py`(瘦身成 mock 纯函数)
- Modify: `app/main.py`(lifespan 启动扫描)
- Modify: `app/config.py`(`tool_schema_tokens` 重测)
- Modify: `tests/test_tools_mock.py`、`tests/test_tool_ticket.py`、`tests/test_tools_infra.py`、`tests/test_context_budget.py`
- Create: `tests/tools/__init__.py`、`tests/tools/test_registry.py`

**Interfaces:**
- Produces:
  - `@dataclass ToolSpec(name, description, json_schema, tool, permission, source, mcp_server=None, timeout=None, inject_conversation=False, retryable=True, format_result=None)`
  - `registry.register(spec) -> None`(重名先到者保留 + warning)
  - `registry.builtin_specs() -> list[ToolSpec]`(惰性扫描)
  - `registry.get_builtin_spec(name) -> ToolSpec | None`
  - `registry.permission_for(name) -> str`(`WRITE_TOOLS` 独裁)
  - `registry.spec_from_langchain_tool(tool, *, source, mcp_server=None, timeout=None, inject_conversation=False, retryable=True, format_result=None) -> ToolSpec`
- 过渡兼容(本任务保留,Task 5 删):`get_all_tools()`、`get_tool(name)`、`NO_RETRY`、`INJECT_CONVERSATION`、`TOOL_TIMEOUTS`(旧 `infra.py` 与 `nodes.py` 仍在用)。

- [ ] **Step 1: 写失败测试 `tests/tools/test_registry.py`**

```python
import logging

from app.tools import registry


def test_builtin_scan_registers_five_tools_without_query_logistics():
    names = {s.name for s in registry.builtin_specs()}
    assert names == {"query_order", "query_product", "query_faq",
                     "create_ticket", "submit_refund"}


def test_every_spec_has_three_essentials():
    for s in registry.builtin_specs():
        assert s.name and s.description
        assert isinstance(s.json_schema, dict) and s.json_schema.get("properties") is not None


def test_permissions_only_trust_our_side():
    by = {s.name: s for s in registry.builtin_specs()}
    assert by["create_ticket"].permission == "write"
    assert all(s.permission == "read" for n, s in by.items() if n != "create_ticket")
    assert registry.permission_for("mcp_random_tool") == "read"


def test_create_ticket_schema_excludes_injected_conversation_id():
    spec = registry.get_builtin_spec("create_ticket")
    assert spec.inject_conversation is True
    assert "conversation_id" not in spec.json_schema.get("properties", {})


def test_duplicate_register_is_dropped_with_warning(caplog):
    spec = registry.builtin_specs()[0]
    dup = registry.ToolSpec(
        name=spec.name, description="dup",
        json_schema={"type": "object", "properties": {}},
        tool=spec.tool, permission="read", source="builtin")
    with caplog.at_level(logging.WARNING, logger="app.tools.registry"):
        registry.register(dup)
    assert registry.get_builtin_spec(spec.name).description != "dup"
    assert any("重名" in r.message for r in caplog.records)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/tools/test_registry.py -v`
Expected: FAIL(`builtin_specs` 不存在)。

- [ ] **Step 3: 迁移内置工具**

`app/tools/business.py` 只留 `order_snapshot(order_id)` 与 `list_user_orders(user_id)`(签名不动;文件头注释改为「mock 数据源(纯函数),工具实现见 app/tools/builtin/」)。

`app/tools/builtin/orders.py`:迁入 `OrderInput / ProductInput / query_order / query_product`,尾部:

```python
registry.register(registry.spec_from_langchain_tool(query_order, source="builtin"))
registry.register(registry.spec_from_langchain_tool(query_product, source="builtin"))
```

`faq.py`:迁入 `FaqInput / query_faq`,尾部 `timeout=45.0, retryable=False`(重 RAG,重试会叠加分钟级等待)。

`tickets.py`:迁入 `create_ticket`,加验收 6 延迟钩子,描述改为:

```python
    """创建人工工单。仅当用户明确要求建工单/要求人工跟进时才调用;调用前必须确认 description
    (问题描述)已从用户处问清,信息不足时先向用户追问,严禁编造或用占位文本。
    ticket_type 从 售后/投诉/咨询 中选;工单关联的会话号由系统注入,你不要传。"""
```

尾部 `inject_conversation=True`。函数体在落库前加:

```python
    if settings.demo_ticket_delay_seconds > 0:
        await asyncio.sleep(settings.demo_ticket_delay_seconds)
```

`refunds.py`:迁入 `RefundInput / submit_refund`,尾部注册。

- [ ] **Step 4: 重造 `app/tools/registry.py`**

按 spec §4.1 实现:`ToolSpec`、`WRITE_TOOLS`、`permission_for`、`_json_schema_of`(MCP dict 直用;内置取 `tool_call_schema.model_json_schema()`)、`spec_from_langchain_tool`、`register`、`scan_builtin`(幂等)、`builtin_specs`、`get_builtin_spec`。过渡兼容段:

```python
NO_RETRY = {"create_ticket", "query_faq", "submit_refund"}
INJECT_CONVERSATION = {"create_ticket"}
TOOL_TIMEOUTS = {"query_faq": 45.0}


def get_all_tools() -> list[BaseTool]:
    return [s.tool for s in builtin_specs()]


def get_tool(name: str) -> BaseTool | None:
    spec = get_builtin_spec(name)
    return spec.tool if spec else None
```

`app/main.py` lifespan 里 `await runtime.init_graph()` 前加:

```python
    from app.tools import registry as tool_registry
    tool_registry.scan_builtin()   # ch08:内置工具服务启动时登记
```

- [ ] **Step 5: 更新受影响旧测试**

- `tests/test_tools_infra.py::test_registry_has_six_tools` → 期望集合去掉 `query_logistics`,改名 `test_registry_has_five_tools`。
- `tests/test_tools_mock.py`:工具 import 改 `from app.tools.builtin.orders import query_order, query_product`;删除 `query_logistics` 相关用例(MCP 侧 Task 4 另测)。
- `tests/test_tool_ticket.py`:`from app.tools.builtin.tickets import create_ticket`。
- `tests/test_context_budget.py`:`test_budget_constants_match_prompt_and_tools` 的 `get_all_tools()` 暂保留(Task 5 再改 import)。

- [ ] **Step 6: 重测 `tool_schema_tokens` 并同步期望**

Run:

```bash
uv run python - <<'EOF'
import json
from langchain_core.utils.function_calling import convert_to_openai_tool
from app.config import Settings
from app.core import context_budget as cb
from app.tools import registry
s = Settings(_env_file=None, chat_model="m", chat_base_url="u", chat_api_key="k", embed_api_key="k")
tools = [x.tool for x in registry.builtin_specs()]
print("tool_schema_tokens =", cb.estimate_text_tokens(
    json.dumps([convert_to_openai_tool(t) for t in tools], ensure_ascii=False), s))
for kw in (
    dict(model_context_window=18000, max_output_tokens=2000, max_user_input_tokens=2000,
         max_agent_steps=3, tool_result_max_tokens=1200, rerank_top_k=5),
    dict(model_context_window=18000),
    dict(),
):
    b = cb.compute_budget(s.model_copy(update=kw))
    print(kw, (b.window, b.fixed, b.react_peak, b.history, b.layer1, b.layer2))
EOF
```

把打印值写入 `settings.tool_schema_tokens`,并更新 `tests/test_context_budget.py` 三组期望元组(数值以打印为准,断言结构不变);`test_demo_budget_matches_acceptance_numbers` 改名为 `test_budget_with_demo_config`(ch07 验收数字已随工具清单变化,名字不能再声称复现旧数字)。Expected: 该测试与全量测试恢复绿。

- [ ] **Step 7: 跑测试**

Run: `uv run pytest tests/tools/test_registry.py tests/test_tools_infra.py tests/test_tools_mock.py tests/test_tool_ticket.py tests/test_context_budget.py -v`
Expected: 全 PASS。

- [ ] **Step 8: Commit + dev-notes**

```bash
git add -A app/tools app/main.py app/config.py tests
git commit -m "feat(ch08): ToolSpec registry and builtin package scanning"
```

追记 dev-notes:迁移了哪些文件、`tool_schema_tokens` 前后值、`query_logistics` 下线。

---

### Task 3: 统一执行引擎 `app/tools/engine.py`

**Files:**
- Create: `app/tools/engine.py`
- Create: `tests/tools/test_engine.py`

**Interfaces:**
- Consumes: Task 2 `ToolSpec`;Task 1 `repository.insert_tool_audit`、settings。
- Produces:
  - `@dataclass ToolRun(tool_call_id: str, name: str, ok: bool, tool_message: ToolMessage, status: str, retry_count: int = 0, duration_ms: int = 0)`
  - `async engine.execute_tool_call(tool_call: dict, conversation_id: int, specs: dict[str, ToolSpec], *, confirmed: bool = False, deny_note: str | None = None) -> ToolRun`
  - `engine.validate_args(spec: ToolSpec, args: dict) -> str | None`

- [ ] **Step 1: 写失败测试 `tests/tools/test_engine.py`**

用 spec §7 的用例集,至少覆盖:

```python
import asyncio
import pytest
from app.tools import engine
from app.tools.registry import ToolSpec

SCHEMA = {"type": "object", "properties": {"order_id": {"type": "string"}},
          "required": ["order_id"]}


def _spec(name="query_order", *, permission="read", source="builtin", tool=None,
          schema=SCHEMA, timeout=None, inject=False, retryable=True, fmt=None):
    return ToolSpec(name=name, description="测试工具", json_schema=schema, tool=tool,
                    permission=permission, source=source, timeout=timeout,
                    inject_conversation=inject, retryable=retryable, format_result=fmt)


def _tool(fn, name="query_order"):
    return type("T", (), {"name": name, "ainvoke": staticmethod(fn)})()


@pytest.fixture()
def audits(monkeypatch):
    rows = []

    async def fake_audit(**kw):
        rows.append(kw)

    monkeypatch.setattr(engine.repository, "insert_tool_audit", fake_audit)
    return rows


async def test_unknown_tool(audits): ...
async def test_validation_blocks_and_feeds_back(audits): ...
async def test_write_without_confirmation_denied(audits): ...
async def test_write_confirmed_executes_and_not_retried(audits): ...
async def test_transient_timeout_retries_then_gives_up(audits): ...
async def test_business_error_not_retried(audits): ...
async def test_retry_succeeds_second_attempt(audits): ...
async def test_retryable_false_never_retries(audits): ...
async def test_format_result_hook_translates_enum(audits): ...
async def test_audit_failure_never_blocks_execution(monkeypatch): ...
async def test_inject_conversation_after_validation(audits): ...
```

断言口径(spec §4.4):校验拦下 `status="校验拦下"` 且 `"参数校验未通过" in content`;未确认写 `权限拒绝`;写失败调用次数 1、`retry_count=0`;超时重试 2 次、审计 `status="超时"`、`duration_ms is not None`;`ConnectionError` 重试后成功 `retry_count=1`;业务 `ValueError` 不重试;`retryable=False` 的只读超时也只调用一次;格式化钩子输出 `运输中` 且不含 `internal_ref`;审计抛异常时工具仍 `ok=True`;注入发生在校验之后(schema 不含 `conversation_id` 也能执行,工具看到 `conversation_id=7`)。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/tools/test_engine.py -v`
Expected: FAIL(`app.tools.engine` 不存在)。

- [ ] **Step 3: 实现 `app/tools/engine.py`**

按 spec §4.4 实现。关键点:

```python
_TRANSIENT = (asyncio.TimeoutError, TimeoutError, ConnectionError, httpx.TransportError)
_SUMMARY_LIMIT = 500


def validate_args(spec, args) -> str | None:
    err = best_match(Draft202012Validator(spec.json_schema).iter_errors(args))
    if err is None:
        return None
    where = f"(字段 {err.json_path})" if err.json_path != "$" else ""
    return f"{err.message}{where}"
```

执行段:

```python
    retries = 0 if (spec.permission == "write" or not spec.retryable) else settings.tool_max_retries
```

`_format_content` 先解 `tuple`/`str`,再 `format_result`,最后 `json.dumps(ensure_ascii=False, default=str)`。

`_audit` 整体 try/except;未知工具 `tool_source="builtin"`。

- [ ] **Step 4: 跑测试 + 全量**

Run: `uv run pytest tests/tools/test_engine.py -v` → 全 PASS。
Run: `uv run pytest -q` → 与 ch07 基线一致(除 Task 2 已更新的预算期望)。

- [ ] **Step 5: Commit + dev-notes**

```bash
git add app/tools/engine.py tests/tools/test_engine.py
git commit -m "feat(ch08): unified tool execution engine with validation, permissions and audit"
```

---

### Task 4: 两台业务 MCP Server(独立进程,Streamable HTTP)+ 起停脚本

**Files:**
- Create: `mcp_servers/__init__.py`、`mcp_servers/logistics_server.py`、`mcp_servers/aftersales_server.py`
- Modify: `Makefile`(`mcp-up` / `mcp-down`)、`scripts/dev.sh`
- Create: `tests/tools/test_mcp_servers.py`

**Interfaces:**
- Produces: `http://127.0.0.1:8101/mcp`(`query_logistics`)、`http://127.0.0.1:8102/mcp`(`query_warranty` / `query_return_status`);mock 返回含内部枚举码 `status_code` / `warranty_code` / `return_code`;env `MOCK_DELAY_SECONDS` 注入延迟、`PORT` 覆盖端口。

- [ ] **Step 1: 写两台 Server**

`logistics_server.py` 骨架:

```python
try:
    from mcp.server import MCPServer
except ImportError:                      # 旧版 SDK 兜底
    from mcp.server.fastmcp import FastMCP as MCPServer

mcp = MCPServer("logistics")
_DELAY = float(os.environ.get("MOCK_DELAY_SECONDS", "0"))


@mcp.tool()
async def query_logistics(
    tracking_no: Annotated[str, Field(description="物流单号(形如 SF 开头),需先用 query_order 查订单拿到该单号")],
) -> dict:
    """用物流单号查询物流状态、当前位置和轨迹。用于用户询问物流/快递到哪了时。
    物流单号不是订单号,需先用 query_order 查订单拿到 tracking_no,再调用本工具。"""
    if _DELAY > 0:
        await asyncio.sleep(_DELAY)
    rng = random.Random(f"logistics:{tracking_no}")
    code = rng.choice(["PICKED_UP", "IN_TRANSIT", "DELIVERING", "DELIVERED"])
    city = rng.choice(["深圳", "广州", "杭州", "上海", "成都"])
    return {"tracking_no": tracking_no, "status_code": code, "current_city": city,
            "trace": [f"{city}分拨中心 已发出", f"内部状态码:{code}"],
            "carrier_code": "SF-EXP-01"}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1",
            port=int(os.environ.get("PORT", "8101")))
```

`aftersales_server.py` 同构,两个工具 `query_warranty(order_id)`、`query_return_status(order_id)`,稳定种子、内部码字段、`PORT` 默认 8102。

- [ ] **Step 2: Makefile + dev.sh**

Makefile 追加(`.PHONY` 加 `mcp-up mcp-down`;**行首必须是 Tab**,下面代码块显示为空格,实装时替换):

```makefile
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
```

`scripts/dev.sh` 在 `uv run uvicorn ...` 前加:

```bash
make mcp-up   # ch08: 物流/售后 MCP Server(独立进程)
```

- [ ] **Step 3: 写集成测试 `tests/tools/test_mcp_servers.py`**

```python
import os
import socket
import subprocess
import sys
import time

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient


def _wait_port(port: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"port {port} not ready")


@pytest.fixture(scope="module")
def mcp_procs():
    env = {**os.environ, "PYTHONUTF8": "1"}
    p1 = subprocess.Popen([sys.executable, "mcp_servers/logistics_server.py"],
                          env={**env, "PORT": "18101"})
    p2 = subprocess.Popen([sys.executable, "mcp_servers/aftersales_server.py"],
                          env={**env, "PORT": "18102"})
    _wait_port(18101); _wait_port(18102)
    yield
    for p in (p1, p2):
        p.terminate(); p.wait(timeout=5)


def _client() -> MultiServerMCPClient:
    return MultiServerMCPClient({
        "logistics": {"transport": "streamable_http", "url": "http://127.0.0.1:18101/mcp"},
        "aftersales": {"transport": "streamable_http", "url": "http://127.0.0.1:18102/mcp"},
    }, handle_tool_errors=False)


async def test_list_tools_three_essentials(mcp_procs):
    tools = await _client().get_tools()
    by = {t.name: t for t in tools}
    assert set(by) == {"query_logistics", "query_warranty", "query_return_status"}
    for t in by.values():
        assert t.description
        schema = t.args_schema if isinstance(t.args_schema, dict) else t.args_schema.model_json_schema()
        assert schema.get("properties")


async def test_invoke_logistics_stable_mock(mcp_procs):
    tools = {t.name: t for t in await _client().get_tools(server_name="logistics")}
    r1 = await tools["query_logistics"].ainvoke({"tracking_no": "SF123"})
    r2 = await tools["query_logistics"].ainvoke({"tracking_no": "SF123"})
    assert r1 == r2 and "status_code" in str(r1)
```

- [ ] **Step 4: 跑集成测试 + 起停**

Run: `uv run pytest tests/tools/test_mcp_servers.py -v` → 2 passed。
Run: `make mcp-up && sleep 1 && make mcp-down` → 正常,`log/` 下两份日志。

- [ ] **Step 5: Commit + dev-notes**

```bash
git add mcp_servers Makefile scripts/dev.sh tests/tools/test_mcp_servers.py
git commit -m "feat(ch08): logistics and aftersales MCP servers over streamable HTTP"
```

追记:实际 SDK 版本、端口、起停方式。

---

### Task 5: MCP Client 接入 + 清单合并 + graph 接线(现问现拿)

**Files:**
- Create: `app/tools/mcp_client.py`
- Modify: `app/tools/registry.py`(加 `get_all_specs()`;删过渡兼容段)
- Delete: `app/tools/infra.py`、`tests/test_tools_infra.py`
- Modify: `app/graph/nodes.py`(`agent_llm` bind 全量、`agent_tools` 走 engine;create_ticket 旧拦截本任务先保留)
- Modify: `app/config.py`(新增 `mcp_tool_schema_tokens`)
- Modify: `app/core/context_budget.py`(`fixed` 加入 MCP 预留)
- Modify: `tests/test_context_budget.py`、`tests/graph/test_agent_nodes.py`、`scripts/bare_agent_loop.py`
- Create: `tests/tools/test_mcp_client.py`

**Interfaces:**
- Produces:
  - `async registry.get_all_specs() -> list[ToolSpec]`(内置 + MCP,重名内置优先)
  - `mcp_client.fetch_mcp_specs() -> list[ToolSpec]`
  - `mcp_client.FORMATTERS: dict[str, Callable]`
- 消费:Task 2/3/4。

- [ ] **Step 1: 写失败测试 `tests/tools/test_mcp_client.py`**

```python
from app.tools import mcp_client, registry
from app.tools.registry import ToolSpec


class _FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = f"{name} 描述"
        self.args_schema = {"type": "object", "properties": {"x": {"type": "string"}},
                            "required": ["x"]}


async def test_fetch_mcp_specs_marks_source_and_permission(monkeypatch):
    async def fake_get_tools(*, server_name=None):
        return ([_FakeTool("query_logistics")] if server_name == "logistics"
                else [_FakeTool("query_warranty")])
    monkeypatch.setattr(mcp_client, "_get_tools_of", fake_get_tools)
    specs = await mcp_client.fetch_mcp_specs()
    by = {s.name: s for s in specs}
    assert by["query_logistics"].source == "mcp"
    assert by["query_logistics"].mcp_server == "logistics"
    assert all(s.permission == "read" for s in specs)
    assert by["query_logistics"].format_result is mcp_client.FORMATTERS["query_logistics"]


async def test_one_server_down_degrades_gracefully(monkeypatch, caplog):
    async def fake_get_tools(*, server_name=None):
        if server_name == "logistics":
            raise ConnectionError("拒绝连接")
        return [_FakeTool("query_warranty")]
    monkeypatch.setattr(mcp_client, "_get_tools_of", fake_get_tools)
    specs = await mcp_client.fetch_mcp_specs()
    assert {s.name for s in specs} == {"query_warranty"}
    assert any("不可达" in r.message for r in caplog.records)


async def test_get_all_specs_merges_builtin_wins(monkeypatch):
    async def fake_fetch():
        return [
            ToolSpec(name="query_order", description="MCP 冒名",
                     json_schema={"type": "object", "properties": {}},
                     tool=_FakeTool("query_order"), permission="read",
                     source="mcp", mcp_server="x"),
            ToolSpec(name="query_logistics", description="物流",
                     json_schema={"type": "object", "properties": {}},
                     tool=_FakeTool("query_logistics"), permission="read",
                     source="mcp", mcp_server="logistics"),
        ]
    monkeypatch.setattr(mcp_client, "fetch_mcp_specs", fake_fetch)
    specs = await registry.get_all_specs()
    by = {s.name: s for s in specs}
    assert by["query_order"].source == "builtin"
    assert by["query_logistics"].source == "mcp"
    assert {"query_faq", "create_ticket", "submit_refund", "query_product"} <= set(by)


def test_logistics_formatter_translates_codes():
    out = mcp_client.FORMATTERS["query_logistics"]({
        "tracking_no": "SF1", "status_code": "IN_TRANSIT", "current_city": "深圳",
        "trace": ["深圳分拨中心 已发出"], "carrier_code": "SF-EXP-01"})
    assert out == {"tracking_no": "SF1", "status": "运输中",
                   "current_city": "深圳", "trace": ["深圳分拨中心 已发出"]}
```

- [ ] **Step 2: 跑测试确认失败** → FAIL(`mcp_client` 不存在)。

- [ ] **Step 3: 实现 `app/tools/mcp_client.py`**

按 spec §4.3:连接配置带 `timeout=settings.mcp_tool_timeout`;`_get_tools_of` 薄壳;`fetch_mcp_specs` 用 `asyncio.gather` 并发、每台 `asyncio.wait_for(..., settings.mcp_discovery_timeout)`、失败 warning 跳过;`FORMATTERS` 三个翻译函数(剔除内部编码)。

示例:

```python
def _connections() -> dict:
    return {
        "logistics": {"transport": "streamable_http", "url": settings.mcp_logistics_url,
                      "timeout": settings.mcp_tool_timeout},
        "aftersales": {"transport": "streamable_http", "url": settings.mcp_aftersales_url,
                       "timeout": settings.mcp_tool_timeout},
    }


async def _fetch_one(server: str) -> list[ToolSpec]:
    tools = await asyncio.wait_for(_get_tools_of(server_name=server),
                                   timeout=settings.mcp_discovery_timeout)
    return [registry.spec_from_langchain_tool(
        t, source="mcp", mcp_server=server,
        format_result=FORMATTERS.get(t.name)) for t in tools]


async def fetch_mcp_specs() -> list[ToolSpec]:
    results = await asyncio.gather(
        *(_fetch_one(s) for s in _connections()), return_exceptions=True)
    specs: list[ToolSpec] = []
    for server, result in zip(_connections(), results):
        if isinstance(result, BaseException):
            logger.warning("MCP Server「%s」不可达,本轮跳过其工具:%s",
                           server, type(result).__name__)
            continue
        specs.extend(result)
    return specs
```

`registry.get_all_specs()` 按 spec §4.1 合并;删除过渡兼容段。

- [ ] **Step 4: nodes.py 接线(不动 create_ticket 语义)**

`agent_llm`:

```python
    specs = await registry.get_all_specs()
    model = get_chat_model(streaming=True).bind_tools(_tools_for(state, specs))
```

`_tools_for(state, specs)` 返回 `[s.tool for s in specs]`(退款子流程外过滤 `submit_refund`)。

`agent_tools`:

```python
    specs = {s.name: s for s in await registry.get_all_specs()}
    ...
    run = await engine.execute_tool_call(tc, state.get("conversation_id", 0), specs)
```

`create_ticket`/`submit_refund` 的旧拦截分支本任务保持原样(Task 6 改 create_ticket)。

删除 `app/tools/infra.py` 与 `tests/test_tools_infra.py`;`tests/graph/test_agent_nodes.py` 的 `from app.tools.infra import ToolRun` 改 `from app.tools.engine import ToolRun`,monkeypatch 点位从 `nodes.execute_tool_call` 改到 `nodes.engine.execute_tool_call`,假实现的签名改为 `(tc, cid, specs, *, confirmed=False, deny_note=None)`;`tests/graph/test_ch06_nodes.py::test_tools_for_hides_submit_refund_outside_refund_flow` 改为 `nodes._tools_for({"route": ...}, registry.builtin_specs())`;`scripts/bare_agent_loop.py` 改用 `await registry.get_all_specs()`。

- [ ] **Step 5: 重测 MCP schema 预留**

先 `make mcp-up`,再:

```bash
uv run python - <<'EOF'
import asyncio, json
from langchain_core.utils.function_calling import convert_to_openai_tool
from app.config import Settings
from app.core import context_budget as cb
from app.tools import registry
async def main():
    s = Settings(_env_file=None, chat_model="m", chat_base_url="u", chat_api_key="k", embed_api_key="k")
    specs = [x for x in await registry.get_all_specs() if x.source == "mcp"]
    print("mcp_tool_schema_tokens =", cb.estimate_text_tokens(
        json.dumps([convert_to_openai_tool(x.tool) for x in specs], ensure_ascii=False), s))
asyncio.run(main())
EOF
make mcp-down
```

把打印值写入 `settings.mcp_tool_schema_tokens`(新增字段,默认等于实测值),`compute_budget().fixed` 加入该项;重跑 Task 2 Step 6 的打印脚本,更新 `tests/test_context_budget.py` 三组期望;`test_budget_constants_match_prompt_and_tools` 的 import 改为 `registry.builtin_specs()`(只比对内置,并在断言旁注释 MCP 预留单独计量)。

- [ ] **Step 6: 跑测试 + 真链路冒烟**

Run: `uv run pytest tests/tools tests/graph -v` → 全 PASS;`make test` 无回归。

```bash
make mcp-up && sleep 1
uv run python - <<'EOF'
import asyncio
from app.tools import engine, registry
async def main():
    specs = {s.name: s for s in await registry.get_all_specs()}
    print("清单:", sorted(specs))
    run = await engine.execute_tool_call(
        {"name": "query_logistics", "args": {"tracking_no": "SF999"}, "id": "smoke"}, 0, specs)
    print("结果:", run.tool_message.content, run.status)
    assert run.ok and "status" in run.tool_message.content
asyncio.run(main())
EOF
make mcp-down
```

Expected: 清单含 5 个内置 + 3 个 MCP;结果已翻译成中文状态(审计会落一条到 dev 库)。

- [ ] **Step 7: Commit + dev-notes**

```bash
git add -A app mcp_servers tests scripts Makefile
git commit -m "feat(ch08): MCP client integration with live tool discovery"
```

追记:MCP schema 预留实测值、冒烟输出。

---

### Task 6: 建工单确认流(interrupt 预览 → resume 放行/权限拒绝)

**Files:**
- Modify: `app/graph/nodes.py`(`agent_tools` 重写 create_ticket 分支)
- Modify: `app/core/prompts.py`(AGENT_SYSTEM)
- Modify: `app/graph/runtime.py`(interrupt payload 泛化)、`app/api/sse.py`、`app/schemas/actions.py`、`app/schemas/agent.py`、`app/api/actions.py`
- Modify: `app/config.py`(`system_prompt_tokens` 重测)
- Modify: `scripts/eval_ch06.py`、`scripts/smoke_ch07.py`(interrupt 消费形状)
- Create: `tests/graph/test_ch08_confirm_ticket.py`
- Modify: `tests/test_actions_api.py`、`tests/test_context_budget.py`

**Interfaces:**
- Produces: interrupt payload `{"type": "confirm_ticket", "preview": {"ticket_type": str, "description": str}}`;resume 值 `{"confirmed": bool}`;SSE 帧 `{"event":"interrupt","kind":"confirm_ticket","preview":{...},"conversation_id":N}`;`ResumeRequest{conversation_id, order_id?, confirmed?}`。

- [ ] **Step 1: 写失败测试 `tests/graph/test_ch08_confirm_ticket.py`**

```python
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import nodes
from app.graph.build import build_graph
from app.tools import engine, registry


class FakeModel:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, msgs, config=None):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "create_ticket", "id": "tc-1",
                "args": {"description": "猫砂盆漏电", "ticket_type": "售后"}}])
        return AIMessage(content="已为您创建工单")


@pytest.fixture()
def wired(monkeypatch):
    async def fake_coref(q, h=""):
        return q

    async def fake_intent(q, h=""):
        return {"intent": "订单", "confidence": 0.95}

    monkeypatch.setattr(nodes.coref_mod, "resolve", fake_coref)
    monkeypatch.setattr(nodes.intent_mod, "classify", fake_intent)
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kw: FakeModel())
    async def only_builtin():
        return registry.builtin_specs()
    monkeypatch.setattr(nodes.registry, "get_all_specs", only_builtin)

    created, audits = [], []
    async def fake_create(cid, desc, ttype):
        created.append((cid, desc, ttype))
        return "T20261114001"
    async def fake_audit(**kw):
        audits.append(kw)
    monkeypatch.setattr("app.db.repository.create_ticket", fake_create)
    monkeypatch.setattr(engine.repository, "insert_tool_audit", fake_audit)
    async def no_msg(*a, **kw):
        return 1
    monkeypatch.setattr(nodes.repository, "append_message", no_msg)
    return created, audits


def _input(text):
    return {
        "messages": [HumanMessage(text)], "user_id": "u1", "conversation_id": 1,
        "history_messages": [], "summary": "", "summary_upto_msg_id": 0,
        "layer1_from_msg_id": 0, "intent": "", "route": "", "evidence": "",
        "citations": [], "evidence_strong": False, "answer": "", "steps": 0,
        "tokens_used": 0, "suggested_actions": [], "trace": {},
        "resolved_query": "", "intent_confidence": 0.0,
        "order_id": "", "order_data": {},
    }


async def test_confirm_true_creates_ticket(wired):
    created, audits = wired
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    st = await g.ainvoke(_input("帮我建个工单,猫砂盆漏电"), cfg)
    intr = st["__interrupt__"][0].value
    assert intr["type"] == "confirm_ticket"
    assert intr["preview"] == {"ticket_type": "售后", "description": "猫砂盆漏电"}
    assert created == []

    st2 = await g.ainvoke(Command(resume={"confirmed": True}), cfg)
    assert created == [(1, "猫砂盆漏电", "售后")]
    assert any(a["status"] == "成功" for a in audits)
    assert "T20261114001" in str(st2["messages"])


async def test_confirm_false_denied_and_audited(wired):
    created, audits = wired
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t2"}}
    await g.ainvoke(_input("帮我建个工单,猫砂盆漏电"), cfg)
    await g.ainvoke(Command(resume={"confirmed": False}), cfg)
    assert created == []
    assert any(a["status"] == "权限拒绝" for a in audits)


async def test_resume_value_not_dict_treated_as_cancel(wired):
    created, audits = wired
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t3"}}
    await g.ainvoke(_input("帮我建个工单,猫砂盆漏电"), cfg)
    await g.ainvoke(Command(resume="oops"), cfg)
    assert created == []
    assert any(a["status"] == "权限拒绝" for a in audits)


async def test_missing_description_asks_instead_of_interrupt(wired, monkeypatch):
    created, audits = wired

    class NoDescModel(FakeModel):
        async def ainvoke(self, msgs, config=None):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{
                    "name": "create_ticket", "id": "tc-1",
                    "args": {"ticket_type": "咨询"}}])
            return AIMessage(content="请问您遇到的具体问题是什么呢?")

    monkeypatch.setattr(nodes, "get_chat_model", lambda **kw: NoDescModel())
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t4"}}
    st = await g.ainvoke(_input("帮我建个工单"), cfg)
    assert "__interrupt__" not in st
    assert created == []
    assert any(a["status"] == "校验拦下" for a in audits)
```

- [ ] **Step 2: 跑测试确认失败** → FAIL(仍是旧 `suggested_actions` 拦截)。

- [ ] **Step 3: 重写 `agent_tools`**

按 spec §4.6。核心:

```python
    ticket_calls = [tc for tc in last.tool_calls if tc["name"] == "create_ticket"]
    tspec = specs.get("create_ticket")
    decision = None
    if ticket_calls and tspec is not None and engine.validate_args(
            tspec, dict(ticket_calls[0].get("args") or {})) is None:
        args = ticket_calls[0]["args"]
        decision = interrupt({"type": "confirm_ticket",
                              "preview": {"ticket_type": args.get("ticket_type", "咨询"),
                                          "description": args.get("description", "")}})
        if not isinstance(decision, dict):
            decision = {"confirmed": False}
```

循环里:

```python
        if tc["name"] == "submit_refund":
            ...  # ch06 拦截保持原样
        elif tc["name"] == "create_ticket":
            if decision is not None and tc is ticket_calls[0]:
                if decision.get("confirmed"):
                    run = await engine.execute_tool_call(tc, cid, specs, confirmed=True)
                else:
                    run = await engine.execute_tool_call(
                        tc, cid, specs, confirmed=False,
                        deny_note="用户在工单预览卡片上点了取消,本次不建单。请勿再发起,除非用户再次明确要求。")
            else:
                run = await engine.execute_tool_call(
                    tc, cid, specs, confirmed=False,
                    deny_note="一次只处理一个建工单请求,本次调用未执行。")
            tool_msgs.append(run.tool_message)
        else:
            run = await engine.execute_tool_call(tc, cid, specs)
            tool_msgs.append(run.tool_message)
```

本任务**不**输出 `ticket_followup`(State 字段 Task 7 才加,提前写未知键会被 LangGraph 拒绝);追问后的粘滞在 Task 7 补。

- [ ] **Step 4: prompt 改造**

`AGENT_SYSTEM` 的工具使用原则第三条替换为:

```
- 只有用户明确要求建工单/要求人工跟进时,才发起 create_ticket;发起前先核对问题描述等必填信息,
  缺什么就先向用户追问,严禁编造或用占位文本充数。系统会把工单预览交给用户确认,用户取消时不要擅自重试。
- 工具清单可能动态变化(如物流、在保、退货进度等由外部服务提供),按各工具的用途描述选用;
  工具返回错误说明时,按说明修正参数或如实告知用户,不要编造结果。
```

- [ ] **Step 5: runtime / SSE / schema / API 泛化**

- `runtime._interrupt_payload` 取整个 dict;`_interrupt_event` 透传 `kind` 与除 `type` 外的字段;`_turn_result` 与 `run_turn` 的 `interrupt` 返回 payload dict。
- `app/api/sse.py::event_frames` 的 interrupt 分支改为:

```python
            yield sse({"event": "interrupt",
                       **{k: v for k, v in ev.items() if k != "type"}})
```

- `AgentResponse.interrupt: dict | None`.
- `ResumeRequest`:`order_id: str | None = Field(default=None, min_length=1)`、`confirmed: bool | None = None`。
- `api/actions.py`:`if req.order_id is None and req.confirmed is None: raise HTTPException(400, "order_id 与 confirmed 至少传一个")`;`resume_value = req.order_id if req.order_id is not None else {"confirmed": bool(req.confirmed)}`;传给 `stream_resume`。
- `scripts/eval_ch06.py`、`scripts/smoke_ch07.py`:`orders = (b.get("interrupt") or {}).get("orders") or []`。

- [ ] **Step 6: 重测 `system_prompt_tokens`**

跑 Task 2 Step 6 的打印脚本;把 `AGENT_SYSTEM` 实测值写入 `settings.system_prompt_tokens`,更新 `tests/test_context_budget.py` 三组期望。

- [ ] **Step 7: 跑测试**

Run: `uv run pytest tests/graph/test_ch08_confirm_ticket.py tests/test_actions_api.py tests/test_context_budget.py -v` → 全 PASS。
Run: `uv run pytest -q` → 全绿。已知需对位的旧断言:`tests/test_agent_api.py::test_agent_endpoint_surfaces_interrupt` 的假 `interrupt` 改成 dict、断言随新形状;`tests/graph/test_runtime.py` 的 `run_turn` 待决中断返回从 orders 列表变成 payload dict,`stream_turn` 的事件形状保持 `kind/orders` 不变。

- [ ] **Step 8: Commit + dev-notes**

```bash
git add -A app scripts tests
git commit -m "feat(ch08): create_ticket confirmation flow via interrupt and resume"
```

---

### Task 7: 意图第九类「人工」+ 追问后一轮粘滞

**Files:**
- Modify: `app/core/intent.py`、`app/core/prompts.py`(意图 prompt)、`app/graph/routing.py`、`app/graph/state.py`、`app/graph/nodes.py`
- Modify: `scripts/eval_intent.py`
- Create: `tests/graph/test_ch08_ticket_followup.py`
- Modify: `tests/graph/test_routing.py`、`tests/test_intent.py`

**Interfaces:**
- Produces: `Intent` 含 `"人工"`;`INTENT_TO_ROUTE["人工"]="business"`;`ConversationState.ticket_followup: bool`;`agent_tools` 在校验失败且未弹卡时置 True;`classify_intent` 见到该标记直接返回 `intent="人工", route="business", ticket_followup=False`,不调分类模型。

- [ ] **Step 1: 写失败测试**

`tests/graph/test_ch08_ticket_followup.py`:

```python
import pytest
from langchain_core.messages import HumanMessage

from app.graph import nodes


async def test_sticky_flag_routes_without_classifier(monkeypatch):
    called = []

    async def fake_classify(q, h=""):
        called.append((q, h))
        return {"intent": "售后", "confidence": 0.92}

    monkeypatch.setattr(nodes.intent_mod, "classify", fake_classify)
    out = await nodes.classify_intent({
        "messages": [HumanMessage("猫砂盆漏电")],
        "resolved_query": "猫砂盆漏电", "ticket_followup": True})
    assert out["intent"] == "人工" and out["route"] == "business"
    assert out["ticket_followup"] is False
    assert called == []


async def test_sticky_flag_cleared_after_one_turn(monkeypatch):
    async def fake_classify(q, h=""):
        return {"intent": "售后", "confidence": 0.92}

    monkeypatch.setattr(nodes.intent_mod, "classify", fake_classify)
    out = await nodes.classify_intent({
        "messages": [HumanMessage("那换个问题")],
        "resolved_query": "那换个问题", "ticket_followup": False})
    assert out["route"] == "refund_flow"      # 恢复正常分流
```

`tests/graph/test_routing.py` 补一条 `assert route_by_intent({"intent": "人工"}) == "business"`。

- [ ] **Step 2: 跑测试确认失败** → FAIL(`人工` 不在 Literal / 无 `ticket_followup` 分支)。

- [ ] **Step 3: 实现意图与路由**

- `app/core/intent.py`:`Intent = Literal[..., "人工"]`;`_Intent.intent` 描述改「九类意图之一」;docstring 改九类。
- `app/core/prompts.py` 意图 prompt:
  - 标题「判断标准(九选一)」;
  - 新增第 9 条:「人工:明确要求建工单、转人工客服、让人工跟进(哪怕同时带着不满)。」
  - few-shot 补:「帮我建个工单」→人工;「我要转人工跟进」→人工;「你们这什么破服务,我要投诉」→投诉(只要说法、没要求建单);「人工客服电话多少」→商品咨询(问信息,不是要求转接)。
  - 输出要求改「九类中文标签之一」。
- `app/graph/routing.py`:`"人工": "business"`。
- `app/graph/state.py`:加 `ticket_followup: bool`。
- `nodes.classify_intent` 开头:

```python
    if state.get("ticket_followup"):
        return {"intent": "人工", "intent_confidence": 1.0, "route": "business",
                "ticket_followup": False,
                "trace": trace_of(state, intent="人工", intent_confidence=1.0,
                                  route="business", ticket_followup=True)}
```

- `agent_tools` 在 `ticket_calls and decision is None` 时返回 `"ticket_followup": True`(参数不全、模型要追问的那一轮)。
- `_graph_input` 的显式重置列表**不要**加 `ticket_followup`:它的值要跨轮留在 checkpoint 里,由 `classify_intent` 消费后清零。

- [ ] **Step 4: 图级两轮测试(验收 4 的追问路径)**

在 `tests/graph/test_ch08_ticket_followup.py` 增加:

```python
async def test_followup_turn_reaches_agent_and_interrupts(monkeypatch):
    # 第一轮:create_ticket 缺 description → 校验拦下 → 追问;第二轮只说「猫砂盆漏电」
    # 意图分类器会把这句话判成售后,但 ticket_followup 粘滞必须把第二轮送进 business,
    # 模型补齐描述后弹 confirm_ticket。
```

模型 stub 两轮:第 1 轮发缺 description 的 create_ticket,第 2 轮回追问文本;第三轮(用户补充后)发完整 create_ticket。断言第二轮 classify 未被调用、第三轮出现 `confirm_ticket` 中断。测试复用 Task 6 的 `wired` fixture 风格。

- [ ] **Step 5: 跑测试 + 评估集**

Run: `uv run pytest tests/graph/test_ch08_ticket_followup.py tests/graph/test_routing.py tests/test_intent.py -v` → PASS。
Run(需上游):`PYTHONUTF8=1 uv run python -m scripts.eval_intent` → 原 18 条零回归 + 新增 2 条判对;结果记 dev-notes(模型抖动如实记录)。

- [ ] **Step 6: Commit + dev-notes**

```bash
git add app scripts tests
git commit -m "feat(ch08): human-handoff intent class with one-turn ticket followup stickiness"
```

---

### Task 8: 前端工单预览卡(Vibe Coding,不套 TDD)

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- Consumes: SSE `{"event":"interrupt","kind":"confirm_ticket","preview":{...},"conversation_id":N}`;`POST /api/actions/resume {conversation_id, confirmed}`。

- [ ] **Step 1: 实现**

- `readSse` 返回的 interrupt 在 `sendMessage`/`resumeWithOrder` 处按 `kind` 分发:`select_order` → 现有 `renderOrders`;`confirm_ticket` → `renderTicketConfirm(bubble, payload.preview, payload.conversation_id)`。
- `renderTicketConfirm`:复用 `.order-cards` 容器;卡内标题「工单预览」、两行「工单类型」「问题描述」、两个按钮「确认提交」「取消」;点击后两按钮禁用,气泡内回显「已确认提交」/「已取消」,新开 bot 气泡 `streamWithResume({conversation_id, confirmed})`,续流渲染与现有 `resumeWithOrder` 同构(抽一个 `resumeWith(payload)` 共用,避免两份)。
- 续流失败恢复按钮可点,并给 notice;结束后 `loadConversations()`。
- 刷新丢预览卡(与订单卡一致)。

- [ ] **Step 2: 手测**

起全套服务(`make dev` + `make mcp-up`),浏览器走:缺描述追问 → 补描述 → 预览卡 → 确认(回复带工单号)/取消(气泡显示已取消)。按用户反馈迭代样式。

- [ ] **Step 3: Commit + dev-notes**

```bash
git add app/static/index.html
git commit -m "feat(ch08): ticket preview card with confirm and cancel actions"
```

---

### Task 9: 验收配套(promotions 演示工具 + eval_ch08 + Makefile)

**Files:**
- Create: `scripts/demo_ch08_promotions.py.txt`(验收 1 演示物料,不预先放进 builtin/)
- Create: `scripts/eval_ch08.py`
- Modify: `Makefile`(`eval-ch08`)
- Modify: `scripts/eval_intent.py`(已由 Task 7 改,若未改在此补)

**Interfaces:**
- Produces: `make eval-ch08`;`cp scripts/demo_ch08_promotions.py.txt app/tools/builtin/promotions.py` 演示注册。

- [ ] **Step 1: promotions 演示文件**

```python
"""ch08 验收 1 演示:新内置工具 = 把本文件复制为 app/tools/builtin/promotions.py,
重启服务即被扫描注册,核心代码零改动,Agent 对话里即可用。"""
import random

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.tools import registry


class PromotionInput(BaseModel):
    keyword: str = Field(description="想查的活动关键词或商品品类,例如 猫粮")


@tool(args_schema=PromotionInput)
async def query_promotions(keyword: str) -> dict:
    """查询当前进行中的优惠活动/促销信息。用于用户问「有什么活动/优惠/折扣」时。"""
    rng = random.Random(f"promo:{keyword}")
    return {"keyword": keyword,
            "promotions": [f"{keyword}品类满{rng.randint(99, 299)}减{rng.randint(10, 50)}",
                           f"新客专享 {rng.choice(['9折', '8.5折', '95折'])}"],
            "valid_until": "2026-11-30"}


registry.register(registry.spec_from_langchain_tool(query_promotions, source="builtin"))
```

- [ ] **Step 2: `scripts/eval_ch08.py`**

直连 runtime 层,监听真库;样例至少覆盖:

1. 「帮我建个工单」→ 无 confirm_ticket 中断、tickets 不落、答复含追问。
2. 同一会话补「猫砂盆漏电」→ 出现 confirm_ticket 中断且 preview 字段齐(验证粘滞)。
3. `resume {"confirmed": true}` → tickets +1、答复含工单号。
4. 新会话走到预览 → `resume {"confirmed": false}` → 无新工单、审计最新 create_ticket 为「权限拒绝」。
5. 「订单 1001 的物流到哪了」→ 走 MCP `query_logistics`,答复含翻译后的中文状态。
6. 「最近有什么优惠活动」→ 若 `promotions.py` 已放好则走 `query_promotions`;未放好则跳过并打印 SKIP。

输出 PASS/FAIL 汇总,FAIL 时退出码非 0。需要 `make dev` + `make mcp-up`。

- [ ] **Step 3: Makefile + 跑**

```makefile
eval-ch08:  ## ch08 验收样例端到端(需 make dev + make mcp-up)
	PYTHONUTF8=1 uv run python scripts/eval_ch08.py
```

Run: `make mcp-up && make eval-ch08`(dev 服务在跑,`.env` 有上游 key)。
Expected: 全 PASS;失败即修再跑。

- [ ] **Step 4: Commit + dev-notes**

```bash
git add scripts Makefile
git commit -m "feat(ch08): acceptance eval and promotions demo material"
```

---

### Task 10: 端到端验收 + code review + finish

**Files:** 无新增(验收记录进 dev-notes)。

- [ ] **Step 1: 全套服务**

`make dev`(已含 `mcp-up`);确认 dev 库已应用 `sql/ch08-ddl.sql`。

- [ ] **Step 2: 六条验收逐条过**

1. `cp scripts/demo_ch08_promotions.py.txt app/tools/builtin/promotions.py` → 重启 uvicorn → 问「最近有什么优惠活动」→ Agent 调 `query_promotions` 回答;演示后按用户意见保留或删除。
2. 问「订单 1001 的物流到哪了」→ Agent 先 `query_order` 拿 tracking_no,再经 MCP `query_logistics` 回答。
3. 向 `mcp_servers/logistics_server.py` 临时加 `query_delivery_eta` → `make mcp-down && make mcp-up`(只重启 MCP,客服系统不动)→ 问「订单 1001 预计几号送到」→ 用上新工具;演示后回滚临时工具。
4. 「帮我建个工单」→ 追问 → 补「猫砂盆漏电」→ 预览卡 → 确认提交 → `SELECT` tickets 最新一条,回复带工单号。
5. 同流程到预览 → 取消 → tickets 无新增;`SELECT status FROM tool_audit_logs WHERE tool_name='create_ticket' ORDER BY id DESC LIMIT 1` = 权限拒绝。
6. `MOCK_DELAY_SECONDS=12 PORT=8101 uv run python mcp_servers/logistics_server.py`(替换物流 Server)→ 问物流 → 审计该次调用 `status=超时`、`retry_count=2`、`duration_ms` 落在 30s 段位;恢复正常 Server。`DEMO_TICKET_DELAY_SECONDS=8` 重启主服务 → 确认建单 → 超时 → 审计 `create_ticket status=超时, retry_count=0`。恢复环境。

- [ ] **Step 3: 两轴 code review**

按用户 `/code-review` 习惯:Standards 轴(CLAUDE.md + Fowler smell baseline)与 Spec 轴(ch08 spec)并行独立子代理;结果按必要/建议/可选三档呈现;前端(Vibe Coding)按用户此前要求可不纳入或只做人工检查。

- [ ] **Step 4: 修复与复跑**

必要与建议项逐一修;每个修复先写失败测试再改代码;全量 `make test` 复跑绿。

- [ ] **Step 5: finish**

- 全量 `make test` + `make eval-ch08` 结果入 dev-notes。
- dev-notes 补 finish 段:交付清单(演示命令、测试结果、spec/plan/dev-notes 路径)。
- 用 superpowers:finishing-a-development-branch 给出合并/PR/保留三选项;用户未明确说合并前不合并。

---

## Self-Review 记录

- **Spec 覆盖**:§4.1/4.2 → Task 2;§4.3 → Task 5;§4.4 → Task 3;§4.5 → Task 2/3;§4.6 → Task 6;§4.7 → Task 7;§4.8 → Task 8;§4.9 → Task 1;§4.10 → Task 2/5/6 的常量步骤;§6 失败模式 → Task 3/5 测试;六条验收 → Task 9/10。
- **类型一致性**:`ToolSpec`/`ToolRun`/`execute_tool_call`/`validate_args`/`get_all_specs`/`ticket_followup` 在各任务间签名一致;interrupt payload 与前端消费字段一致。
- **Review Focus**:5 条各有落点测试,已在各任务 Step 中写明。
- **有意留白**:`eval_ch08.py` 全文、前端具体 DOM/CSS、`test_followup_turn_reaches_agent_and_interrupts` 的完整 stub 在对应任务里按现有 fixture 对位实现,不属于 TBD 型占位。
