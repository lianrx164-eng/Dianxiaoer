# Ch07 上下文管理(三层 + 异步分段摘要)实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> 本次按用户既有约定走 **inline 执行**(executing-plans)。

**Goal:** 把「全量历史直递模型」升级为三层上下文:层 1 最近原文、层 2 规则截短、层 3 分段追加摘要;两个 DB 锚点划边界,层 2 超预算后台异步压成新段,预算从模型窗口倒推;全程可观测,前端加会话侧栏。

**Architecture:** State.messages(add_messages + checkpoint)存全量;每轮入口读 DB 已落 user/assistant 行,构造带 `db-{id}` 锚点的精简历史放 State;`_agent_messages` 按 `summary_upto`/`layer1_from` 切层并拼装(system 恒定,梗概+证据+退款数据挂当前用户之后);层 1 超预算在 runtime 降级并写回 `layer1_from_msg_id`;轮后 `asyncio.create_task` 后台把层 2 压成 `conversation_summaries` 新段,追 `summary_upto_msg_id`。

**Tech Stack:** LangGraph 1.2(State/add_messages/AsyncSqliteSaver)、LangChain 1.5(`trim_messages` 兜底)、SQLAlchemy 2 async + MySQL(conversations/messages/conversation_summaries)、FastAPI、原生 JS 前端。

**Spec:** `docs/superpowers/specs/2026-11-14-ch07-context-management-design.md`

## Global Constraints

- 技术选型定死:LangGraph State + LangChain trim_messages + MySQL 两表(加摘要表);走不通停下来问,不换方案。
- 分支:`ch07-context-management`(从 `main` 切),不直接改 main。
- **不自动 commit**:每个 Task 结束跑完测试、输出变更摘要;提交等用户明确要求(用户全局规则)。
- 基线:`uv run pytest -q` 234 passed(2026-11-14)。
- 三层口径:`id ≤ summary_upto` 层 3;`summary_upto < id ≤ layer1_from` 层 2;`id > layer1_from` 层 1。层 1 从 user 消息开始,边界只落在轮之间。
- 当前用户消息不计入层 1/层 2 预算,由 `max_user_input_tokens` 预留;永远保留、不降级。
- 拼装顺序固定:`[System(AGENT_SYSTEM), 层2, 层1, 当前用户, Human(turn_ctx), 本轮 ReAct 后续]`;system 不含任何可变内容。
- 预算常量(默认):`chars_per_token=2.0`、system 389、tools 1081、包装 30、output 2000、user 2000、evidence 400×top_k、summary 750、safety 500、peak=steps×tool_result;`layer1=history*7//10-1`、`layer2=history-layer1-1`。
- 演示配置必须算出 5650/3954/1695;只调窗口(默认步数/rerank)必须算出 50/34/15 并报警;默认 64000 算出 30000/20999/9000。
- 摘要红线:只提炼事实与诉求;没出现的一个字不许编;寒暄不留;几十到一两百字;旧梗概只作背景、不参与合并;一段一行只追加。
- 摘要失败只 log 不重试;两字段只在成功后同一事务更新。
- `log_node` 仍是 user/assistant 唯一写点;不改变「失败轮不落 MySQL」语义。
- 日志落 `log/app.log`: `model_ctx`(每次模型调用)、`history_ctx`(每轮一次)、`layer1 降级 X→Y`、`summary trigger/skip/done 第N段/failed`。
- 每完成一个 Task 在 `dev-notes/ch07.md` 追记一段(关键原话/产出/拒绝纠正/翻车返工),不许收尾补记。
- API 测试打桩 repository;repository 测试用真库 fixture;前端 Vibe Coding,浏览器手动验。
- Context7 已核:`add_messages` 同 id 更新覆盖、`AsyncSession.begin/commit`、`trim_messages` 自定义 token_counter;实现中遇到新 API 先查再写。

## Review Focus

1. **暂停在订单选择器时用户又发新消息(D9)**:不得启动新一轮、不得落库、不得降级边界;重吐中断。resume 后上下文里 pending 用户消息恰好一次。
2. **单轮/单条超预算**(超长用户消息或超长客服答复):保留最新一轮,当前用户永不丢,不返回空窗/孤儿 tool;只 log 警告。
3. **旧会话/无锚点消息列表**:DB 行运行时一律带 `db-{id}` 锚点,旧会话按新机制正常管理;只有无锚点消息列表(直接调用/旧 checkpoint)全部视为层 1,超预算走 `trim_messages` 兜底,不写 DB 边界、不触发摘要。
4. **摘要失败/空输出**:`conversation_summaries`、`conversations.summary`、`summary_upto_msg_id` 一个都不写;防抖表清理;下一轮仍满足条件时重触发。
5. **摘要任务与下一轮并发/服务重启**:边界最多晚一轮生效;段 `seq` 唯一、旧段内容不变、新段不重复;重启丢任务无妨。

---

### Task 1: 配置 + 预算模块(context_budget)

**Files:**
- Create: `app/core/context_budget.py`
- Modify: `app/config.py`(新增 ch07 配置块)
- Test: `tests/test_context_budget.py`(新建)、`tests/test_config.py`(追加 env 覆盖)
- 不碰 DB/图。

**Interfaces:**
- Produces:
  - `estimate_text_tokens(text: str | None, s: Settings = settings) -> int`
  - `estimate_message_tokens(m: BaseMessage, s: Settings = settings) -> int`
  - `estimate_messages_tokens(msgs: list[BaseMessage], s: Settings = settings) -> int`
  - `@dataclass(frozen=True) class ContextBudget: window, fixed, react_peak, history, layer1, layer2`
  - `compute_budget(s: Settings = settings) -> ContextBudget`
  - `self_check(s: Settings = settings) -> ContextBudget`(不足时 `logger.error("上下文预算不足: ...")`)
- Settings 新增:
  ```python
  # ch07 会话上下文预算(全部可从 env 覆盖;演示见 spec §7.2)
  model_context_window: int = 64000
  max_output_tokens: int = 2000
  max_user_input_tokens: int = 2000
  tool_result_max_tokens: int = 1200
  chars_per_token: float = 2.0
  system_prompt_tokens: int = 389       # AGENT_SYSTEM 777 字 @ 2.0
  tool_schema_tokens: int = 1081        # 6 个工具 schema 2161 字 @ 2.0
  system_overhead_tokens: int = 30
  evidence_chunk_tokens: int = 400
  summary_budget_tokens: int = 750
  safety_margin_tokens: int = 500
  desired_history_turns: int = 30
  turn_token_estimate: int = 1000
  layer2_assistant_chars: int = 40
  layer2_tool_keep_chars: int = 120
  ```

- [ ] **Step 1: 写失败测试** `tests/test_context_budget.py`
  ```python
  import logging

  from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

  from app.config import Settings
  from app.core import context_budget as cb


  def _settings(**kw):
      base = dict(chat_model="m", chat_base_url="u", chat_api_key="k",
                  embed_api_key="k", _env_file=None)
      base.update(kw)
      return Settings(**base)


  def test_estimate_text_tokens_uses_chars_per_token():
      s = _settings()
      assert cb.estimate_text_tokens("", s) == 0
      assert cb.estimate_text_tokens("a" * 777, s) == 389
      assert cb.estimate_text_tokens("a" * 2161, s) == 1081


  def test_demo_budget_matches_acceptance_numbers():
      b = cb.compute_budget(_settings(
          model_context_window=18000, max_output_tokens=2000,
          max_user_input_tokens=2000, max_agent_steps=3,
          tool_result_max_tokens=1200, rerank_top_k=5))
      assert (b.window, b.fixed, b.react_peak, b.history, b.layer1, b.layer2) == (
          18000, 8750, 3600, 5650, 3954, 1695)


  def test_only_window_set_yields_50_and_self_check_warns(caplog):
      s = _settings(model_context_window=18000)      # 步数/rerank 保持默认 6/10
      b = cb.compute_budget(s)
      assert (b.fixed, b.react_peak, b.history, b.layer1, b.layer2) == (
          10750, 7200, 50, 34, 15)
      with caplog.at_level(logging.ERROR, logger="app.core.context_budget"):
          cb.self_check(s)
      assert "上下文预算不足" in caplog.text


  def test_default_budget():
      b = cb.compute_budget(_settings())
      assert (b.window, b.fixed, b.react_peak, b.history, b.layer1, b.layer2) == (
          64000, 10750, 7200, 30000, 20999, 9000)


  def test_budget_constants_match_prompt_and_tools():
      """提示词或工具 schema 变了,这条先红;重测后同步 system/tools 常量。"""
      import json

      from langchain_core.utils.function_calling import convert_to_openai_tool

      from app.core.prompts import AGENT_SYSTEM
      from app.tools.registry import get_all_tools

      s = _settings()
      tools_json = json.dumps([convert_to_openai_tool(t) for t in get_all_tools()],
                              ensure_ascii=False)
      assert cb.estimate_text_tokens(AGENT_SYSTEM, s) == s.system_prompt_tokens
      assert cb.estimate_text_tokens(tools_json, s) == s.tool_schema_tokens


  def test_estimate_messages_tokens_counts_tool_calls():
      s = _settings()
      msgs = [
          HumanMessage("你好" * 10),
          AIMessage("", tool_calls=[{"name": "query_order",
                                     "args": {"order_id": "1001"}, "id": "c1"}]),
          ToolMessage("x" * 100, tool_call_id="c1", name="query_order"),
      ]
      assert cb.estimate_messages_tokens(msgs, s) > cb.estimate_text_tokens("你好" * 10, s)
  ```

- [ ] **Step 2: 跑测试确认失败**
  Run: `uv run pytest tests/test_context_budget.py -v`
  Expected: FAIL(`ModuleNotFoundError: app.core.context_budget`)。

- [ ] **Step 3: 写 `app/config.py` 配置块** + **实现 `app/core/context_budget.py`**
  - `compute_budget` 公式(逐字对照 spec §7.1):
    ```python
    fixed = (s.system_prompt_tokens + s.tool_schema_tokens + s.system_overhead_tokens
             + s.max_output_tokens + s.max_user_input_tokens
             + s.rerank_top_k * s.evidence_chunk_tokens
             + s.summary_budget_tokens + s.safety_margin_tokens)
    react_peak = s.max_agent_steps * s.tool_result_max_tokens
    leftover = s.model_context_window - fixed - react_peak
    history = max(0, min(s.desired_history_turns * s.turn_token_estimate, leftover))
    layer1 = max(0, history * 7 // 10 - 1)
    layer2 = max(0, history - layer1 - 1)
    ```
  - `estimate_text_tokens` 用 `math.ceil(len(text) / s.chars_per_token)`;空串返回 0。
  - `estimate_message_tokens`:4 token/条包装 + 文本 + tool_calls 的 name/args JSON。
  - `self_check`:`history <= 0 or layer1 < s.turn_token_estimate` → `logger.error`;返回 budget,不抛。
  - `context_budget.py` 只依赖 `app.config` 与 `langchain_core`,不 import DB/工具(避免循环)。

- [ ] **Step 4: 跑测试确认通过 + 全量回归**
  Run: `uv run pytest tests/test_context_budget.py tests/test_config.py -v` → 全 PASS
  Run: `uv run pytest -q` → 无回归

- [ ] **Step 5: 变更摘要(不 commit)**
  输出 `git diff --stat` + 新增/修改文件清单,等用户确认。

---

### Task 2: DB 层——三列 + 摘要表 + repository 函数

**Files:**
- Create: `sql/ch07-ddl.sql`(原样采用桌面 `CH_07/ch_07.sql`)
- Modify: `app/db/models.py`
- Modify: `tests/conftest.py`
- Modify: `app/db/repository.py`
- Test: `tests/test_repository.py`(追加)、`tests/test_models.py`(追加)

**Interfaces:**
- Produces:
  - `Conversation.summary: str | None`、`summary_upto_msg_id: int | None`、`layer1_from_msg_id: int | None`
  - `ConversationSummary`(`id, conversation_id, seq, from_msg_id, upto_msg_id, content, created_at`)
  - `repository.list_dialog_messages(cid) -> list[Message]`
  - `repository.update_layer1_from(cid, msg_id) -> None`
  - `repository.append_summary(cid, from_msg_id, upto_msg_id, content, max_projection_tokens=None) -> int`(返回 seq)
  - `repository.list_summaries(cid) -> list[ConversationSummary]`
  - `repository.list_conversations(user_id, limit=50) -> list[dict]`(id/status/preview/has_summary/created_at;按 id desc)

- [ ] **Step 1: 落 DDL 与 conftest**
  - 把桌面 `CH_07/ch_07.sql` **原样**复制为 `sql/ch07-ddl.sql`(两段 ALTER + `conversation_summaries`;不改字段/索引)。
  - `tests/conftest.py`:
    ```python
    _DDL_FILES = [..., pathlib.Path(...) / "sql" / "ch07-ddl.sql"]
    # _ddl_statements 的筛选:
    if "CREATE TABLE" in s.upper() or "ALTER TABLE" in s.upper()
    # _TABLES 首部:
    "conversation_summaries", "faith_cases", ...
    ```

- [ ] **Step 2: 写失败测试** `tests/test_repository.py` 追加
  ```python
  async def test_conversation_summary_columns_and_layer1(db_session_factory, db_clean):
      cid = await repo.create_conversation("u1")
      conv = await repo.get_conversation(cid)
      assert conv.summary is None
      assert conv.summary_upto_msg_id is None
      assert conv.layer1_from_msg_id is None
      await repo.update_layer1_from(cid, 5)
      assert (await repo.get_conversation(cid)).layer1_from_msg_id == 5


  async def test_list_dialog_messages_filters_tool(db_session_factory, db_clean):
      cid = await repo.create_conversation("u1")
      await repo.append_message(cid, "user", content="订单1001到哪了")
      await repo.append_message(cid, "tool", content='{"s":1}', tool_call_id="c1")
      await repo.append_message(cid, "assistant", content="在路上")
      rows = await repo.list_dialog_messages(cid)
      assert [r.role for r in rows] == ["user", "assistant"]      # tool 行过滤,id 升序


  async def test_append_summary_appends_and_projects(db_session_factory, db_clean):
      cid = await repo.create_conversation("u1")
      ids = []
      for i in range(4):
          ids.append(await repo.append_message(cid, "user", content=f"q{i} 订单100{i}"))
          ids.append(await repo.append_message(cid, "assistant", content=f"a{i}"))
      assert await repo.append_summary(cid, ids[0], ids[5], "第一段:订单1001/手机13800138000") == 1
      conv = await repo.get_conversation(cid)
      assert conv.summary_upto_msg_id == ids[5] and "订单1001" in conv.summary
      assert await repo.append_summary(cid, ids[6], ids[7], "第二段:补发立柱") == 2
      conv = await repo.get_conversation(cid)
      assert conv.summary_upto_msg_id == ids[7]
      assert "第一段" in conv.summary and "第二段" in conv.summary
      rows = await repo.list_summaries(cid)
      assert [r.seq for r in rows] == [1, 2]
      assert rows[0].content == "第一段:订单1001/手机13800138000"   # 旧段不被重写


  async def test_append_summary_projection_keeps_newest(db_session_factory, db_clean):
      cid = await repo.create_conversation("u1")
      ids = [await repo.append_message(cid, "user", content=f"q{i}") for i in range(4)]
      await repo.append_summary(cid, ids[0], ids[1], "旧段" * 200, max_projection_tokens=10)
      await repo.append_summary(cid, ids[2], ids[3], "新段", max_projection_tokens=10)
      conv = await repo.get_conversation(cid)
      assert "旧段" not in conv.summary and "新段" in conv.summary
      assert len(await repo.list_summaries(cid)) == 2              # 表里两段都在


  async def test_list_conversations_preview_has_summary_order(db_session_factory, db_clean):
      c1 = await repo.create_conversation("u1")
      await repo.append_message(c1, "user", content="第一问")
      await repo.append_message(c1, "assistant", content="答")
      c2 = await repo.create_conversation("u1")
      m1 = await repo.append_message(c2, "user", content="第二问")
      m2 = await repo.append_message(c2, "assistant", content="答")
      c3 = await repo.create_conversation("u2")
      await repo.append_message(c3, "user", content="别人的会话")
      await repo.append_summary(c2, m1, m2, "摘要")
      items = await repo.list_conversations("u1")
      assert [i["id"] for i in items] == [c2, c1]                  # id desc 新在前
      assert items[0]["preview"] == "第二问"
      assert items[0]["has_summary"] is True and items[1]["has_summary"] is False
      assert all(i["id"] != c3 for i in items)                     # 用户隔离
  ```

- [ ] **Step 3: 跑测试确认失败**
  Run: `uv run pytest tests/test_repository.py -v -k "summary or dialog_messages or list_conversations"`
  Expected: FAIL(`AttributeError: summary` / `no attribute 'list_dialog_messages'`)。

- [ ] **Step 4: 实现**
  - `models.py`:
    ```python
    # Conversation(created_at 之前)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_upto_msg_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    layer1_from_msg_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    class ConversationSummary(Base):
        __tablename__ = "conversation_summaries"
        id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
        conversation_id: Mapped[int] = mapped_column(BigInteger)
        seq: Mapped[int] = mapped_column(Integer)
        from_msg_id: Mapped[int] = mapped_column(BigInteger)
        upto_msg_id: Mapped[int] = mapped_column(BigInteger)
        content: Mapped[str] = mapped_column(Text)
        created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    ```
  - `repository.py`(追加;`from app.core.context_budget import estimate_text_tokens`、`from app.config import settings`):
    ```python
    async def list_dialog_messages(conversation_id: int) -> list[Message]:
        async with db.async_session() as s:
            result = await s.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id,
                       Message.role.in_(("user", "assistant")))
                .order_by(Message.id))
            return list(result.scalars())

    async def update_layer1_from(conversation_id: int, msg_id: int) -> None:
        async with db.async_session() as s:
            conv = await s.get(Conversation, conversation_id)
            if conv is not None:
                conv.layer1_from_msg_id = msg_id
                await s.commit()

    async def list_summaries(conversation_id: int) -> list[ConversationSummary]:
        async with db.async_session() as s:
            result = await s.execute(
                select(ConversationSummary)
                .where(ConversationSummary.conversation_id == conversation_id)
                .order_by(ConversationSummary.seq))
            return list(result.scalars())

    async def append_summary(conversation_id, from_msg_id, upto_msg_id, content,
                             max_projection_tokens: int | None = None) -> int:
        """插一段 + 重建投影 + 追 summary_upto,单事务;旧段永不改。"""
        budget = settings.summary_budget_tokens if max_projection_tokens is None else max_projection_tokens
        async with db.async_session() as s:
            conv = await s.get(Conversation, conversation_id)
            if conv is None:
                return 0
            max_seq = (await s.execute(
                select(func.max(ConversationSummary.seq))
                .where(ConversationSummary.conversation_id == conversation_id)
            )).scalar_one()
            seq = int(max_seq or 0) + 1
            s.add(ConversationSummary(
                conversation_id=conversation_id, seq=seq,
                from_msg_id=from_msg_id, upto_msg_id=upto_msg_id, content=content))
            await s.flush()
            rows = list((await s.execute(
                select(ConversationSummary)
                .where(ConversationSummary.conversation_id == conversation_id)
                .order_by(ConversationSummary.seq.desc()))).scalars())
            kept, total = [], 0
            for row in rows:                      # 从新到旧攒,超预算停
                t = estimate_text_tokens(row.content)
                if kept and total + t > budget:
                    break
                kept.append(row.content)
                total += t
            kept.reverse()                        # 投影按时间正序
            conv.summary = "\n".join(kept)
            conv.summary_upto_msg_id = upto_msg_id
            await s.commit()
            return seq

    async def list_conversations(user_id: str, limit: int = 50) -> list[dict]:
        async with db.async_session() as s:
            convs = list((await s.execute(
                select(Conversation).where(Conversation.user_id == user_id)
                .order_by(Conversation.id.desc()).limit(limit))).scalars())
            out = []
            for conv in convs:
                preview = (await s.execute(
                    select(Message.content)
                    .where(Message.conversation_id == conv.id, Message.role == "user")
                    .order_by(Message.id).limit(1))).scalar_one_or_none()
                out.append({
                    "id": conv.id, "status": conv.status,
                    "preview": (preview or "").strip()[:40],
                    "has_summary": bool(conv.summary_upto_msg_id),
                    "created_at": conv.created_at.isoformat() if conv.created_at else None,
                })
            return out
    ```

- [ ] **Step 5: 跑测试确认通过 + 全量回归**
  Run: `uv run pytest tests/test_repository.py tests/test_models.py -v` → 全 PASS
  Run: `uv run pytest -q` → 无回归

- [ ] **Step 6: 变更摘要(不 commit)**

---

### Task 3: memory 三层切分/降级/截短

**Files:**
- Modify: `app/core/memory.py`(`SessionStore`/`trim_history` 保留不动,追加)
- Test: `tests/test_memory.py`(追加)

**Interfaces:**
- Produces:
  - `db_id(m: BaseMessage) -> int | None`(解 `db-{id}`)
  - `split_layers(history_messages, summary_upto: int, layer1_from: int) -> tuple[list[BaseMessage], list[BaseMessage]]` → `(layer2, layer1)`
  - `degrade_layer1(history_messages, layer1_from: int, budget_tokens: int, s=settings) -> int`(新边界;不降返回原值)
  - `render_layer2(messages, assistant_chars: int, tool_keep_chars: int) -> list[BaseMessage]`
  - `trim_turns_to_budget(messages, budget_tokens: int, s=settings) -> list[BaseMessage]`(按整轮)
  - `build_layers(history_messages, summary_upto, layer1_from, layer1_budget, layer2_budget, assistant_chars, tool_keep_chars, s=settings) -> tuple[list[BaseMessage], list[BaseMessage]]` → `(layer2_trimmed, layer1_trimmed)`
  - `row_to_message(row) -> HumanMessage | AIMessage`(DB 行 → 带 `db-{id}` 锚点的消息;runtime/summarizer 共用)
  - `summary_line(summary: str | None) -> str`

- [ ] **Step 1: 写失败测试** `tests/test_memory.py` 追加
  ```python
  from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

  from app.core import context_budget
  from app.core.memory import (
      build_layers, db_id, degrade_layer1, render_layer2, split_layers,
      summary_line, trim_turns_to_budget)


  def _history(n_turns: int, start_id: int = 1) -> list:
      msgs, did = [], start_id
      for i in range(n_turns):
          msgs.append(HumanMessage(f"问题{i}", id=f"db-{did}")); did += 1
          msgs.append(AIMessage(f"回答{i}", id=f"db-{did}")); did += 1
      return msgs


  def _turn_tokens(turn, s=None):
      return context_budget.estimate_messages_tokens(turn, s or context_budget.settings)


  def test_split_layers_cuts_at_boundaries():
      h = _history(10)                       # ids 1..20
      layer2, layer1 = split_layers(h, summary_upto=6, layer1_from=14)
      assert [m.id for m in layer2] == [f"db-{i}" for i in range(7, 15)]
      assert [m.id for m in layer1] == [f"db-{i}" for i in range(15, 21)]


  def test_split_layers_legacy_without_ids_keeps_all_in_layer1():
      h = [HumanMessage("旧问题"), AIMessage("答")] * 3
      layer2, layer1 = split_layers(h, 0, 0)
      assert layer2 == [] and layer1 == h


  def test_degrade_layer1_moves_whole_turns():
      h = _history(10)
      turns = [h[i:i + 2] for i in range(0, len(h), 2)]
      budget = sum(_turn_tokens(t) for t in turns[-4:]) + 1
      new_from = degrade_layer1(h, 0, budget)
      assert new_from == 12                   # 丢第 1-6 轮,边界=第 6 轮 assistant id
      layer2, layer1 = split_layers(h, 0, new_from)
      assert layer1[0].id == "db-13"          # 层 1 从第 7 轮 user 开始


  def test_degrade_layer1_keeps_newest_oversized_turn():
      h = [HumanMessage("超长" + "喵" * 5000, id="db-1"), AIMessage("答", id="db-2")]
      assert degrade_layer1(h, 0, 10) == 0     # 只有一轮:宁可超预算也保留


  def test_degrade_layer1_moves_forward_from_boundary():
      h = _history(4)                         # ids 1..8
      assert degrade_layer1(h, 4, 1) == 6     # 丢第 3 轮,新边界=id6;只向前


  def test_render_layer2_truncates_assistant_and_collapses_tools():
      msgs = [
          HumanMessage("完整问题"),
          AIMessage("答" * 100),
          AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}]),
          ToolMessage("x" * 500, tool_call_id="c1", name="query_order"),
      ]
      out = render_layer2(msgs, assistant_chars=40, tool_keep_chars=120)
      assert out[0].content == "完整问题"                  # 用户原话不动
      assert out[1].content == "答" * 40 + "…"
      assert out[2].content.startswith("[调用工具:query_order")
      assert len(out) == 3                                  # tool 行被折叠,不单独出现


  def test_trim_turns_to_budget_keeps_whole_turns():
      h = _history(3)
      budget = _turn_tokens(h[-2:]) + 1
      out = trim_turns_to_budget(h, budget)
      assert [m.content for m in out] == ["问题2", "回答2"]   # 整轮保留,不拆
      assert isinstance(out[0], HumanMessage)


  def test_summary_line_flattens_multiline():
      assert summary_line(None) == "" and summary_line("") == ""
      assert summary_line("第一段\n第二段") == "(早前对话摘要:第一段 第二段)"


  def test_build_layers_trims_layer2_to_budget():
      h = _history(6)
      layer2, layer1 = build_layers(
          h, 0, 8, layer1_budget=10000, layer2_budget=1,
          assistant_chars=40, tool_keep_chars=120)
      assert len(layer2) == 2 and isinstance(layer2[0], HumanMessage)
  ```

- [ ] **Step 2: 跑测试确认失败**
  Run: `uv run pytest tests/test_memory.py -v`
  Expected: FAIL(`ImportError: cannot import name 'build_layers'`)。

- [ ] **Step 3: 实现 `app/core/memory.py`**
  - `db_id`: 前缀 `db-`,`int()` 失败返回 None。
  - `row_to_message`:`role == "user"` → `HumanMessage`,否则 `AIMessage`;`content=row.content or ""`,`id=f"db-{row.id}"`。
  - `_group_turns(messages)`:
    ```python
    def _group_turns(messages):
        turns, cur = [], []
        for m in messages:
            if isinstance(m, HumanMessage) and cur:
                turns.append(cur); cur = []
            cur.append(m)
        if cur: turns.append(cur)
        return turns
    ```
    - 层 1/层 2 的 DB 行天然 user 开头;这个函数兜底旧消息。
  - `split_layers`:
    ```python
    def split_layers(history_messages, summary_upto, layer1_from):
        ids = [db_id(m) for m in history_messages]
        if not any(i is not None for i in ids):
            return [], list(history_messages)          # 无锚点:全层 1
        layer2, layer1 = [], []
        for m, did in zip(history_messages, ids):
            if did is None:                             # 无锚点跟随前一段
                (layer1 if layer1 else layer2).append(m)
            elif did <= summary_upto:
                continue
            elif did <= layer1_from:
                layer2.append(m)
            else:
                layer1.append(m)
        return layer2, layer1
    ```
  - `degrade_layer1`:
    ```python
    def degrade_layer1(history_messages, layer1_from, budget_tokens, s=settings):
        ids = [db_id(m) for m in history_messages]
        if not any(i is not None for i in ids):
            return layer1_from                           # 无锚点不挪边界
        start = next((i for i, did in enumerate(ids)
                      if did is not None and did > layer1_from), len(history_messages))
        turns = _group_turns(history_messages[start:])
        if not turns:
            return layer1_from
        if estimate_messages_tokens([m for t in turns for m in t], s) <= budget_tokens:
            return layer1_from
        keep, acc = [], 0
        for turn in reversed(turns):
            t = estimate_messages_tokens(turn, s)
            if keep and acc + t > budget_tokens:
                break
            keep.append(turn); acc += t                   # 最新一轮永远保留
        if len(keep) == len(turns):
            return layer1_from
        dropped = turns[:len(turns) - len(keep)]
        new_id = db_id(dropped[-1][-1])
        if new_id is None:
            return layer1_from
        return max(layer1_from, new_id)                   # 只向前
    ```
  - `render_layer2`:
    ```python
    def render_layer2(messages, assistant_chars, tool_keep_chars):
        out, skip_tool = [], False
        for m in messages:
            if isinstance(m, HumanMessage):
                out.append(HumanMessage(m.content if isinstance(m.content, str) else ""))
                skip_tool = False
            elif isinstance(m, AIMessage):
                if getattr(m, "tool_calls", None):
                    names = ",".join(tc.get("name", "") for tc in m.tool_calls)
                    out.append(AIMessage(f"[调用工具:{names},结果已省略]"))
                    skip_tool = True
                else:
                    text = m.content if isinstance(m.content, str) else ""
                    out.append(AIMessage(
                        text[:assistant_chars] + ("…" if len(text) > assistant_chars else "")))
                    skip_tool = False
            elif isinstance(m, ToolMessage):
                if not skip_tool:                          # 孤立工具行给一行标识
                    size = len(m.content) if isinstance(m.content, str) else 0
                    out.append(AIMessage(f"[工具结果 {m.name or ''} 已省略,{size}字]"))
        return out
    ```
  - `trim_turns_to_budget`:按 `_group_turns`,从新到旧攒;最新一轮永远保留;返回扁平列表。
  - `build_layers`:split → render → trim layer2 → trim layer1(无锚点列表走 `trim_history`)。
  - `summary_line`:flatten。

- [ ] **Step 4: 跑测试确认通过 + 全量回归**
  Run: `uv run pytest tests/test_memory.py -v` → 全 PASS
  Run: `uv run pytest -q` → 无回归

- [ ] **Step 5: 变更摘要(不 commit)**

---

### Task 4: 摘要 prompt + summarizer(触发/防抖/任务体)

**Files:**
- Modify: `app/core/prompts.py`(尾部追加 `SUMMARY_SYSTEM` / `SUMMARY_PROMPT`)
- Create: `app/core/summarizer.py`
- Test: `tests/test_summarizer.py`(新建)

**Interfaces:**
- Produces:
  - `summarizer.summarize_dialog(old_summary: str, dialog: str) -> str`
  - `summarizer.run_summary(cid: int) -> None`
  - `summarizer.maybe_schedule_summary(cid: int) -> None`
  - `summarizer._running: dict[int, asyncio.Task]`
- Consumes: Task 1 `compute_budget`、Task 2 repository 函数、Task 3 `render_layer2`/`estimate_messages_tokens`。

- [ ] **Step 1: 写失败测试** `tests/test_summarizer.py`
  ```python
  import asyncio
  from types import SimpleNamespace

  import pytest
  from langchain_core.messages import AIMessage, HumanMessage

  from app.core import context_budget, summarizer
  from app.db import repository as repo


  async def _seed(cid: int, turns: int) -> list[int]:
      ids = []
      for i in range(turns):
          ids.append(await repo.append_message(cid, "user", content=f"问题{i} 订单1001"))
          ids.append(await repo.append_message(cid, "assistant", content=f"回答{i}"))
      return ids


  async def test_run_summary_appends_segment_and_skips_second_time(
          db_session_factory, db_clean, monkeypatch, caplog):
      cid = await repo.create_conversation("u1")
      ids = await _seed(cid, 12)                         # ids[0..23]
      await repo.update_layer1_from(cid, ids[15])        # 第 8 轮 assistant
      calls = []

      async def fake(old, dialog):
          calls.append((old, dialog))
          return "第一段:订单1001/手机13800138000"
      monkeypatch.setattr(summarizer, "summarize_dialog", fake)

      with caplog.at_level("INFO", logger="app.core.summarizer"):
          await summarizer.run_summary(cid)
      conv = await repo.get_conversation(cid)
      assert conv.summary_upto_msg_id == ids[15]
      assert conv.layer1_from_msg_id == ids[15]
      assert "第一段" in conv.summary
      rows = await repo.list_summaries(cid)
      assert len(rows) == 1 and rows[0].from_msg_id == ids[0] and rows[0].upto_msg_id == ids[15]
      assert "summary done" in caplog.text

      await summarizer.run_summary(cid)                  # 边界没前进 → 跳过
      assert len(await repo.list_summaries(cid)) == 1
      assert len(calls) == 1

      await repo.update_layer1_from(cid, ids[19])        # 层 1 又前移
      async def fake2(old, dialog):
          calls.append((old, dialog))
          return "第二段:补发立柱"
      monkeypatch.setattr(summarizer, "summarize_dialog", fake2)
      await summarizer.run_summary(cid)
      rows = await repo.list_summaries(cid)
      assert [r.seq for r in rows] == [1, 2]
      assert rows[0].content == "第一段:订单1001/手机13800138000"   # 旧段不变
      conv = await repo.get_conversation(cid)
      assert "第一段" in conv.summary and "第二段" in conv.summary
      assert conv.summary_upto_msg_id == ids[19]


  async def test_run_summary_failure_or_empty_writes_nothing(
          db_session_factory, db_clean, monkeypatch):
      cid = await repo.create_conversation("u1")
      ids = await _seed(cid, 4)
      await repo.update_layer1_from(cid, ids[3])

      async def boom(old, dialog):
          raise RuntimeError("upstream")
      monkeypatch.setattr(summarizer, "summarize_dialog", boom)
      await summarizer.run_summary(cid)
      conv = await repo.get_conversation(cid)
      assert conv.summary is None and conv.summary_upto_msg_id is None
      assert await repo.list_summaries(cid) == []

      async def empty(old, dialog):
          return "   "
      monkeypatch.setattr(summarizer, "summarize_dialog", empty)
      await summarizer.run_summary(cid)
      assert (await repo.get_conversation(cid)).summary is None
      assert await repo.list_summaries(cid) == []


  async def test_maybe_schedule_skips_empty_layer2(db_session_factory, db_clean, monkeypatch):
      cid = await repo.create_conversation("u1")
      await _seed(cid, 2)
      called = []
      async def fake(c):
          called.append(c)
      monkeypatch.setattr(summarizer, "run_summary", fake)
      await summarizer.maybe_schedule_summary(cid)       # layer1_from=0 → 层 2 空
      assert called == [] and summarizer._running.get(cid) is None


  async def test_maybe_schedule_skips_below_budget(db_session_factory, db_clean, monkeypatch):
      cid = await repo.create_conversation("u1")
      ids = await _seed(cid, 4)
      await repo.update_layer1_from(cid, ids[5])
      monkeypatch.setattr(summarizer, "compute_budget",
                          lambda: context_budget.ContextBudget(
                              64000, 0, 0, 30000, 20000, 10**9))
      called = []
      async def fake(c):
          called.append(c)
      monkeypatch.setattr(summarizer, "run_summary", fake)
      await summarizer.maybe_schedule_summary(cid)
      assert called == []


  async def test_maybe_schedule_fires_and_debounces(db_session_factory, db_clean, monkeypatch):
      cid = await repo.create_conversation("u1")
      ids = await _seed(cid, 4)
      await repo.update_layer1_from(cid, ids[5])
      monkeypatch.setattr(summarizer, "compute_budget",
                          lambda: context_budget.ContextBudget(
                              64000, 0, 0, 30000, 20000, 1))   # 层 2 预算 1,必触发
      gate = asyncio.Event()
      ran = []
      async def slow(c):
          ran.append(c)
          await gate.wait()
      monkeypatch.setattr(summarizer, "run_summary", slow)

      await summarizer.maybe_schedule_summary(cid)
      await summarizer.maybe_schedule_summary(cid)       # 在跑 → 防抖
      await asyncio.sleep(0)
      assert ran == [cid]
      gate.set()
      await summarizer._running[cid]
      assert cid not in summarizer._running
  ```

- [ ] **Step 2: 跑测试确认失败**
  Run: `uv run pytest tests/test_summarizer.py -v`
  Expected: FAIL(`ModuleNotFoundError: app.core.summarizer`)。

- [ ] **Step 3: 实现**
  - `prompts.py` 追加(注意 ChatPromptTemplate 的双花括号):
    ```python
    SUMMARY_SYSTEM = """## 角色
    你是客服对话摘要器。把一段较早的客服对话压成简洁摘要,供后续轮次当背景用。

    ## 规则
    - 只提炼事实与诉求:问过哪款商品、报过的订单号/手机号、用户的明确诉求、还没解决的问题
    - 对话里没出现的内容一个字不许编;寒暄闲聊不留
    - 旧梗概只作背景参考,不要复述、不要合并进输出;只总结「本次新增对话」
    - 长度几十到一两百字
    - 严格输出 JSON:{{"summary": "string"}}"""

    SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
        ("system", SUMMARY_SYSTEM),
        ("human", "旧梗概(仅背景,不要复述):\n{old_summary}\n\n本次新增对话:\n{dialog}"),
    ])
    ```
  - `app/core/summarizer.py`:
    ```python
    """ch07 后台分段摘要:层 2 超预算 → 轮后 create_task 压成新段;不阻塞当轮回复。
    失败只 log 不重试;新段插入 + 投影重建 + summary_upto 更新在一个事务里。"""
    import asyncio
    import logging
    import time

    from pydantic import BaseModel, Field

    from app.config import settings
    from app.core import memory
    from app.core.context_budget import compute_budget, estimate_messages_tokens
    from app.core.llm import get_chat_model, structured_model, thinking_body
    from app.core.prompts import SUMMARY_PROMPT
    from app.db import repository

    logger = logging.getLogger(__name__)
    _running: dict[int, asyncio.Task] = {}

    class _Summary(BaseModel):
        summary: str = Field(description="本次新增对话的分段摘要,只含事实与诉求")

    async def summarize_dialog(old_summary: str, dialog: str) -> str:
        model = get_chat_model(extra_body=thinking_body())
        r: _Summary = await structured_model(_Summary, model=model).ainvoke(
            SUMMARY_PROMPT.format_messages(
                old_summary=old_summary or "(无)", dialog=dialog))
        return (r.summary or "").strip()

    async def run_summary(conversation_id: int) -> None:
        t0 = time.monotonic()
        try:
            conv = await repository.get_conversation(conversation_id)
            if conv is None:
                return
            boundary = conv.layer1_from_msg_id or 0
            old_upto = conv.summary_upto_msg_id or 0
            if boundary <= old_upto:
                logger.info("summary skip conv=%s reason=no_boundary upto=%s layer1_from=%s",
                            conversation_id, old_upto, boundary)
                return
            rows = await repository.list_dialog_messages(conversation_id)
            seg = [r for r in rows if old_upto < r.id <= boundary]
            if not seg:
                logger.info("summary skip conv=%s reason=empty_batch", conversation_id)
                return
            dialog = "\n".join(
                f"{'用户' if r.role == 'user' else '客服'}:{r.content}"
                for r in seg if r.content)
            logger.info("summary start conv=%s msgs=%s upto %s->%s",
                        conversation_id, len(seg), old_upto, boundary)
            summary = await summarize_dialog(conv.summary or "", dialog)
            if not summary:
                raise ValueError("摘要为空,放弃更新")
            seq = await repository.append_summary(
                conversation_id, seg[0].id, boundary, summary)
            logger.info("summary done conv=%s 第%s段 from=%s upto=%s len=%s cost=%.0fms",
                        conversation_id, seq, seg[0].id, boundary, len(summary),
                        (time.monotonic() - t0) * 1000)
        except Exception:  # noqa: BLE001 - 后台任务失败不拖垮主链路
            logger.exception("summary failed conv=%s", conversation_id)

    async def maybe_schedule_summary(conversation_id: int) -> None:
        if conversation_id in _running:
            logger.info("summary skip conv=%s reason=running", conversation_id)
            return
        conv = await repository.get_conversation(conversation_id)
        if conv is None:
            return
        upto = conv.summary_upto_msg_id or 0
        boundary = conv.layer1_from_msg_id or 0
        if boundary <= upto:
            logger.info("summary skip conv=%s reason=layer2_empty upto=%s layer1_from=%s",
                        conversation_id, upto, boundary)
            return
        rows = await repository.list_dialog_messages(conversation_id)
        layer2 = [memory.row_to_message(r) for r in rows if upto < r.id <= boundary]
        rendered = memory.render_layer2(
            layer2, settings.layer2_assistant_chars, settings.layer2_tool_keep_chars)
        tokens = estimate_messages_tokens(rendered)
        budget = compute_budget()
        if tokens <= budget.layer2:
            logger.info("summary skip conv=%s reason=below_budget tokens=%s budget=%s",
                        conversation_id, tokens, budget.layer2)
            return
        logger.info("summary trigger conv=%s layer2_tokens=%s budget=%s upto=%s layer1_from=%s",
                    conversation_id, tokens, budget.layer2, upto, boundary)
        task = asyncio.create_task(run_summary(conversation_id))
        _running[conversation_id] = task

        def _done(t):
            _running.pop(conversation_id, None)
        task.add_done_callback(_done)
    ```
  - `run_summary` 里 `append_summary` 之后 log 这一段(逐字):
    ```python
    logger.info("summary done conv=%s 第%s段 from=%s upto=%s len=%s cost=%.0fms",
                conversation_id, seq, seg[0].id, boundary, len(summary),
                (time.monotonic() - t0) * 1000)
    ```
  - 说明:Task 3 的 `memory.row_to_message` 供 runtime 与本模块共用;`compute_budget`/`estimate_messages_tokens` 在模块顶层直接 import,测试才可 `monkeypatch.setattr(summarizer, "compute_budget", ...)`。

- [ ] **Step 4: 跑测试确认通过 + 全量回归**
  Run: `uv run pytest tests/test_summarizer.py -v` → 全 PASS
  Run: `uv run pytest -q` → 无回归

- [ ] **Step 5: 变更摘要(不 commit)**

---

### Task 5: 接线——State/runtime/nodes 拼装

**Files:**
- Modify: `app/graph/state.py`
- Modify: `app/graph/runtime.py`
- Modify: `app/graph/nodes.py`
- Create: `tests/test_context_assembly.py`
- Modify: `tests/graph/test_runtime.py`、`tests/graph/test_agent_nodes.py`、`tests/graph/test_ch06_nodes.py`、`tests/test_agent_orchestration.py`、`tests/test_agent_stream.py`、`tests/test_agent_citations.py`

**Interfaces:**
- State 新增:`history_messages: list[AnyMessage]`、`summary: str`、`summary_upto_msg_id: int`、`layer1_from_msg_id: int`
- runtime:
  - `_load_context(cid: int) -> dict`(`history_messages/summary/summary_upto/layer1_from`;含层 1 降级与写回)
  - `_graph_input(user_id, message, cid, prev_messages, ctx) -> dict`
  - 四个入口轮后 `await summarizer.maybe_schedule_summary(cid)`
- nodes:
  - `_history_text(state, max_messages=6)`(优先 `history_messages`,旧测试回落 `messages[:-1]`)
  - `_current_turn_messages(state)`
  - `_turn_context(state)`
  - `_agent_messages(state)`
  - `_log_model_context(state, msgs)`
  - `coref` 打 `history_ctx`;`agent_llm` 调模型前打 `model_ctx`

- [ ] **Step 1: 写失败测试** `tests/test_context_assembly.py`
  ```python
  import logging

  from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

  from app.core.prompts import AGENT_SYSTEM
  from app.graph import nodes


  def _state(n_turns=2, summary="", upto=0, layer1_from=0, current="现在的问题",
             evidence="", route="business", extra=()):
      hist, did = [], 1
      for i in range(n_turns):
          hist.append(HumanMessage(f"问题{i}", id=f"db-{did}")); did += 1
          hist.append(AIMessage(f"回答{i}", id=f"db-{did}")); did += 1
      prior = [m for m in hist if int(m.id.split("-")[1]) > upto]
      return {"messages": [*hist, HumanMessage(current), *extra],
              "history_messages": prior, "summary": summary,
              "summary_upto_msg_id": upto, "layer1_from_msg_id": layer1_from,
              "evidence": evidence, "route": route, "conversation_id": 7}


  def test_order_with_summary():
      msgs = nodes._agent_messages(_state(
          n_turns=2, summary="用户问过订单1001", evidence="[1] 7天无理由",
          route="knowledge", current="能退吗"))
      assert isinstance(msgs[0], SystemMessage) and msgs[0].content == AGENT_SYSTEM
      assert "[1]" not in msgs[0].content                  # 证据不进 system
      assert isinstance(msgs[1], HumanMessage) and msgs[1].content == "问题0"
      assert msgs[-2].content == "能退吗"                    # 当前用户
      assert "订单1001" in msgs[-1].content and "[1]" in msgs[-1].content


  def test_system_is_always_static_for_all_routes():
      for state in (
          _state(route="business"),
          _state(route="knowledge", evidence="[1] 运费: 满99包邮"),
          _state(route="refund_flow", evidence="[1] 7天无理由",
                 extra=(AIMessage("", tool_calls=[
                     {"name": "submit_refund", "args": {"order_id": "1001"}, "id": "r1"}]),)),
      ):
          assert nodes._agent_messages(state)[0].content == AGENT_SYSTEM


  def test_step1_prompt_is_strict_prefix_of_step2():
      q = HumanMessage("能退吗", id="db-tmp")
      s1 = nodes._agent_messages({"messages": [q], "history_messages": [],
                                  "route": "knowledge", "evidence": "[1] 7天"})
      s2 = nodes._agent_messages({"messages": [
          q, AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}])],
          "history_messages": [], "route": "knowledge", "evidence": "[1] 7天"})
      shape = lambda ms: [(m.type, m.content) for m in ms]      # noqa: E731
      assert shape(s2)[:len(s1)] == shape(s1)


  def test_history_text_prepends_summary_and_windows():
      text = nodes._history_text(_state(n_turns=6, summary="用户问过订单1001"))
      assert text.startswith("(早前对话摘要:用户问过订单1001")
      assert "问题0" not in text and "问题3" in text


  def test_history_text_legacy_falls_back_to_messages():
      text = nodes._history_text({"messages": [
          HumanMessage("q0"), AIMessage("a0"), HumanMessage("q1")]})
      assert "q0" in text and "a0" in text and "q1" not in text


  def test_log_model_context_contains_summary_and_window(caplog):
      state = _state(n_turns=2, summary="用户问过订单1001", evidence="[1] 7天",
                     route="knowledge", current="能退吗")
      msgs = nodes._agent_messages(state)
      with caplog.at_level(logging.INFO, logger="app.graph.nodes"):
          nodes._log_model_context(state, msgs)
      assert "model_ctx" in caplog.text
      assert "用户问过订单1001" in caplog.text
      assert "[human] '问题0" in caplog.text
      assert "tokens=" in caplog.text
  ```

- [ ] **Step 2: 跑测试确认失败**
  Run: `uv run pytest tests/test_context_assembly.py -v`
  Expected: FAIL(拼装顺序/证据位置/`_log_model_context` 不存在)。

- [ ] **Step 3: 实现 state.py**
  ```python
  history_messages: list[AnyMessage]   # 本轮精简历史(DB 已落层2+层1,带 db-id 锚点)
  summary: str                         # conversations.summary 投影
  summary_upto_msg_id: int             # 梗概覆盖到哪条
  layer1_from_msg_id: int              # 层1起点;id > 此值为原文
  ```

- [ ] **Step 4: 实现 nodes.py**
  - 移除 `from app.core.memory import trim_history`(不再用),新增 `from app.core import context_budget, memory`(evidence 已有)。
  - `_history_text`:
    ```python
    def _history_text(state, max_messages: int = 6) -> str:
        if "history_messages" in state:
            prior = state.get("history_messages") or []
        else:
            prior = state.get("messages", [])[:-1]       # 旧测试/直接调用回落
        lines = []
        for m in prior[-max_messages:]:
            if isinstance(m, HumanMessage):
                role = "用户"
            elif isinstance(m, AIMessage):
                role = "客服"
            else:
                continue
            text = m.content if isinstance(m.content, str) else ""
            if text:
                lines.append(f"{role}:{text}")
        body = "\n".join(lines)
        head = memory.summary_line(state.get("summary"))
        return f"{head}\n{body}".strip() if head else body
    ```
  - `TURN_CTX_ID = "turn-ctx"`;`_current_turn_messages`;`_turn_context`;`_agent_messages`(见下,固定顺序)。
    ```python
    def _turn_context(state) -> str:
        parts = []
        if state.get("summary"):
            parts.append("\n\n## 早前对话摘要(更早轮次已压缩,其中事实可信)\n"
                         + state["summary"])
        if state.get("evidence"):
            parts.append(_KNOWLEDGE_EVIDENCE_HINT + state["evidence"])
        if _in_refund_flow(state):
            parts.append(REFUND_JUDGE_HINT
                         + json.dumps(state.get("order_data") or {}, ensure_ascii=False))
        return "".join(parts)

    def _agent_messages(state) -> list:
        budget = context_budget.compute_budget()
        layer2, layer1 = memory.build_layers(
            state.get("history_messages", []),
            state.get("summary_upto_msg_id", 0) or 0,
            state.get("layer1_from_msg_id", 0) or 0,
            budget.layer1, budget.layer2,
            settings.layer2_assistant_chars, settings.layer2_tool_keep_chars)
        current = _current_turn_messages(state)
        ctx = _turn_context(state)
        msgs = [SystemMessage(AGENT_SYSTEM), *layer2, *layer1]
        if current:
            msgs.append(current[0])                       # 当前用户
            if ctx:
                msgs.append(HumanMessage(ctx, id=TURN_CTX_ID))
            msgs.extend(current[1:])                      # 本轮 ReAct 后续
        elif ctx:
            msgs.append(HumanMessage(ctx, id=TURN_CTX_ID))
        return msgs
    ```
  - `_log_model_context`(每次模型调用前,逐字):
    ```python
    def _log_model_context(state, msgs) -> None:
        """调模型前把实际上下文打进 log/app.log(验收可观测)。"""
        window = msgs[1:]                                 # 去掉静态 system
        desc = [f"[{m.type}] '{(m.content if isinstance(m.content, str) else '')[:40]}'"
                for m in window]
        logger.info(
            "model_ctx conv=%s step=%s summary=%r window_msgs=%d tokens=%d window=%s",
            state.get("conversation_id"), state.get("steps", 0),
            state.get("summary") or "", len(window),
            context_budget.estimate_messages_tokens(msgs), " | ".join(desc))
    ```
  - `_log_model_context` 在 `agent_llm` 里 `ainvoke` 之前调用。

- [ ] **Step 5: 实现 runtime.py**
  - 导入 `context_budget, memory, summarizer`。
  - `_load_context(cid)`:
    ```python
    async def _load_context(cid: int) -> dict:
        conv = await repository.get_conversation(cid)
        if conv is None:
            raise ConversationNotFound(cid)
        rows = await repository.list_dialog_messages(cid)
        history = [memory.row_to_message(r) for r in rows]
        summary = conv.summary or ""
        upto = conv.summary_upto_msg_id or 0
        layer1_from = conv.layer1_from_msg_id or 0
        if layer1_from < upto:
            layer1_from = upto
        budget = context_budget.compute_budget()
        new_from = memory.degrade_layer1(history, layer1_from, budget.layer1)
        if new_from != layer1_from:
            await repository.update_layer1_from(cid, new_from)
            logger.info("layer1 降级 %s->%s conv=%s layer1_budget=%s",
                        layer1_from, new_from, cid, budget.layer1)
            layer1_from = new_from
        kept = [m for m in history if (memory.db_id(m) or 0) > upto]
        return {"history_messages": kept, "summary": summary,
                "summary_upto": upto, "layer1_from": layer1_from}
    ```
  - `_graph_input` 增参 `ctx`,写入四个字段;`run_turn`/`stream_turn` 在 pending 检查之后调用 `_load_context(cid)`,再 invoke/stream;结束后 `await summarizer.maybe_schedule_summary(cid)`。
  - `resume_turn`/`stream_resume` 在 ainvoke/流结束后同样 schedule。
  - **注意**:pending 检查保持在 `_load_context` 之前,避免暂停中降级/写库(D9)。
  - `_ensure_conversation` 不变(返回 cid);`_load_context` 再读一次 Conversation(可接受;测试里打桩)。
  - `init_graph` 编译图后调用 `context_budget.self_check()`,并把 budget 五项打一条 INFO。

- [ ] **Step 6: 更新已有测试**(逐条)
  - `tests/graph/test_agent_nodes.py`:
    - 证据断言移到 `msgs[-1]`;system 断言 `msgs[0].content == AGENT_SYSTEM`。
    - `test_agent_messages_trims_long_history`:旧消息不再出现(新拼装从最后 human 起切)。
  - `tests/graph/test_ch06_nodes.py`:
    - `test_agent_messages_injects_order_and_policy_on_refund` / `test_agent_messages_knowledge_path_still_injects_evidence`:改断 `msgs[-1]`;system 恒为 `AGENT_SYSTEM`。
  - `tests/test_agent_orchestration.py`:fake_agent 里 `assert any("已检索到的知识证据" in (m.content or "") for m in nodes._agent_messages(state))`。
  - `tests/graph/test_runtime.py`:autouse fixture 打桩 `get_conversation`(返回带三字段的 `SimpleNamespace`)、`list_dialog_messages`、`maybe_schedule_summary`;新增一条断言 `_graph_input` 带 `history_messages/summary/summary_upto_msg_id/layer1_from_msg_id`,且 schedule 被调用。
    ```python
    from types import SimpleNamespace

    @pytest.fixture(autouse=True)
    def _quiet(monkeypatch):
        async def fake_get(cid):
            return SimpleNamespace(summary="", summary_upto_msg_id=0, layer1_from_msg_id=0)
        async def fake_rows(cid):
            return []
        async def fake_schedule(cid):
            return None
        monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
        monkeypatch.setattr(runtime.repository, "list_dialog_messages", fake_rows)
        monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", fake_schedule)
    ```
    - `test_run_turn_raises_when_conversation_missing` 在测试体内把自己 `get_conversation` 改成返回 None,覆盖 fixture,仍然抛 `ConversationNotFound`。
  - `tests/test_agent_stream.py`、`tests/test_agent_citations.py`:在各自 `_repo` autouse fixture 里追加同样三个打桩(`get_conversation` / `list_dialog_messages` / `maybe_schedule_summary`)。
  - `tests/test_chat_api.py` 不动(打桩 stream_turn)。

- [ ] **Step 7: 跑测试确认通过 + 全量回归**
  Run: `uv run pytest tests/test_context_assembly.py -v` → 全 PASS
  Run: `uv run pytest -q` → 无回归(重点 `test_agent_*` / `test_chat_api` / `test_actions_api`)

- [ ] **Step 8: 变更摘要(不 commit)**

---

### Task 6: 会话只读接口 + 侧栏 + app.log

**Files:**
- Create: `app/api/conversations.py`、`tests/test_conversations_api.py`
- Modify: `app/main.py`(注册路由 + FileHandler)
- Modify: `app/static/index.html`(侧栏/切换/回载)
- Modify: `tests/test_static.py`

**Interfaces:**
- Produces:
  - `GET /api/conversations?user_id=` → `{"items":[{id,status,preview,has_summary,created_at}]}`
  - `GET /api/conversations/{id}/messages` → `{"items":[{role,content,created_at}]}`;会话不存在 404。
- Consumes: Task 2 repository。

- [ ] **Step 1: 写失败测试** `tests/test_conversations_api.py`
  ```python
  from types import SimpleNamespace

  from fastapi.testclient import TestClient

  from app.api import conversations
  from app.main import app


  def test_list_conversations_passes_user_id(monkeypatch):
      async def fake_list(uid):
          assert uid == "u1"
          return [{"id": 2, "status": "进行中", "preview": "第二问",
                   "has_summary": True, "created_at": "2026-11-14T10:00:00"}]
      monkeypatch.setattr(conversations.repository, "list_conversations", fake_list)
      r = TestClient(app).get("/api/conversations?user_id=u1")
      assert r.status_code == 200
      assert r.json()["items"][0]["id"] == 2


  def test_messages_404(monkeypatch):
      async def fake_get(cid):
          return None
      monkeypatch.setattr(conversations.repository, "get_conversation", fake_get)
      r = TestClient(app).get("/api/conversations/9/messages")
      assert r.status_code == 404


  def test_messages_returns_user_assistant(monkeypatch):
      async def fake_get(cid):
          return SimpleNamespace(id=cid)
      async def fake_rows(cid):
          return [SimpleNamespace(role="user", content="q", created_at=None),
                  SimpleNamespace(role="assistant", content="a", created_at=None)]
      monkeypatch.setattr(conversations.repository, "get_conversation", fake_get)
      monkeypatch.setattr(conversations.repository, "list_dialog_messages", fake_rows)
      r = TestClient(app).get("/api/conversations/3/messages")
      assert r.status_code == 200
      assert [i["role"] for i in r.json()["items"]] == ["user", "assistant"]
  ```

- [ ] **Step 2: 跑测试确认失败**
  Run: `uv run pytest tests/test_conversations_api.py -v`
  Expected: FAIL(`ModuleNotFoundError: app.api.conversations` / 404 路由不存在)。

- [ ] **Step 3: 实现 API + 注册 + 日志文件**
  - `app/api/conversations.py`(两个 GET;404 `HTTPException`;`created_at` isoformat 可空)。
  - `app/main.py`:`include_router(conversations_router)`;`log/` 目录 + `FileHandler("log/app.log", encoding="utf-8")`,格式与 StreamHandler 一致,已存在 FileHandler 时不重复加。
  - `tests/test_static.py` 追加:
    ```python
    r = await c.get("/")
    assert 'id="sidebar"' in r.text and "/api/conversations" in r.text
    ```

- [ ] **Step 4: 前端侧栏(Vibe Coding)**
  - HTML:`main.app` 改横向;加 `<aside id="sidebar" class="sidebar"><div class="sidebar-head">会话</div><div id="convList" class="conv-list"></div></aside>`,原 topbar/messages/composer 包进 `<div class="chat">`。
  - CSS:`.app{flex-direction:row} .sidebar{width:224px;flex:none;display:flex;flex-direction:column;border-right:3px solid #1c1c1c;background:#f3ead9} .chat{flex:1;min-width:0;display:flex;flex-direction:column} .conv-item{...} .conv-item.active{background:#ffe9c2} .conv-badge{...} @media(max-width:780px){.sidebar{display:none}.app{flex-direction:column}}`。
  - JS:
    - `let convItems = [];`
    - `loadConversations()`:`fetch('/api/conversations?user_id='+encodeURIComponent(userId))`,失败 `console.warn` 静默;`renderConversations()`。
    - `renderConversations()`:清空 `convList`,每项 button(preview + 创建时间 + 可选「已摘要」),`active` 高亮;点击 `switchConversation(item.id)`。
    - `switchConversation(id)`:`busy` 时 return;GET `/api/conversations/{id}/messages`;成功后 `saveConversationId(id)`、清消息区、逐条渲染(user 纯文本 / assistant `renderMarkdown`)、空则欢迎语、滚动到底、重渲染高亮;失败 `console.warn` 静默保留当前聊天。
    - 页面加载:`loadConversations()`;有 `conversationId` 则 `switchConversation(conversationId)`,失败/无则 `addAssistant(WELCOME)`。
    - `sendMessage`/`resumeWithOrder` 结束(含中断暂停)调用 `loadConversations()`;`newChat` 清空后重渲染高亮并显示欢迎语。
  - 手动验证:两个会话来回切、历史完整、能继续聊;窄屏侧栏消失。

- [ ] **Step 5: 跑测试 + 全量回归**
  Run: `uv run pytest tests/test_conversations_api.py tests/test_static.py -v` → 全 PASS
  Run: `uv run pytest -q` → 无回归

- [ ] **Step 6: 变更摘要(不 commit)**

---

### Task 7: 摘要 prompt 评估集(eval 代 TDD)+ Makefile

**Files:**
- Create: `scripts/eval_ch07.py`
- Modify: `Makefile`(`eval-ch07` target + `.PHONY`)

**Interfaces:**
- Consumes: Task 4 `summarize_dialog`。
- Produces: `make eval-ch07` / `uv run python -m scripts.eval_ch07`(需上游可用)。

- [ ] **Step 1: 写脚本** `scripts/eval_ch07.py`
  - 三条样例:
    1. 事实保留 + 寒暄不留:`must=["1001","13800138000","猫爬架"]`,`ban=["在吗","您好,我是小喵"]`;
    2. 无编造:摘要里的 4 位以上数字必须都在原文出现(`src_digits`);
    3. 滚动背景不合并:旧梗概含订单号/手机/承重;新对话只讲立柱缺失/补发;`must=["补发","立柱"]`,`ban=["承重15公斤"]`(旧事实不重复)。
  - `check(name, summary, must, ban, src_digits)`:长度 20–250;关键字;数字子集;打印 ✅/❌ 与摘要。
  - 结束 `raise SystemExit(0 if all else 1)`。
  - 提示词不过就改 `SUMMARY_SYSTEM` 重跑,改动记 dev-notes。

- [ ] **Step 2: Makefile**
  ```makefile
  # .PHONY 追加 eval-ch07
  eval-ch07:
      PYTHONUTF8=1 uv run python -m scripts.eval_ch07
  ```

- [ ] **Step 3: 跑真模型**
  Run: `make eval-ch07`
  Expected: `3/3 通过`。

- [ ] **Step 4: 变更摘要(不 commit)**

---

### Task 8: 冒烟、DDL 应用、交付

**Files:**
- Create: `scripts/smoke_ch07.py`
- Modify: `Makefile`(`smoke-ch07` / `smoke-ch07-demo`)
- Apply: `sql/ch07-ddl.sql` 到 dev 库
- 无新业务代码。

- [ ] **Step 1: 写冒烟脚本**
  - 直接走 `runtime.init_graph()` + `runtime.run_turn`(不走 HTTP);给 `app` logger 挂捕获 handler。
  - `mode=default`:22 轮短消息(约 50–100 字/轮),断言捕获日志无 `layer1 降级`、无 `summary trigger`,`context` 不爆。
  - `mode=demo`:30 轮长消息(约 200 字/轮,含订单 1001、手机号、诉求;25 轮实测层 2 1515 < 1695,30 轮过阈值),断言出现 `layer1 降级`、`summary trigger`、`summary done`;结束后 `await asyncio.gather(*summarizer._running.values())`,查 `conversations.summary` 含 `1001`、`list_summaries` 非空;第 31 问答案含早期订单号(软断言,打印)。
  - 结束 `close_graph()`。
  - Makefile:
    ```makefile
    smoke-ch07:
        PYTHONUTF8=1 uv run python -m scripts.smoke_ch07 default
    smoke-ch07-demo:
        MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5 PYTHONUTF8=1 uv run python -m scripts.smoke_ch07 demo
    ```

- [ ] **Step 2: 应用 DDL 到 dev 库**
  Run: `uv run python scripts/seed_db.py sql/ch07-ddl.sql`
  Verify: `SHOW COLUMNS FROM conversations` 含三列;`SHOW TABLES LIKE 'conversation_summaries'` 存在。
  (已核 dev 库当前无这三列/表;只需 apply 一次。)

- [ ] **Step 3: 全量验证**
  Run: `uv run pytest -q` → 全 PASS
  Run: `make eval-ch07` → 3/3
  Run: `make smoke-ch07` → 默认 22 轮无压
  Run: `make smoke-ch07-demo` → 级联 + 摘要落库
  - 浏览器手动:侧栏多会话、切回回载、默认 20+ 轮不压、演示配置级联;`grep model_ctx / history_ctx / layer1 降级 / summary log/app.log`。

- [ ] **Step 4: dev-notes 追记 + code review**
  - `dev-notes/ch07.md` 补每个 Task 的完成段、冒烟结果、验收证据。
  - 请求 `superpowers:requesting-code-review` / `/code-review`(两轴),结论记 dev-notes;按「必要+建议」修复;可选记入 deferred。
  - 最后按 `superpowers:finishing-a-development-branch` 停在「保留分支」,等用户说合并。

- [ ] **Step 5: 变更摘要(不 commit;由用户决定提交/合并)**

---

## Self-Review 结论

- **Spec 覆盖**:三层边界(T2/T3)、预算(T1)、摘要任务(T4)、拼装(T5)、可观测(T5/T6)、侧栏(T6)、eval(T7)、冒烟/DDL(T8);spec §13 每条都有落点。
- **占位扫描**:无 TBD;每个 Step 有命令/预期或明确签名。
- **类型一致性**:`ContextBudget` 六字段;`degrade_layer1(messages, layer1_from, budget, s)`;`build_layers(...)->(layer2, layer1)`;`append_summary(...)->seq`;State 四字段;`_load_context(cid)->dict`;`maybe_schedule_summary(cid)` 四入口一致。
- **Review Focus**:五类输入分别落在 T5(暂停/超长/旧会话)、T4(失败/并发)、T8(长对话/连续 user 消息)。
- **比例**:计划包含关键测试与签名,未抄整份实现;前端只给行为与关键片段。
