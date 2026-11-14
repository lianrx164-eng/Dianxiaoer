# Ch07 会话上下文管理设计:简单裁剪 → 三层上下文 + 异步分段摘要

> 以用户桌面 `CH_07/Sepc.md`、`CH_07/Plan.md`、`CH_07/ch_07.sql` 为基础,按本仓 ch06 实际基线与用户本轮需求消息修订。桌面 Spec/Plan 对应的是「第一步:双层滑窗 + 滚动摘要」;本轮需求与 `ch_07.sql` 第二步是「三层上下文 + 分段追加摘要」。两者冲突处以**用户本轮需求消息 + `ch_07.sql` 第二步**为准,桌面文档作为第一步的背景与可复用实现细节。凡与桌面文档不同之处,在 §2 列明。

## 1. 背景与目标

ch05/ch06 的 `agent_llm` 每轮把 system + checkpoint 里的全量 `state["messages"]` 递给模型;`app/core/memory.py` 里 ch02 的 `trim_history` 虽已存在,但只按 token 砍尾,不区分远近、不留摘要。轮数一多 token 必爆。本章把第一版简单裁剪升级成三层上下文管理。只管当前会话,不跨会话、不存用户画像。

**目标(需求 7 条):**

1. **三层结构**:按离当前轮的远近切三段。层 1 最近原文一字不压;层 2 中间段按规则截短(用户原话不动、客服答复只留开头几十字、大块工具结果换一行标识);层 3 最远段异步压成分段梗概。两个 DB 锚点 `summary_upto_msg_id`(梗概覆盖到哪条)与 `layer1_from_msg_id`(层 1 起点)划边界;降级只挪 id,不搬数据。梗概单开 `conversation_summaries` 表,一段一行只追加。
2. **摘要后台异步**:层 2 渲染后的 token 用量超过其预算 → 轮结束后后台起一次摘要任务,把层 2 这一批压成新的一段追加进摘要表,`summary_upto_msg_id` 追到层 1 起点;全程不 await,不阻塞当轮回复。旧梗概只作背景给摘要模型看,不参与合并;压完的段不回头重写。
3. **拼装顺序固定**:system 人设/红线/工具定义打头(每轮完全相同)→ 层 2 截短对话 → 层 1 最近原文 → 用户当前这句 → 早期梗概 + 检索证据(+ 退款路订单数据)合成一条,挂在用户这句之后。梗概**不单独占一条 system**:上游 chat template 会把所有 system 上提合并,工具定义被挤到可变内容之后,前缀缓存整段作废。
4. **token 预算从窗口倒推**:扣掉 system/工具定义、检索证据、注入梗概、输出预留、用户输入预留、单轮 ReAct 瞬时峰值与安全余量;历史预算取「想留住的轮数 × 每轮稳态占用」与「窗口实际匀得出来的」两者较小值;层 1 拿七成、层 2 拿三成。启动自检,连一轮都装不下就报警。中文按字数折 token 的口径(`chars_per_token`)与预算常量一起校准。
5. **双轨并行**:`conversations` / `messages` 两表落对话;State 的 `messages` 挂 `add_messages` reducer,checkpoint 存全量历史;调模型前用 DB 已落消息 + 本轮 State 切片另拼一份裁过、带梗概的精简版。完整历史与精简版各走各的,互不影响。
6. **上下文可观测**:每轮调模型前把实际发出的上下文写进 `log/app.log`——主力 Agent 打 `model_ctx`(梗概全文 + 滑窗逐条 + 窗口条数 + tokens 估算),coref/意图共用的历史打 `history_ctx`(摘要行 + 滑窗,每轮必打,闲聊兜底轮也有);摘要任务触发/开始/完成/跳过/失败全留痕,带覆盖边界与耗时。
7. **前端多会话**:聊天页加会话侧栏,列出该用户全部会话(新在前、首问预览、已摘要标记),点击切换并回载历史原文继续聊;「新对话」只开新会话,旧会话仍可切回。后端两个只读接口 `GET /api/conversations`、`GET /api/conversations/{id}/messages`;侧栏加载失败静默降级。

**非目标:** 语义检索捞历史;主题重要度留关键事实;跨会话长期记忆;用户画像;摘要任务持久化重试/队列(失败只 log,下一轮自然重触发)。

## 2. 与桌面 Spec/Plan 的关系(本仓裁定)

| # | 桌面 Spec/Plan(第一步) | 本轮需求 + `ch_07.sql` 第二步 | 本章处理 |
|---|---|---|---|
| A1 | 双层:滑窗 + `conversations.summary` 一列 | 三层:层 1 原文 / 层 2 截短 / 层 3 分段梗概 | 按三层实现。`conversations.summary` 保留,降级为**最近几段梗概的投影**,供注入;真源是 `conversation_summaries` 表 |
| A2 | 摘要按「新增 N 条消息」触发(`summary_trigger_messages=30`) | 触发看 token 用量,不数条数(工具结果不落库,数条数看不见它涨) | 删掉条数配置,改为层 2 渲染 token > `layer2` 预算触发 |
| A3 | 旧摘要 + 新段 → 滚动重写,单列覆盖 | 一段一行只追加;旧梗概只作背景,不参与合并 | 摘要任务只输出本批新段;插入 `conversation_summaries(seq+1)`;投影重建;旧段内容永不改 |
| A4 | 摘要以第二条 `SystemMessage` 紧跟人设 | 梗概与会话材料挂在用户当前这句之后 | system 恒为 `AGENT_SYSTEM`;梗概 + 证据 + 退款订单数据合成一条 `HumanMessage` 插在当前用户消息后 |
| A5 | 入口写 user 消息拿 `db-{msg_id}` 锚点入 State | 需求 5:State 全量 + 精简版双轨 | **保留 ch05/ch06 单点写**:user/assistant 仍只在 `log_node` 落库(失败轮不留半条);精简历史在入口从 DB 读已落 user/assistant 行,带 `db-{id}` 锚点;State.messages 只负责全量 checkpoint |
| A6 | `trim_messages(max_tokens=context_window_max_tokens=3000)` 固定窗口 | 预算从模型窗口倒推,层 1/层 2 分预算 | 新建 `app/core/context_budget.py`,按 §7 公式计算;`trim_messages` 仅作无锚点消息列表的兜底 |
| A7 | 配置 4 项(`summary_trigger_messages` 等) | 演示配置 6 项 + 预算常量 | 配置按 §7 重列;演示环境变量沿用 `MODEL_CONTEXT_WINDOW / MAX_OUTPUT_TOKENS / MAX_USER_INPUT_TOKENS / MAX_AGENT_STEPS / TOOL_RESULT_MAX_TOKENS / RERANK_TOP_K` |
| A8 | 前端侧栏 224px、<780px 隐藏 | 相同 | 沿用;另外页面加载时若 localStorage 有会话 id,自动回载历史 |

## 3. 三层与边界语义

`ch_07.sql` 的约定(**闭/开区间以 SQL 注释为准**):

```
id ≤ summary_upto_msg_id                 层 3:已进分段梗概
summary_upto < id ≤ layer1_from_msg_id   层 2:半压(规则截短)
id > layer1_from_msg_id                  层 1:原文,一字不压
```

- 两个锚点都指向 `messages.id`(user/assistant 行;tool 行不落库)。
- 边界只允许落在「轮」之间:层 1 从一条 **user 消息**开始,`layer1_from_msg_id` 是它前一条 assistant 消息的 id(没有则为 0)。这样切窗不会从 assistant/tool 半截开始,ReAct 结构不会被切碎。
- 降级:层 1 超预算 → 从最老的轮开始整轮移入层 2,只把 `layer1_from_msg_id` 前移;数据一行不搬。
- 摘要完成:`summary_upto_msg_id` 追到 `layer1_from_msg_id`(层 1 起点),层 2 清空;新段追加,旧段不动。
- 运行时从 DB 读出的行一律带 `db-{id}` 锚点:旧会话也走正常的层 1 降级/摘要,不会因为“老”而改走别的路径。无锚点的消息列表(直接调用 / 旧 checkpoint / 测试)才退化为 `trim_messages` 纯 token 裁剪,不改 DB 边界、不崩。

## 4. 双轨:State/checkpoint 与 DB 读取

- **全量轨**:`ConversationState.messages` 挂 `add_messages`;各节点只吐新消息;checkpoint(`AsyncSqliteSaver`)保留完整历史,一条不删。工具调用/结果只在这条轨里存在。
- **精简轨**:每轮入口 `repository.list_dialog_messages(cid)` 读**已落库**的 user/assistant 行(按 id 升序),构造带 `id=f"db-{row.id}"` 的 `HumanMessage`/`AIMessage`,作为 `history_messages` 放进本轮 State。层 2/层 1 用它拼;本轮新增消息从 `state["messages"]` 的**最后一条 HumanMessage 起**切片,二者拼接,不重不漏。
- 为什么不用桌面 Plan 的「入口写 user + State 锚点」:本仓 ch05/ch06 已把 user+assistant 统一放在 `log_node` 落库,并有「失败轮不落 MySQL」的测试;入口写会改变该语义,还要在 `log_node` 处理 assistant 锚点回填。改用「DB 已落行 + 本轮 State 切片」不改写路径,边界仍是 DB id,且与需求 5 的双轨表述一致。
- 代价:每轮多一次 `list_dialog_messages` 查询(有 `idx_conversation_id`,量级可忽略);层 2/层 1 不含历史 tool 行(它们本来就不落库,也不需要进对话上下文)。

## 5. 每轮数据流

```
run_turn / stream_turn(runtime)
  ├─ _ensure_conversation → cid / Conversation(summary, summary_upto, layer1_from)
  ├─ rows = repository.list_dialog_messages(cid)          # user/assistant,id 升序
  ├─ history_messages = [Human/AI(id=f"db-{row.id}") ...]
  ├─ degrade_layer1(history_messages, layer1_from, layer1 预算)
  │     └─ 超预算:整轮移入层 2;UPDATE conversations.layer1_from_msg_id;log「layer1 降级 X→Y」
  ├─ _graph_input:
  │     messages=[HumanMessage(本轮原话)]                 # 全量轨;log 后落库
  │     history_messages=[id > summary_upto 的已落行]      # 精简轨原料(层 2+层 1)
  │     summary / summary_upto_msg_id / layer1_from_msg_id
  ↓
图:
  coref / classify_intent  ← summary_line + history_messages 最近 6 条;log history_ctx
  agent_llm:
     msgs = [SystemMessage(AGENT_SYSTEM)]                 # 恒定
          + render_layer2(history_messages ≤ layer1_from) # 用户原话 + 客服前 40 字
          + layer1(history_messages > layer1_from)        # 原文
          + 本轮 State 从最后一条 HumanMessage 起的切片     # 当前用户 + ReAct 轨迹
          + HumanMessage(turn_ctx) 插在当前用户之后        # 梗概 + 证据 + 退款订单数据
     log model_ctx; 调模型
  ↓
log_node: 落 user + assistant 到 MySQL(唯一写点)
  ↓
轮结束: summarizer.maybe_schedule_summary(cid)           # 不 await create_task
     ├─ 层 2 渲染 token ≤ layer2 预算 → log skip
     └─ 超预算且无在跑任务 → log trigger → 后台 run_summary
```

interrupt/续跑:首轮 `fetch_order` 暂停时尚未落库;resume 沿用本轮 State 里的 `history_messages` 与当前用户消息,`log_node` 在续跑结束落一次 user+assistant;`stream_resume`/`resume_turn` 结束后同样做触发检查。

## 6. 上下文拼装(`nodes._agent_messages`)

固定顺序,只有一条 `SystemMessage`:

```
[System(AGENT_SYSTEM)]                       # 人设 + 红线 + 工具定义(恒定,不含任何可变内容)
[层 2 截短消息 ...]                          # 用户原话;客服答复前 40 字 + 「…」;大块工具结果一行标识
[层 1 原文消息 ...]                          # DB 已落 user/assistant 逐字(历史 tool 行不落库,天然不在)
[当前用户 HumanMessage]
[HumanMessage(turn_ctx)]                     # 梗概全文 + 知识证据 + 退款订单数据 + 判定指令
[本轮 ReAct 后续消息 ...]                    # AI(tool_calls) / Tool / 最终 AI
```

- `turn_ctx` 插在「最后一条 HumanMessage」之后;没有用户消息则追加。第 1 步 `[Sys, 层2, 层1, 用户, turn_ctx]` 必须是第 2 步 `[Sys, 层2, 层1, 用户, turn_ctx, AI(tool), Tool]` 的严格前缀,轮内前缀缓存才命中。
- 摘要/证据/订单数据**绝不进 system**;`AGENT_SYSTEM` 一个可变字符都不拼。
- `turn_ctx` 是发给模型的构造消息,不写入 State(不会污染下一轮历史)。
- `_history_text`(coref/意图):`summary_line + history_messages 最近 6 条`;生产路径用 `history_messages`,单元测试没给时回落到 `state["messages"][:-1]`,保持旧测试可用。

## 7. token 预算(`app/core/context_budget.py`)

### 7.1 公式

```
fixed       = system_prompt_tokens + tool_schema_tokens + system_overhead_tokens
            + max_output_tokens + max_user_input_tokens
            + rerank_top_k × evidence_chunk_tokens
            + summary_budget_tokens + safety_margin_tokens
react_peak  = max_agent_steps × tool_result_max_tokens
leftover    = model_context_window - fixed - react_peak
desired     = desired_history_turns × turn_token_estimate
history     = max(0, min(desired, leftover))
layer1      = max(0, history * 7 // 10 - 1)         # 七成(整数运算,避免浮点);1 token 边界余量
layer2      = max(0, history - layer1 - 1)
```

- `fixed` 里每一项都是 `settings` 字段,可从 env 覆盖。默认值按当前 `AGENT_SYSTEM` 与 6 个工具 schema 实测取整:
  - `chars_per_token=2.0`;`estimate_text_tokens(text)=ceil(len(text)/2.0)`;
  - `AGENT_SYSTEM` 777 字 → 389 token;6 个工具 schema JSON 2161 字 → 1081 token;chat template/角色包装 `system_overhead_tokens=30`;
  - 三项合计 **1500**。改提示词或工具后重测这三个数,连同 `chars_per_token` 一起校准。
  - `count_tokens_approximately` 对中文偏乐观(777 字只报 199 token),2.0 字/token 更保守,也与演示数值一致;两者必须成对调整,单独改一个会把预算推反。
- 中文折算统一用同一个 `estimate_text_tokens`:预算常量、层 1/层 2 计量、`model_ctx` 估算共用一套口径,只改一个会反向。
- `layer1 + layer2 = history - 1`;那 1 token 明写为「边界余量」,用于对齐演示数值,不承载业务。
- **口径**:`history` / `layer1` / `layer2` 只算**已落库的历史**;用户当前这句话由 `fixed` 里的 `max_user_input_tokens` 单独预留,不重复计入层 1,也永远保留、不参与降级。它超过预留时只 log warning,不拿历史预算补。

### 7.2 演示配置(验收标准 2)

```
MODEL_CONTEXT_WINDOW=18000  MAX_OUTPUT_TOKENS=2000  MAX_USER_INPUT_TOKENS=2000
MAX_AGENT_STEPS=3           TOOL_RESULT_MAX_TOKENS=1200  RERANK_TOP_K=5
```

| 项 | 计算 | 值 |
|---|---|---|
| fixed | 389+1081+30+2000+2000+5×400+750+500 | 8750 |
| react_peak | 3×1200 | 3600 |
| leftover | 18000-8750-3600 | 5650 |
| desired | 30×1000 = 30000 | — |
| **history(滑窗)** | min(30000, 5650) | **5650** |
| **layer1 预算** | 5650×7//10-1 | **3954** |
| **layer2 预算** | 5650-3954-1 | **1695** |

### 7.3 只调窗口的失败现场(自检要报警)

只设 `MODEL_CONTEXT_WINDOW=18000`,其余保持默认(`MAX_AGENT_STEPS=6`、`RERANK_TOP_K=10`):

| 项 | 值 |
|---|---|
| fixed | 389+1081+30+2000+2000+10×400+750+500 = 10750 |
| react_peak | 6×1200 = 7200 |
| leftover | 18000-10750-7200 = 50 |
| history / layer1 / layer2 | 50 / 34 / 15 |

`layer1=34 < turn_token_estimate=1000` → 启动自检 `logger.error("上下文预算不足: ...")`,层 1 实际归零。

### 7.4 默认配置(验收标准 3)

`model_context_window=64000`、`max_agent_steps=6`、`rerank_top_k=10`、`desired_history_turns=30`、`turn_token_estimate=1000`:

- fixed=10750,react_peak=7200,leftover=46050,history=min(30000,46050)=30000,layer1=20999,layer2=9000。
- 目标:20 轮普通对话(每轮稳态 ≤1000 token)全部留在层 1,不触发降级/摘要。以 `scripts/smoke_ch07.py` 实测校准;若默认窗口下 20 轮仍触发,调 `desired_history_turns`/`turn_token_estimate` 或默认窗口,并记录到 dev-notes。

### 7.5 自检

`context_budget.self_check()` 在 `runtime.init_graph()` 启动时调用:`history <= 0` 或 `layer1 < turn_token_estimate` → `logger.error("上下文预算不足: ...")`;不抛异常,应用仍可起。

## 8. 后台分段摘要(`app/core/summarizer.py`)

- **触发(看用量)**:轮结束后 `maybe_schedule_summary(cid)`:
  1. 读 Conversation 与 `list_dialog_messages`;
  2. `layer2_rows = summary_upto < id <= layer1_from`;空 → log `summary skip`(layer2_empty);
  3. 按 §6 规则渲染层 2,估算 token;≤ `layer2` 预算 → log `summary skip`(below_budget);
  4. 该 cid 已有在跑任务 → log `summary skip`(running);
  5. 否则 log `summary trigger conv=... layer2_tokens=... budget=... upto=... layer1_from=...`,`asyncio.create_task(run_summary(cid))`,不 await;`_running[cid]` 防抖,done 回调移除。
- **任务体** `run_summary(cid)`:
  1. 读 Conversation 与全部 user/assistant 行;
  2. 新边界 = `layer1_from_msg_id`;`≤ summary_upto` 或空批 → log skip;
  3. 批 = `summary_upto < id ≤ 新边界`;
  4. `summarize_dialog(old_projection, dialog)`:旧梗概只作背景,输出**本批**新段,严格 JSON `{"summary": "..."}`(`structured_model`);
  5. 成功:`repository.append_summary(...)` 在**一个事务**里插入 `conversation_summaries(seq=max+1, from_msg_id=批首, upto_msg_id=新边界, content)`、重建 `conversations.summary` 投影(最近段拼到 `summary_budget_tokens` 以内)、`summary_upto_msg_id=新边界`;log `summary done conv=... seq=N from=... upto=... len=... cost=...ms`;
  6. 失败/解析失败:log exception,不更新任何字段;计数条件仍满足,下一轮自然重触发。
- **提示词红线**:只提炼事实与诉求(问过哪款商品、报过的订单号/手机号、明确诉求、未决问题);对话里没出现的一个字不许编;寒暄闲聊不留;长度几十到一两百字;旧梗概只作背景,不参与合并。
- **投影**:`conversations.summary` 是最近几段梗概的拼接(段间用 `\n`),供注入;注入时 `summary_line` 把换行压成空格,给 coref/意图一行。真源始终是 `conversation_summaries`,投影可在任务里从表重建。

## 9. 数据库与 repository

- `app/db/models.py`:
  - `Conversation` 增加 `summary: Text | None`、`summary_upto_msg_id: BigInteger | None`、`layer1_from_msg_id: BigInteger | None`;
  - 新增 `ConversationSummary`(对应 `sql/ch07-ddl.sql`)。
- `sql/ch07-ddl.sql`:采用用户提供的 `ch_07.sql`(两段 ALTER + `conversation_summaries` 建表)。
- `tests/conftest.py`:`_DDL_FILES` 追加 ch07;筛选收 `ALTER TABLE`;`_TABLES` 首部加 `conversation_summaries`。
- `app/db/repository.py` 新增:
  - `list_dialog_messages(cid) -> list[Message]`(user/assistant,id 升序;摘要、历史回载共用);
  - `update_layer1_from(cid, msg_id) -> None`;
  - `append_summary(cid, from_msg_id, upto_msg_id, content) -> int`(插段 + 重建投影 + 追 `summary_upto`,单事务);
  - `list_summaries(cid) -> list[ConversationSummary]`(测试/诊断);
  - `list_conversations(user_id, limit=50) -> list[dict]`(id/status/preview/has_summary/created_at,按 `Conversation.id desc` 新在前)。
- 索引/约束按 `ch_07.sql`:`uk_conv_seq(conversation_id, seq)`、`idx_conv_upto(conversation_id, upto_msg_id)`。

## 10. 可观测

- **`model_ctx`**:`agent_llm` 每次调模型前 `logger.info`,一条包含:cid、step、summary 全文、窗口条数、`estimate_messages_tokens`、每条非 system 消息的 `[type] 'content[:40]'`。这是「摘要和滑窗真的进了模型」的直接证据。
- **`history_ctx`**:`coref` 每轮打一次,内容是传给 coref/意图的 `summary_line + 最近 6 条`;闲聊兜底轮也经过 coref,必然可见。
- **`layer1 降级 X→Y`**:runtime 层 1 降级时打,带旧/新边界、层 1 token、预算。
- **摘要**:`summary trigger / start / done / skip / failed`,带 cid、层 2 token、预算、边界、段号、长度、耗时 ms;失败 `exc_info`。
- **落盘**:`app/main.py` 给 `app` logger 加 `FileHandler(log/app.log, encoding="utf-8")`;`log/` 已在 `.gitignore`。StreamHandler 保留,终端仍可见。
- 启动自检「上下文预算不足」走 `logger.error`,同进 app.log。

## 11. 前端会话侧栏(`app/static/index.html`)

- 布局:`.app` 改为横向 flex;左侧 `#sidebar`(224px,像素风,`border-right`),右侧 `.chat` 保持现有纵向布局(topbar/messages/composer)。`<780px` 隐藏侧栏,聊天占满。
- 侧栏:标题「会话」;列表项 = 首问预览(截断)+ 创建时间 + `已摘要` 徽标(有 `summary_upto` 时);当前 `conversationId` 高亮。点击 `switchConversation(id)`。
- `switchConversation(id)`:`GET /api/conversations/{id}/messages` → 清空消息区 → 逐条渲染(user 纯文本、bot `renderMarkdown`)→ `saveConversationId(id)` → 滚动到底。失败**静默**:`console.warn`,保留当前聊天,不清空。
- 页面加载:`loadConversations()`;若 localStorage 有 `conversationId`,尝试回载历史,失败则显示欢迎语并保留 id(聊天仍可继续);没有则欢迎语。
- 每轮 `sendMessage` / `resumeWithOrder` 结束(含中断暂停)刷新侧栏;`done` 帧拿到新 cid 后高亮新会话。
- 「+ 新对话」:只清 `conversationId`、清消息区、显示欢迎语并刷新列表高亮;旧会话仍在侧栏。
- 侧栏任何失败不影响 `/api/chat` 主链路。

## 12. 错误与边界

- 摘要 JSON 失败/模型异常 → log + 放弃本次,不写任何字段;下一轮重触发。
- 摘要任务与下一轮并发:任务只写 `conversation_summaries` + `conversations` 三字段;入口读到旧边界最多晚一轮生效。
- interrupt 续跑:`Command(resume)` 不新增用户消息;沿用本轮 `history_messages`;续跑结束落库并做触发检查。
- 旧会话无边界:DB 行在运行时一律带 `db-{id}` 锚点,因此旧会话与普通会话一样走层 1 降级/摘要(行为正确且更好);`build_layers` 的纯 `trim_messages` 兜底只覆盖无锚点消息列表(直接调用/旧 checkpoint),不改 DB 边界。
- 边界不变量:入口把 `layer1_from_msg_id` 夹到 `≥ summary_upto_msg_id`;降级只向前移,永不越过 `summary_upto`,层 2 区间恒为非负。
- 单轮超过层 1 预算:保留最新一轮,log `layer1 单轮超预算`,不无限降级。
- 层 2 渲染超预算:当前轮只带**最新且装得下的整轮**(user 提问 + 对应客服答复,不拆单条),旧的等后台摘要;摘要任务按完整层 2 计量并触发。
- 服务重启:在跑任务丢失无妨;触发条件仍满足,下一轮重起。
- 会话不存在:沿用 `ConversationNotFound`;两个只读接口对不存在会话返回 404。

## 13. 测试与验证

- **代码走 TDD**(测试库 `mewhelp_test` 惯例):
  - `context_budget`:演示数值 5650/3954/1695;只调窗口时 50/34/15 且自检报警;默认配置 30000/20999/9000;`system_prompt_tokens`/`tool_schema_tokens` 与当前 `AGENT_SYSTEM`/工具 schema 的实测值一致(提示词或工具变了,这条先红);
  - `memory`:`split_layers`(边界命中/空层 2/旧会话无锚点)、`degrade_layer1`(整轮移动/单轮超预算/边界不越过 `summary_upto`)、`render_layer2`(用户全留、AI 前 40 字、大块 tool 一行)、层 2 token 预算裁剪;
  - `summarizer`:`compute_boundary`(=layer1_from)、触发/跳过/防抖、`run_summary` 追加段 + 投影 + 原子更新、旧段不变、失败不写;
  - `list_conversations` 排序/预览按 §9;侧栏不每次 touch `conversations.updated_at`(见 D12);
  - repository(真库):日志/摘要读写、`list_dialog_messages` 过滤 tool、`append_summary` seq/投影、`list_conversations` 预览/has_summary/用户隔离;
  - nodes:`_agent_messages` 拼装顺序、system 恒定、本步前缀、`_history_text` 摘要行 + 滑窗、`_log_model_context`(caplog);
  - API 路由层打桩:`GET /api/conversations`、`GET /api/conversations/{id}/messages`(透传/404)。
- **摘要 prompt 标注样例(以 eval 代 TDD)**:`scripts/eval_ch07.py` 三条样例——事实保留/寒暄不留、无编造数字实体、旧梗概背景不合并(只出新事实);断言严格 JSON、关键事实、长度、无寒暄。`make eval-ch07`。
- **冒烟** `scripts/smoke_ch07.py`:
  1. 默认配置聊 22 轮(短消息,约 50–100 字/轮):断言 log 中无 `layer1 降级`、无 `summary trigger`,context 不爆——默认窗口装得下就不该压;
  2. 演示配置聊 30 轮(**长消息**,约 200 字/轮,内容含订单号/手机号/诉求):断言出现 `layer1 降级 X→Y`、`summary trigger`、`summary done 第N段`,摘要落库含早期订单号;25 轮实测层 2 渲染约 1515 token < 1695,30 轮才过阈值(实测校准,见 dev-notes);
  3. 第 26 问「最开始那个订单后来怎么说」能凭梗概答出早期订单号/诉求(允许模型措辞差异,至少摘要库里有事实)。
  - 口径:演示数值(5650/3954/1695)是 token 预算,不承诺「任意 25 条短消息都级联」;短对话装得下就不压,长对话才触发,这正是需求 3 与需求 2 的分工。
- **前端**:浏览器手动验收(用户):开多会话、切换回载、20+ 轮默认不压、演示配置触发级联、`grep model_ctx / history_ctx / summary log/app.log`。
- **最终验收(用户)**:需求 5 条验收标准逐条在浏览器/日志中核对。

## 14. 决策清单

| # | 决策 | 理由/代价 |
|---|---|---|
| D1 | 以本轮需求 + `ch_07.sql` 第二步(三层/分段追加)为准,桌面 Spec/Plan 作第一步背景 | 两者在层数、触发条件、摘要合并方式上互斥;用户消息是最新任务书 |
| D2 | 精简历史从 DB 已落 user/assistant 行构造,不改 `log_node` 单点写 | 保留「失败轮不落 MySQL」语义与现有测试;每轮多一次轻查询 |
| D3 | system 恒为 `AGENT_SYSTEM`;梗概/证据/退款数据合成一条 `HumanMessage` 挂当前用户之后 | 防止上游模板上提 system 导致前缀缓存全 miss |
| D4 | 预算常量:system 389 + tools 1081 + 包装 30 + output 2000 + user 2000 + evidence 400×top_k + summary 750 + safety 500(按 `chars_per_token=2.0` 实测);peak=steps×tool_result;layer1=history*7//10-1、layer2=history-layer1-1 | 精确复现演示 5650/3954/1695;只调窗口时 50/34/15 并报警,与验收描述一致 |
| D5 | 层 2 超预算时当前轮只带最新装得下的一段,后台摘要按完整层 2 计量 | 当前轮不爆窗;被省略的旧段下一轮前由摘要覆盖 |
| D6 | 摘要任务轮后 `create_task`、按 cid 防抖、失败不重试 | 不阻塞当轮回复;条件仍在,下轮自然重触发 |
| D7 | DB 行一律带 `db-{id}` 锚点,旧会话按新机制正常管理;无锚点消息列表 → 全部层 1 + `trim_messages` 兜底 | 运行时旧会话也能享受摘要;纯裁剪只覆盖无锚点列表 |
| D8 | `log/app.log` 由 `FileHandler` 写;`model_ctx` 每次模型调用、`history_ctx` 每轮一次 | 验收直接 grep;日志目录已 gitignore |
| D9 | 前端页面加载自动回载 localStorage 会话;侧栏失败静默 | 长对话验收不依赖单会话 localStorage |
| D10 | `layer1`/`layer2` 只算已落库历史;当前用户消息由 `max_user_input_tokens` 预留,不重复计入,也永不降级 | 避免与 fixed 里的用户输入预留双算,演示 3954 才是历史可用量 |
| D11 | 冒烟/演示允许用长消息制造阈值;短对话装得下就不降级/摘要 | 预算是 token 阈值,不是「25 轮必压」;与验收 3「装得下就不压」一致 |
| D12 | 侧栏按 `Conversation.id desc` 排(新会话在前),显示创建时间;不每轮 touch `updated_at` | 现有写路径只往 messages 插行;为排序加一次无谓写不值,创建顺序即「新在前」 |
| D13 | 层 2 当前轮裁剪按整轮(user+对应答复),不按单条消息 | 避免模型看到没有提问的孤立客服答复 |

## 15. 验收映射

| 用户验收 | 本章落点 |
|---|---|
| 1. 连聊 20+ 轮 token 不爆、系统不崩 | §7.4 默认预算 + `smoke_ch07` 22 轮 |
| 2. 演示配置级联:层 1 降级 → 层 2 触发摘要 → 落第 N 段 → 问「最开始那个订单」靠梗概答对 | §7.2/§7.3 + §8 + `smoke_ch07` 演示配置 |
| 3. 默认窗口 20 轮不触发降级/摘要 | §7.4 + `smoke_ch07` 第 1 组断言 |
| 4. 日志可见摘要痕迹且不阻塞回复;grep `model_ctx`/`history_ctx` | §10 |
| 5. 前端侧栏多会话、切回回载、接着聊 | §11 + 浏览器手动验收 |
