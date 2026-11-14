# Ch06 意图识别与对话管理设计:占位分流器 → 正式版

> 以用户桌面 `CH_06/Sepc.md` 为基础,按本仓 ch05 实际基线修订。凡与桌面文档不同之处,均在 §2 列明理由。

## 1. 背景与目标

ch05 骨架把分流器前段留成占位:`coref` 原样透传、`classify_intent` 单标签七类无 confidence/无「其他」、知识路只做同义词轻扩、路由无退款子流程。本章把分流器做成正式版。

**目标(需求 7 条):**

1. **指代消解 + Query 改写(合一)**:LLM 结合最近几轮历史,把「它能退吗」补全成脱离上下文也能看懂的完整问句;已完整/指代已明确的原样透传,不强改。
2. **Query 扩写**:一句问题泛化成严格 3 条侧重点不同的检索查询(强制 JSON,单字段 `queries`),多 query 检索后去重合并;**只对退款退货/售后核心场景做**;检索侧现查现用,库里知识只留一份。
3. **意图识别四件套(LLM prompt 路线,不训模型)**:①八类枚举成选择题(物流/订单/商品咨询/退款退货/售后/投诉/闲聊/**其他**);②强制 JSON `{intent, confidence}`;③边界 few-shot;④「其他」兜底,拿不准归它。
4. **分流精化**:退款退货/售后走确定性子流程——先拿订单数据、再 Query 扩写并强制检索政策条款,最后只把「这一单能不能退」交主力 Agent 判。
5. **槽位处理**:缺订单号不让模型猜,执行阶段弹订单选择器点选回填;退款原因提交退款单时固定类目下拉;需求澄清留主力 Agent。
6. **前端配套**:订单选择器做成聊天流里可点选的订单卡片,点选自动回填、流程接着走;退款单是简单表单,退款原因固定类目下拉。
7. **模型配置**:意图识别默认配够格大模型(先求准);降级路(小模型判 + 低置信升级大模型)本章只落配置种子。

**非目标:** 微调小模型/BERT 意图分类;跨会话记忆;意图降级路的运行时实现(仅配置种子);两级分类/检索式意图识别;ch06 课程图 21 SVG。

## 2. 与桌面 Spec/Plan 的基线差异(本仓裁定)

| # | 桌面文档假设 | 本仓实际 | 本章处理 |
|---|---|---|---|
| A1 | 基线节点名 `resolve_reference / retrieve_knowledge / main_agent` | 本仓 ch05 为 `coref / forced_rag / agent_llm` | **保留本仓节点名**,把 `coref` 做实;新增 `fetch_order / retrieve_policy / script_reply`;退役 `chitchat_reply`。行为与验收不变 |
| A2 | `docker compose`、`make milvus-up`、`.venv/bin/python` | Windows、Milvus Lite、`uv run`、本地 MySQL | 命令全部改用 `uv run` / `make` 既有目标;不引入 Docker |
| A3 | 裸 `with_structured_output` | 本仓统一 `structured_model`(function_calling + 关闭上游思考) | 意图/扩写都走 `structured_model`;`get_chat_model` 加 `model` 覆盖参 |
| A4 | 扩写检索传 `bm25_text` | `search_knowledge(..., bm25_query=...)` | 按仓库签名 |
| A5 | 闲聊文案「我暂时还不会回答这个…」 | ch05 已有更自然的问候语 | 闲聊沿用 `CHITCHAT_REPLY_TEXT`(已含「引回产品」),「其他」用新文案 |
| A6 | `log_node` 与入口都写用户消息 | ch05 已统一为 `log_node` 单点写 MySQL | 保持单点写;resume 轮也只在 `log` 落一次 |
| A7 | 未覆盖「暂停中用户又发新消息」 | 实测 langgraph 1.2.21 会接受新输入并再次触发同一中断,新消息不处理 | runtime 显式检测未决中断:重新吐 `interrupt` 事件(带 `conversation_id`),不启动新一轮;见 D9 |
| A8 | `tickets` 枚举只改模型 + ALTER | 测试库由 `sql/ch02-ddl.sql` 建表 | 同步改 `ch02-ddl.sql` 的 ENUM,另加 `sql/ch06-ticket-type.sql` 给存量库 |
| A9 | `order_id` 取模型传入值 | 模型可能漏传/传错 | 拦截 `submit_refund` 时优先用工具参数,缺失回落 `state["order_id"]` |
| A10 | 意图解析失败回「其他」 | 与 ch05「未知归 business」哲学不同 | 按本章 spec 回「其他」(最保守),记录为 D12 |

## 3. 数据流与图骨架

```
START
 → coref              指代消解 + Query 改写(吃最近几轮历史;已完整透传)→ resolved_query
 → classify_intent    四件套;→ intent(8 类) + intent_confidence + route
 → route_by_intent  [条件边,五出口]
     投诉             → escalate          complaint_reply(ch05:安抚 + 转人工/建工单两按钮)→ log
     闲聊 / 其他      → fallback_script   script_reply(按意图分文案,零模型)→ log
     商品咨询         → knowledge         forced_rag(强制检索)→ confidence_check →(强)agent_llm
                                                                                 →(弱)fallback_reply
     物流 / 订单      → business          agent_llm(ch05 ReAct,直接交)
     退款退货 / 售后  → refund_flow       fetch_order →(缺单 interrupt/resume)→ retrieve_policy → agent_llm
```

- 主力 Agent(`agent_llm ↔ agent_tools`)全图共用一个;退款路也复用它判「能不能退」。
- 「其他」与闲聊共用 `fallback_script` 出口,靠 `intent` 分文案。
- `fallback_reply`(知识路证据弱,生成前兜底)与 `script_reply`(分流后固定话术)是不同节点,不可合并。

### 3.1 八类意图 → 五出口(`routing.INTENT_TO_ROUTE`)

| intent | route |
|---|---|
| 投诉 | `escalate` |
| 闲聊、其他 | `fallback_script` |
| 商品咨询 | `knowledge` |
| 退款退货、售后 | `refund_flow` |
| 物流、订单 | `business` |
| 未知/缺失(兜底) | `business` |

条件边映射:`escalate→complaint_reply`、`fallback_script→script_reply`、`knowledge→forced_rag`、`refund_flow→fetch_order`、`business→agent_llm`。返回值与映射键单一来源。

## 4. 指代消解 + Query 改写(`coref` 做实)

- 一次 LLM 调用完成 coref + 口语归一。输入 = 本轮消息 + 最近 6 条 human/ai 历史(`_history_text`);输出一句完整问句写 `state["resolved_query"]`。
- 透传边界:prompt 明确「已完整、指代已明确的原样返回,不引入历史里没有的信息」;问平台通用规则的问题(「运费怎么算」)即使上文有订单也不塞订单号。
- 指代指向某笔订单时,改写必须带订单号(同商品多单,只补商品名下游锁不到单)。
- 下游:`classify_intent`、`forced_rag`、`retrieve_policy` 都用 `resolved_query`(意图识别不再改写)。
- 出错兜底:返回原句,不阻断下游;trace `coref = "rewrite" | "passthrough"`。
- 验证:纯 Prompt → `scripts/eval_coref.py`(带指代补全 + 已完整透传两类)。

## 5. Query 扩写(`query_understanding.expand_queries`)

- 把一句问题泛化成**严格 3 条**检索友好查询,强制 JSON 单字段 `{"queries": [...]}`(扁平 `list[str]`,避开 glm 嵌套 502)。
- 规则:保持产品名/型号一致、不引入原问题没有的参数、每条短且侧重点不同。
- 调用点:只在 `refund_flow` 的 `retrieve_policy`;商品咨询 `knowledge` 路沿用 ch05 同义词轻扩。
- 检索侧现查现用:3 条各检索一次 → 按 chunk id 去重(保留最高 `rerank_score`)→ 按分降序 → `arrange_head_tail` → 拼编号证据。库里知识只留一份。
- 模型异常/全空回落 `[query]`,保证至少一条可查。
- 验证:纯 Prompt → `scripts/eval_expand.py`。

## 6. 意图识别四件套(`intent.classify` 重写)

1. **选择题**:八类 + 每类边界说明。
2. **强制 JSON**:`_Intent{intent: Literal[8类], confidence: 0-1}`;解析失败/越界 → 「其他」+ confidence 0.0,记 warning。
3. **few-shot**:边界样例(「这个还能退吗」→退款退货;「退货运费谁出」→商品咨询;「猫爬架坏了能保修吗」→售后;怪问题→其他)。
4. **兜底类「其他」**:拿不准归它,不硬塞业务意图。

- 输入:`resolved_query`(空则回落原话)+ 最近历史(应对物流→退款→物流漂移)。
- 模型:默认 `settings.intent_model`(空则回落 `chat_model`),先求准;`confidence` 照常输出但不参与分流。
- 配置种子(`config.py`):`intent_model=""`、`intent_small_model=""`、`intent_mode="accuracy"`、`intent_conf_threshold=0.6`;本章不写升级逻辑。
- 节点 `classify_intent` 写 `intent / intent_confidence / route`,trace 用 `intent_confidence` 键(避开 `confidence_check` 的 `confidence`)。
- 验证:纯 Prompt → 扩 `scripts/eval_intent.py`(8 类 + 其他 + 边界 + 多轮漂移 + JSON 稳定)。

## 7. 退款退货/售后确定性子流程(`refund_flow`)

```
route = refund_flow
 → fetch_order:
     order_id = state.order_id 或从 resolved_query 抽 4+ 位数字(不猜、不编)
     缺 → interrupt({"type":"select_order","orders": list_user_orders(user_id)})
        resume 回填订单号
     order_data = order_snapshot(order_id)       # 与 query_order 工具同源
 → retrieve_policy:
     queries = expand_queries(resolved_query + 订单状态)   # 严格 3 条
     多 query 强制检索政策 → 按 chunk id 去重合并 → 拼编号证据
 → agent_llm(注入 order_data + 政策证据 + 退款判定指令):
     能退 → 调 submit_refund → agent_tools 拦成 {"type":"refund_form"} 前端可选项(不写库)
     不能退 → 说清原因 + 引导人工(不出按钮)
 → log → END
```

- `submit_refund` 工具承载两出口(仿 ch05 `create_ticket` 拦截),不加额外分类调用。
- 强制检索写死一趟:不让模型凭记忆答「能不能退」;证据为空时指令要求引导人工,不臆造。
- `list_user_orders(user_id)`:mock,按 user_id 稳定出 2-4 单,每单用 `order_snapshot` 同源,选中即可 `query_order`。
- `fetch_order` 在 `interrupt()` 之前只做只读(`list_user_orders`),因为 resume 时节点从头重跑。

## 8. interrupt/resume 全链路(已实测 langgraph 1.2.21)

- 节点内 `from langgraph.types import interrupt, Command`。
- **实测中断面形状**(2026-11-14):
  - 非流式 `ainvoke` 返回 dict 含 `"__interrupt__": [Interrupt(value=...), ...]`,取 `out["__interrupt__"][0].value`;
  - `astream(stream_mode=["messages","updates"])` 的中断出现在 `updates` chunk:`{"__interrupt__": (Interrupt(...),)}`;
  - `aget_state(config).interrupts` 为 `(Interrupt(...),)`,`.next` 指向被中断节点;
  - `ainvoke(Command(resume=v), config)` 从被中断节点**从头重跑**,`interrupt()` 返回 `v`;
  - 暂停中传入新输入会被接受,但会再次触发同一中断,新消息不处理(见 D9)。
- **runtime**:
  - `run_turn` 返回 `{conversation_id, state, interrupt}`;`interrupt` 为 orders 或 None。
  - `resume_turn(cid, value)` 用 `Command(resume=value)` 续跑。
  - `stream_turn` 遇中断吐 `{"type":"interrupt","conversation_id":cid,"kind":"select_order","orders":[...]}` 并结束本次流。
  - `stream_resume(cid, value)` 复用同一事件映射续流。
  - `stream_turn` 开跑前若发现未决中断(D9),重新吐同一 `interrupt` 事件并返回,不启动新一轮。
- **API**:
  - `/api/chat` 映射 `interrupt` 帧。
  - `POST /api/actions/resume {conversation_id, order_id}` → SSE,事件与 `/api/chat` 同构。
  - `/api/agent` 响应加 `interrupt: list | None`。

## 9. 退款工单与数据模型

- `submit_refund` 工具:args `order_id`(+可选 `reason`);被 `agent_tools` 拦成 `{"type":"refund_form","draft":{"order_id","reason"}}`,不写库;合成 ToolMessage 促收敛。
- `POST /api/actions/create-refund {conversation_id, order_id, reason}`:写 `tickets`,`ticket_type='退款'`,`description="退款申请 订单号=... 原因=..."`;固定原因类目:七天无理由/质量问题/发错货/不想要了/其他。
- 数据模型:`Ticket.ticket_type` Enum 加 `退款`;`sql/ch02-ddl.sql` 同步(测试库建表依据);新增 `sql/ch06-ticket-type.sql`(`ALTER ... MODIFY COLUMN`)给存量库。

## 10. 会话状态(`ConversationState`)

新增无 reducer 标量,`_graph_input` 入口清零:

- `resolved_query: str`
- `intent_confidence: float`
- `order_id: str`
- `order_data: dict`

`intent` 值域扩到 8 类;`route` 值域改为五出口;`evidence/citations/suggested_actions` 复用。

## 11. 前端(Vibe Coding)

- 收 `interrupt` 帧 → 聊天流渲染可点选订单卡片(订单号/商品/状态/金额);点选 → `POST /api/actions/resume` → 同一套 SSE 消费函数继续渲染续流(政策判定 → 退款按钮/说明)。
- 收 `refund_form` 动作 → 「提交退款工单」按钮 → 弹表单(订单号只读 + 原因固定下拉,不预选、必选)→ `POST /api/actions/create-refund` → 显示退款单号。
- 复用 ch05 `.action-bar/.action-btn/.ticket-form` 样式;新增 `.order-card`。
- 不套 brainstorm/TDD/code-review,浏览器端到端验收。

## 12. 验证策略

| 产出 | 验证 |
|---|---|
| coref / expand / 意图四件套(纯 Prompt) | 标注评估集:`eval_coref.py`、`eval_expand.py`、扩 `eval_intent.py` |
| routing / State / fetch_order / retrieve_policy / submit_refund 拦截 / runtime resume / 退款端点 | TDD |
| interrupt/resume/stream surface | `scripts/smoke_interrupt.py` 实跑钉死 |
| 订单卡片 / 退款表单 | Vibe Coding + 浏览器验收 |
| 四验收端到端 | `scripts/eval_ch06.py` + 浏览器 |

## 13. 决策记录

- **D1** 订单选择器用 LangGraph `interrupt`/`Command(resume)`(实测通过),不写两轮重入。
- **D2** 退款单复用 `tickets` 表(`ticket_type='退款'`),不新建退款表。
- **D3** 意图降级路只落配置种子,`intent_mode` 恒 `accuracy`。
- **D4** 退款两出口用 `submit_refund` 工具承载,`agent_tools` 拦截为前端动作。
- **D5** 指代消解 + Query 改写合一在 `coref` 节点;意图识别只判不改写。
- **D6** Query 扩写只在 `refund_flow` 检索侧;库里知识一份。
- **D7** 五出口按 README;闲聊 + 其他共用 `fallback_script`,按 intent 分文案;闲聊文案沿用 ch05。
- **D8** 沿用 ch05:置信度闸在 Agent 之前;节点名 `coref/forced_rag/agent_llm` 不变;trace 键 `coref/forced_rag` 保留。
- **D9** 暂停中收到新消息:runtime 重新吐 `interrupt`(带 conversation_id),不启动新一轮、不落 MySQL;用户应点选订单或开新对话。理由:最小、无状态污染;取消哨兵方案留待后续。
- **D10** `submit_refund` 的 `order_id` 以工具参数优先、缺失回落 `state["order_id"]`,避免前端表单空单号。
- **D11** 意图解析失败归「其他」(最保守),confidence=0,记 warning。
- **D12** `orders_snapshot` 抽取时正则用 `(?<!\d)(\d{4,})(?!\d)`(中文与数字之间无 `\b` 边界)。

## 14. 风险与回退

- **R1** interrupt surface 形状已实测;仍写入 `smoke_interrupt.py` 供回归。若升级 langgraph 后形状变化,以冒烟为准。
- **R2** resume 时被中断节点从头重跑:`fetch_order` 的 `interrupt()` 之前只做只读;测试覆盖「缺单→interrupt→resume→query_order 只一次」。
- **R3** 意图 JSON 稳定性:结构化输出 + 失败归「其他」;评估集统计解析失败率。
- **R4** `tickets` 枚举 ALTER 对存量数据:只加值、不动旧行;本地库执行一次并确认。
- **R5** 模型非确定性:eval 抖动如实重跑记录,不粉饰。

## 15. 验收映射

1. 多轮(物流→退款→物流)每轮意图对 + 指代对 → `eval_ch06.py` 多轮用例 + `eval_intent.py` 历史用例。
2. 意图 JSON 稳定可解析、怪问题落「其他」→ `eval_intent.py`。
3. 「这个能退吗」先补指代、再走退款子流程拿订单+政策 → trace(`resolved_query` + `route=refund_flow` + `retrieve_policy`)+ eval。
4. 浏览器不带订单号问退款 → 弹订单选择器(interrupt)→ 点选(resume)→ 子流程走完 → 浏览器真跑 + 截图。
