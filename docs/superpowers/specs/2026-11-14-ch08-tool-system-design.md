# Ch08 工具系统设计:静态内置工具 → 注册中心 + 执行引擎 + MCP + 建单确认流

> 以用户桌面 `CH_08/Sepc.md`、`CH_08/Plan.md`、`CH_08/ch_08.sql` 为基础,按本仓 ch07 之后的实际基线修订。桌面文档与本仓冲突之处在 §2 逐条列明;桌面文档未覆盖、但本仓现状暴露的问题,按 §8 决策记录处理。

## 1. 背景与目标

ch02–ch07 的工具层是「几个写死的内置工具」:`registry.py` 是静态列表 + `NO_RETRY / INJECT_CONVERSATION / TOOL_TIMEOUTS` 三个特例集合;`infra.py` 只做超时和重试,没有参数校验、没有权限分层、没有审计;`create_ticket` 在 `agent_tools` 里被硬编码成前端按钮。本章升级成即插即用的工具系统。

**目标(需求 7 条):**

1. **工具注册中心**:内置与 MCP 工具一律带齐工具名、用途描述、JSON Schema 参数定义,登记进同一份注册表;内置工具服务启动时登记,MCP 工具连上 Server 动态发现、现问现拿;新工具注册进来就能被主力 Agent 用上,不改核心代码、不重启服务。
2. **参数校验**:执行前统一按 JSON Schema 校验;拦下不抛异常,把校验错误说明包成一条工具结果回灌模型。
3. **权限控制**:工具分只读/写两类,写操作由执行引擎把门;本系统写操作只有 `create_ticket`,只有客户明确要求才发起,真正执行要等前端确认回来。外部 MCP 工具的用途声明不可信,权限只认我们侧规则。
4. **执行引擎**:所有工具调用走同一处,统一超时、重试、错误处理、结果格式化;重试只给暂时性故障,业务空结果不重试,写操作不自动重试;错误按参数不合法、查询落空、真故障三类分诊;结果挑回答用得上的字段、内部枚举码翻人话、JSON 中文不转义。
5. **审计留痕**:`tool_audit_logs` 每次调用落一条,含被权限拒、被校验拦的;不挂外键;审计写失败不许反拦工具执行。
6. **MCP 接入**:自建物流、售后两台业务 MCP Server,独立进程、Streamable HTTP、mock 数据不接真实系统不建表;客服系统作为 MCP Client 接入,与内置清单合成一份;内置 `query_logistics` 下线,物流查询由 MCP 接管,不留重名。
7. **建工单确认流(带前端)**:客户明确要求才走;Agent 先核对信息、缺必填主动追问、不许编;凑齐后发起 `create_ticket`,执行引擎不直接执行,用 LangGraph interrupt 推工单预览给前端;确认 → resume 放行落 `tickets` 表、回复带工单号;取消 → 不执行、审计落「权限拒绝」;ch05 投诉流程前端按钮建单路径保持不动。

**非目标:** Skill 机制;仓储等更多外部系统;MCP Server 侧鉴权;注册表落库/跨进程注册中心;工具清单缓存与失效策略(本章刻意现问现拿);工具调用成本统计(归 ch09)。

**技术栈(定死):** MCP Server 用 MCP 官方 Python SDK 的 Streamable HTTP 传输;MCP Client 用 `langchain-mcp-adapters` 的 `MultiServerMCPClient`。实装口径(`adapters 0.3.2` 要求 `mcp<2.0.0`,故落到官方 SDK v1):

- 官方 SDK v1:`from mcp.server.fastmcp import FastMCP`;`FastMCP("name", host=..., port=...)` 在构造器收传输参数,`run(transport="streamable-http")` 启动;默认端点为 `/mcp`。v2 的 `MCPServer` + `run(host=,port=)` 写法在 `mcp<2` 下不可用。
- adapters:连接 `{"transport": "streamable_http", "url": ...}`;`get_tools(server_name=...)`;每次工具调用/清单拉取新建 session,因此「Server 侧加工具、只重启 Server」天然成立;`handle_tool_errors=False` 让 MCP 执行错误抛 `ToolException`,由本仓执行引擎统一分诊。

## 2. 与桌面 Spec/Plan 的差异(本仓裁定)

| # | 桌面文档 | 本仓基线/本轮裁定 |
|---|---|---|
| A1 | Plan 在 `summary_model` 行后追加 ch08 配置 | `summary_model` 在 ch07 审查时已删除;ch08 配置追加在 ch07 预算块之后 |
| A2 | Plan 用 `_graph_input("u1", text, 1, 1, "", 0)` 造测试输入 | ch07 后 `_graph_input(user_id, message, cid, prev_messages, ctx)` 接收上下文 dict;测试按新签名对位 |
| A3 | Plan 说改 `app/api/chat.py` 的 interrupt 帧 | 帧组装实际在 `app/api/sse.py::event_frames`;改那里,`chat.py` 与 `actions.py` 共用 |
| A4 | Plan 未提 ch07 预算常量 | ch08 改 `AGENT_SYSTEM` 与工具清单,ch07 的 `system_prompt_tokens`/`tool_schema_tokens` 校准测试必红;本章新增「预算校准」任务(§4.10) |
| A5 | Spec 记录「意图第九类『人工』」为实现期拍板 | 本轮用户已确认按此执行,纳入本章范围(§4.7) |
| A6 | Plan 的 `query_faq` 超时 30s | 现网是 45s(RAG 管线含改写+检索+重排+自评);保留 45s,不降 |
| A7 | Plan 的 MCP 清单拉取只有 try/except | 增加 `asyncio.wait_for` 发现超时,防单台 Server 挂起拖住整轮(§4.3) |
| A8 | Plan 对一轮多个 `create_ticket` 直接回一条文本、不落审计 | 需求 5 要求每次调用都留痕;其余调用过引擎、按未确认写拒绝并落「权限拒绝」审计(§4.6) |
| A9 | Plan 未提 `.env.example` | 红线:修改 `.env.example` 前单独向用户确认(§8 D12) |

## 3. 方案与备选

**选定方案 A:进程内动态注册表 + 六段式执行管道。** `registry.py` 原位升级为 `ToolSpec` 注册表;内置工具从 `business.py` 迁入 `app/tools/builtin/` 包,启动时 `pkgutil` 扫描导入即注册;MCP 工具每轮现拉、与内置合并;所有调用经 `engine.py` 的「查工具 → 校验 → 权限门 → 执行(超时/重试)→ 分诊 → 格式化 + 审计」管道。`infra.py` 退役。

**方案 B(否):注册表落 MySQL。** 与「MCP 工具现问现拿」直接矛盾:Server 侧新增工具不会自动出现在表里,要么定时同步,要么每次扫描,表只是把内存字典搬到磁盘,还引入缓存失效与事务问题。桌面 Spec 已否决。

**方案 C(否):LangGraph `ToolNode` 定制。** 校验、权限、审计、分诊都要塞进 `ToolNode` 的回调或包装里,框架细节黑盒,教学与排障成本高;且 interrupt 确认流仍需自己写节点逻辑,省不掉。

## 4. 组件设计

### 4.1 ToolSpec 与注册中心(`app/tools/registry.py`)

```python
@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: dict          # 模型可见口径;内置取 tool_call_schema,排除 InjectedToolArg
    tool: BaseTool
    permission: str            # "read" | "write"
    source: str                # "builtin" | "mcp"
    mcp_server: str | None = None
    timeout: float | None = None            # None → engine 按来源取默认
    inject_conversation: bool = False
    retryable: bool = True                  # False → 即使是只读也恒不重试
    format_result: Callable[[dict], dict] | None = None
```

- `WRITE_TOOLS: set[str] = {"create_ticket"}`;`permission_for(name)` 只认这张表,不看 Server 声明。
- `register(spec)`:重名先到者保留,后到丢弃并 `warning`。
- `scan_builtin()`:幂等;`pkgutil.iter_modules(app.tools.builtin.__path__)` 逐模块 import,模块尾部 `registry.register(...)` 完成登记。`builtin_specs()` 惰性触发扫描,测试不必起 lifespan。
- `spec_from_langchain_tool(tool, *, source, ...)`:三样必齐的提取口径——`args_schema` 是 dict(MCP 适配器)直接用;否则取 `tool.tool_call_schema` 转 `model_json_schema()`(`InjectedToolArg` 自动排除,模型看不到 `conversation_id`)。
- `async get_all_specs()`:内置在前;MCP 工具逐个合并,重名丢弃并 warning。每次调用现拉,不缓存工具清单。

### 4.2 内置工具布局(`app/tools/builtin/`)

| 文件 | 工具 | 说明 |
|---|---|---|
| `orders.py` | `query_order`、`query_product` | 原样迁入 |
| `faq.py` | `query_faq` | 迁入,`timeout=45.0`、`retryable=False`(重 RAG,重试会叠加分钟级等待) |
| `tickets.py` | `create_ticket` | 写工具;`inject_conversation=True`;内含 `demo_ticket_delay_seconds` 延迟钩子(验收 6) |
| `refunds.py` | `submit_refund` | 原样迁入 |
| `promotions.py` | `query_promotions` | **不预先入库**;验收 1 演示时从 `scripts/demo_ch08_promotions.py.txt` 复制进来,重启即注册 |

- `query_logistics` **下线不迁**;物流由 MCP Server 以同名工具接管。`query_order` 的描述保留「先拿 tracking_no 再查物流」的引导。
- `business.py` 瘦身为 mock 数据纯函数:`order_snapshot`、`list_user_orders`(签名不动,`graph.fetch_order` 继续用)。
- `main.py` lifespan 里 `await runtime.init_graph()` 之前调用 `registry.scan_builtin()`。

### 4.3 MCP Client(`app/tools/mcp_client.py`)

- 连接配置:`logistics → settings.mcp_logistics_url`,`aftersales → settings.mcp_aftersales_url`;`transport: "streamable_http"`;HTTP `timeout` 取 `mcp_tool_timeout`,让底层请求与引擎超时同量级。
- `_get_tools_of(server_name)` 薄壳,作为单测 monkeypatch 锚点。
- `fetch_mcp_specs()`:两台 Server 用 `asyncio.gather` 并发拉取,每台 `asyncio.wait_for(_get_tools_of(...), settings.mcp_discovery_timeout)`;单台超时/连不上 → `warning` + 跳过,不影响另一台。并发是为了让「两台都挂起」的等待上限仍是一次 3s,而不是 6s。
- 每次拉取新建 session,不缓存清单:验收 3 的机制保证。
- 两台 Server(`mcp_servers/logistics_server.py` / `aftersales_server.py`)各用 `FastMCP`,mock 数据用固定种子稳定;`MOCK_DELAY_SECONDS` 注入延迟(验收 6),`PORT` 覆盖端口。
- 结果格式化只认我们侧登记:`FORMATTERS = {"query_logistics": _fmt_logistics, "query_warranty": _fmt_warranty, "query_return_status": _fmt_return}`;未登记工具透传。格式化挑回答用得上的字段、内部枚举码翻人话(如 `status_code: "IN_TRANSIT"` → `status: "运输中"`),剔除 `carrier_code`/`policy_ref` 这类内部编码,以及 logistics `trace` 中含「内部状态码」的行。

### 4.4 执行引擎(`app/tools/engine.py`)

对外唯一入口:

```python
async def execute_tool_call(
    tool_call: dict, conversation_id: int, specs: dict[str, ToolSpec],
    *, confirmed: bool = False, deny_note: str | None = None,
) -> ToolRun
```

`ToolRun(tool_call_id, name, ok, tool_message, status, retry_count, duration_ms)`。

| 段 | 规则 | 回灌内容 | 审计 status |
|---|---|---|---|
| 查工具 | `specs.get(name)` 为空 | `工具执行失败:未知工具 X` | 失败 |
| 校验 | `Draft202012Validator(spec.json_schema)`,`best_match` 取最相关错误 | `参数校验未通过:<错误>。请修正参数重新调用;缺少的信息请先向用户追问,不要编造。` | 校验拦下 |
| 权限门 | `permission == "write"` 且 `confirmed is not True` | `建工单未执行:<deny_note 或默认说明>` | 权限拒绝 |
| 执行 | 校验通过后才注入 `conversation_id`;`asyncio.wait_for`;只读且 `retryable=True` 默认重试 2 次;写操作与 `retryable=False` 恒 0 次 | — | 成功/失败/超时 |
| 分诊 | 暂时性故障 = `asyncio.TimeoutError`/`TimeoutError`/`ConnectionError`/`httpx.TransportError`;业务异常与 `ToolException` 不重试 | 超时:`工具暂时不可用:执行超时…`;其他:`工具暂时不可用:<异常类名>…` | 超时/失败 |
| 格式化 | `str` 先 `json.loads`,再 `format_result`,最后 `json.dumps(ensure_ascii=False, default=str)` | 业务空结果按正常成功返回,不重试 | 成功 |

- 重试退避 `0.2s × attempt`;`retry_count` 记录实际重试次数。
- 审计字段:`conversation_id / tool_call_id / tool_name / tool_source / mcp_server / arguments / result_summary(截 500 字)/ status / error_message / retry_count / duration_ms`。
- 审计写入整体 `try/except`:失败只 `logger.exception`,绝不改变本次工具结果。
- 未知工具无 `ToolSpec`,审计的 `tool_source` 按 `builtin` 落、`mcp_server` 为空(保守归类,便于发现模型乱造工具名)。

### 4.5 权限模型

- 本系统唯一的写操作是 `create_ticket`;其余内置工具与全部 MCP 工具按只读放行。
- 代码注释明确:生产环境接不受信 MCP Server 时,未登记写操作应默认拒绝;本章两台自建 Server 都是查询类,只读放行足够。这条不实现,只留注记,避免超出本章范围。
- 确认令牌只能由 `agent_tools` 在 interrupt 拿到 `confirmed=True` 后传入;模型无法通过参数伪造。

### 4.6 建工单确认流(`app/graph/nodes.py::agent_tools`)

1. 取 `last.tool_calls`,先 `specs = {s.name: s for s in await registry.get_all_specs()}`。
2. 若有 `create_ticket` 且存在参数通过 `engine.validate_args` 的调用:在**执行任何工具之前** `interrupt({"type": "confirm_ticket", "preview": {"ticket_type": ..., "description": ...}})`,预览取第一个合法调用。interrupt 前只做纯计算,resume 时节点从头重跑安全(照 `fetch_order` 范式)。
3. 参数不通过 → 不 interrupt,交引擎按「校验拦下」回灌并落审计,模型自然向用户追问。`classify_intent` 判出「人工」时置 `ticket_followup=True`(下一轮粘一轮 business),`agent_tools` 在校验失败时也置 True 作兜底——模型有时只口头追问、根本不发 `create_ticket`,不能只靠失败调用来置标记。下一轮 `classify_intent` 见到该标记就**零模型直接判「人工」、路由 business**,并在同一次返回里把标记清掉——只粘一轮。用户若突然换话题,这一轮也走通用 Agent(它本来就能答物流/订单/商品问题),下一轮恢复正常分流。
4. resume 值:`{"confirmed": true}` → 引擎带 `confirmed=True` 真执行,`ToolMessage` 带工单号;`{"confirmed": false}` → 引擎带 `confirmed=False` + 明确 `deny_note`,落「权限拒绝」审计,回灌「用户取消了本次建工单,请不要再发起」。resume 值不是 dict 时按取消处理。
5. 一轮多个 `create_ticket`:只处理第一个参数合法的调用;其余调用也过引擎、按未确认写拒绝或校验拦下落审计,回灌「一次只处理一个建工单请求」。
6. 追问轮补一条 `_TICKET_FOLLOWUP_HINT`(「用户上一轮明确要求建单,这一轮在补充问题;已能概括问题就直接调用,不要额外索要订单号等非必填信息」),只在当前轮最后一条是用户消息时注入——resume 后最后一条是 AIMessage,不再提示,避免确认/取消后模型二次发起。该提示只影响这一轮,不替代权限门。
7. 其余工具(`query_order`、MCP 工具等)在 interrupt 之后照常走引擎;`submit_refund` 的 ch06 拦截保持原样(转前端退款表单,不经引擎、不落审计——它只是动作提议,不执行)。
8. `AGENT_SYSTEM` 工具原则改写:仅用户明确要求建工单/转人工跟进才发起;发起前核对问题描述等必填信息,缺则追问,严禁编造或占位;预览取消后不得擅自重试;补一条「工具清单可能动态变化,按用途描述选用;工具报错时按说明修正或如实告知」。`create_ticket` 的工具描述同步改。
9. ch05 投诉路径不动:`complaint_reply` 仍产出 `create_ticket` 建议动作 → 前端「建工单」按钮 → `POST /api/actions/create-ticket`,点按钮即用户确认。

**interrupt/resume 的传输改造:**

- `runtime._interrupt_payload` 从 Interrupt 序列取整个 dict 载荷(不再只取 `orders`)。
- `_interrupt_event` 返回 `{"type": "interrupt", "conversation_id", "kind", **payload 除 type 外的字段}`;`select_order` 保留 `orders`,`confirm_ticket` 带 `preview`。
- `event_frames` 按 `kind` 透传负载字段,不再写死 `orders`。
- `run_turn`/`resume_turn` 返回的 `interrupt` 变为完整 payload dict;`AgentResponse.interrupt` 相应改为 `dict | None`。
- `POST /api/actions/resume` 先读 `runtime.pending_interrupt(cid)` 校验:`order_id` 与 `confirmed` 都传或都不传、没有未决中断、载荷类型与暂停类型不符(如 `select_order` 暂停时传 `confirmed`)→ 400;只有类型匹配才进入续跑,避免把 dict 当订单号。
- `ResumeRequest`:`conversation_id` 必填;`order_id: str | None`、`confirmed: bool | None`,两者至少一个,否则 400;`order_id` 非空时 resume 值仍是字符串,否则 `{"confirmed": bool}`。

### 4.7 意图第九类「人工」

验收 4 要求「帮我建个工单」能到主力 Agent。现有意图只有八类,这句话会落「其他」→ 兜底话术;带「投诉」的说法会落「投诉」→ ch05 按钮路径。

- `Intent` 增加 `"人工"`;`INTENTS` 随 `get_args` 自动更新。
- 意图 prompt 改九选一,新增边界与样例:明确要求建工单、转人工客服、让人工跟进 → 人工,哪怕同时带着不满;单纯抱怨仍归投诉;「人工客服电话多少」这类问规则的问题仍归商品咨询(问的是信息,不是要求转接)。
- 路由表:`"人工": "business"`。
- 评估集新增 2 条(如「帮我建个工单」「我要转人工跟进」→ 人工);原有 18 条零回归,重跑 `eval_intent` 记录结果。
- 建单追问后的补充句(如「猫砂盆漏电」)单看会被判成售后/商品咨询,靠 §4.6 的一轮粘滞状态兜底:`classify_intent` 判出「人工」即置 `ticket_followup`,下一轮直接判「人工」并清零,不再调分类模型;校验失败路径也置同一标记。
- 已知边界:分类器仍是模型判断,可能把极端说法判偏;不额外加关键词后门(用户已选方案 A)。

### 4.8 前端(`app/static/index.html`,Vibe Coding)

- `readSse` 收到 interrupt 后按 `kind` 分发:`select_order` → 现有订单卡不动;`confirm_ticket` → `renderTicketConfirm(bubble, preview, conversationId)`。
- 预览卡:标题「工单预览」,两行字段「工单类型 / 问题描述」,两个按钮「确认提交」「取消」;样式复用 `.order-cards` / `.order-card` 一族。
- 点任一按钮:两按钮禁用并在气泡内回显选择;新开 bot 气泡,`POST /api/actions/resume {conversation_id, confirmed}` 续流;回复里带工单号(确认)或已取消说明(取消);结束后刷新会话侧栏。
- 刷新丢预览卡,与 ch06 订单选择器同待遇(interrupt 是瞬态,不做历史回载)。

### 4.9 配置 / DDL / 测试库

`app/config.py` 追加:

```python
mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"
mcp_aftersales_url: str = "http://127.0.0.1:8102/mcp"
tool_default_timeout: float = 5.0
mcp_tool_timeout: float = 10.0
mcp_discovery_timeout: float = 3.0
tool_max_retries: int = 2
demo_ticket_delay_seconds: float = 0.0
```

`.env.example` 追加 `MCP_LOGISTICS_URL`、`MCP_AFTERSALES_URL`、`DEMO_TICKET_DELAY_SECONDS`(红线,动手前单独确认)。

- 启停:Makefile `mcp-up` / `mcp-down`(nohup + pid 落 `data/*.pid`,已 gitignore);`scripts/dev.sh` 先 `make mcp-up` 再起 uvicorn,`make dev` 一条命令拉起全栈。

- `sql/ch08-ddl.sql`:桌面 `ch_08.sql` 原样复制(无 ASCII `;` 注释问题,已核)。
- `ToolAuditLog` 模型对齐 DDL,无外键;`repository.insert_tool_audit(...)` 尾部追加。
- `tests/conftest.py`:`_DDL_FILES` 加 ch08;`_TABLES` 首部加 `tool_audit_logs`。
- dev 库用 `docker exec -i dianxiaoer-mysql ... < sql/ch08-ddl.sql` 应用。

### 4.10 ch07 预算常量校准(本章新增任务)

ch08 改了 `AGENT_SYSTEM`(变长)和工具清单(6 个内置变 5 个内置 + 3 个 MCP),ch07 的 `test_budget_constants_match_prompt_and_tools` 会红。

- `system_prompt_tokens`:按新 `AGENT_SYSTEM` 字数 ÷ `chars_per_token` 重测。
- `tool_schema_tokens`:只覆盖内置工具(单测可比对),按 `convert_to_openai_tool` 后的 JSON 字数重测。
- 新增 `mcp_tool_schema_tokens`:三台 MCP 工具 schema 的静态预留,按实测值写入并注释来源;`compute_budget().fixed` 加入该项。
- 更新 `tests/test_context_budget.py` 三组期望值;ch07 历史演示数字(5650/3954/1695)不再复现,这是工具清单变化的必然代价,记录在 dev-notes。
- 未登记 MCP 新工具的 schema 增长靠 `safety_margin_tokens` 兜底;本章不再做动态计量。

## 5. 每轮数据流

```
前端 → POST /api/chat
  → stream_turn → _load_context(ch07 三层上下文不变)
  → coref → classify_intent(九类;人工 → business)
  → agent_llm:
       specs = await registry.get_all_specs()      # 内置 + MCP 现拉
       bind_tools([s.tool for s in specs if 可见])
       → 模型可能发 tool_calls
  → agent_tools:
       specs = 现拉
       create_ticket 参数齐 → interrupt(confirm_ticket, preview) ⇄ 前端卡片
            resume confirmed=true  → engine(confirmed=True)  → tickets 表 + 工单号
            resume confirmed=false → engine(confirmed=False) → 审计「权限拒绝」
       其余工具 → engine.execute_tool_call(...) → ToolMessage 回灌
  → agent_llm 收敛 → log_node 落 MySQL → SSE done
```

- 每步 ReAct 现拉两次清单(bind 与执行各一次);两台 Server 并发,单次拉取最坏 3s(仅 Server 挂起时;未起时连接拒绝是毫秒级),一步最坏 6s;换来零缓存失效逻辑。
- `submit_refund` 仍只对退款子流程可见(`_tools_for` 过滤保持)。

## 6. 失败模式与边界

| 场景 | 行为 |
|---|---|
| 一台 MCP Server 没起 | 发现超时/连接失败 → warning + 跳过该台,内置与另一台照常可用 |
| 两台都没起 | 只剩内置工具;模型不会看到物流工具,不会调它 |
| MCP Server 挂起不响应 | 每次清单拉取最多 3s(两台并发),一个 ReAct 步两次拉取最坏约 6s;不无限挂住整轮 |
| 工具超时 | 只读重试 2 次(退避 0.2/0.4s),最终回灌「工具暂时不可用:执行超时」;写操作 0 次 |
| 参数缺/类型错 | 校验拦下,回灌可读错误,不执行、不弹确认卡 |
| 未确认的写操作 | 权限门拒绝,审计「权限拒绝」 |
| 用户在预览卡点取消 | 不建单,审计「权限拒绝」,提示模型不再发起 |
| 审计库写失败 | 只 log,工具结果照常返回 |
| 模型乱造工具名 | 未知工具回灌 + 审计「失败」 |
| 旧会话/无 interrupt 的客户端 | resume 缺 `order_id`/`confirmed` → 400;resume 值不是 dict 按取消处理,不会崩 |
| 前端刷新 | 预览卡丢失,和 ch06 订单卡一致;重新提问即可 |
| 追问后用户换话题 | 那一轮仍走通用 Agent(能答物流/订单/商品/知识),下一轮恢复正常意图分流;`ticket_followup` 只粘一轮 |

## 7. 测试与验收映射

| 验收 | 验证方式 |
|---|---|
| 1. 丢文件即注册 | `promotions.py` 演示 + registry 单测(扫描/三样齐/重名) |
| 2. 物流走 MCP | 两台 Server 集成测试 + `eval_ch08` 真实对话;验收只问物流轨迹 |
| 3. Server 加工具不重启客服 | 集成测试:重启 Server 后 `get_tools` 立即可见;手工演示 `query_delivery_eta` |
| 4. 追问 → 预览 → 确认落表 | graph 级 interrupt/resume 测试 + `eval_ch08` 样例 + 浏览器手测 |
| 5. 取消 → 权限拒绝审计 | graph 级测试断言审计 status + 浏览器手测 |
| 6. 超时重试/写不重试 | engine 单测(超时重试/写不重试/审计字段)+ 环境变量注入手工演示 |

- 可单测部分走 TDD:registry、engine、repository 审计、mcp_client 降级、intent/routing、graph 确认流、SSE 帧。
- MCP Server 集成测试用子进程 + 端口探活(18101/18102),连不上直接 fail(两台 Server 是本章交付物,不 skip)。
- Prompt 改动(建单追问规矩)不可单测:用标注对话样例跑 `eval_ch08` 验证。
- 前端走 Vibe Coding,不套 TDD;浏览器手测确认/取消两分支。
- 全量 `make test` 必须与 ch07 基线一致(除因工具清单变化而更新的预算测试)。

## 8. 决策记录

| # | 决策 | 理由 | 代价/风险 |
|---|---|---|---|
| D1 | 方案 A:进程内注册表 + 六段管道 | 与「现问现拿」一致,零缓存失效 | 每步两次 MCP 清单拉取(毫秒级) |
| D2 | `query_logistics` 下线,MCP 同名接管 | 不留重名、不让模型二选一 | 物流链路依赖 MCP Server 存活 |
| D3 | 不缓存 MCP 工具清单;两台并发发现,每台超时 3s;单台失败跳过 | 验收 3 的机制保证;单台故障不拖垮对话 | 两台都挂起时每步最多多约 6s |
| D4 | 权限只认 `WRITE_TOOLS`;MCP 默认只读 | Server 声明不可信;本章两台都是查询类 | 生产接写类 MCP 需先改规则 |
| D5 | 确认流 interrupt 顶置,先纯校验再弹卡 | 缺信息不打扰用户;resume 重跑安全 | 一轮里有其他工具时,它们要等确认后才执行 |
| D6 | 多个 `create_ticket` 只处理第一个,其余落权限拒绝审计 | 需求 5「每次调用留痕」 | 模型若发多个,会看到一条拒绝说明 |
| D7 | `submit_refund` 拦截保持 ch06,不经引擎不落审计 | 它不是执行,只是把退款表单交给前端;真正落库走表单端点 | 审计表看不到退款动作提议 |
| D8 | 意图新增第九类「人工」 | 验收 4 的唯一干净路径;用户已确认 | 意图评估集要重跑,依赖上游 |
| D9 | 审计失败只 log | 审计不能反向影响业务 | 极端情况下审计缺失 |
| D10 | 写操作恒不重试;只读且 `retryable=True` 只对暂时性故障重试 2 次;`query_faq` 显式 `retryable=False` | 超时未必没执行,重复建单比失败更糟;`query_faq` 一次 45s,重试会叠加分钟级等待 | 暂时性故障最坏多等约 0.6s |
| D11 | ch07 预算常量重测,新增 MCP schema 预留 | 工具清单变了,预算必须跟着变 | ch07 历史演示数字不再复现 |
| D12 | `.env.example` 修改前单独确认 | CLAUDE.md 红线 | 执行时多一次打断 |
| D13 | `query_faq` 超时保留 45s | 现网 RAG 管线耗时,降到 30s 有回归风险 | 与桌面 Plan 数值不同,已记录 |
| D14 | MCP 结果格式化只对三个已登记工具 | 未知工具不可信,透传更安全 | 新工具若带内部码,需补 formatter |
| D15 | 预览卡不持久化 | 与 ch06 订单卡一致,interrupt 是瞬态 | 刷新后丢失,重新提问 |
| D16 | 建单追问后加一轮 `ticket_followup` 粘滞 | 「猫砂盆漏电」单看会被判售后/商品咨询,粘一轮才能保证验收 4 稳定 | 用户下一轮换话题也会走一次通用 Agent;只影响一轮 |

## 9. 风险与留白

- 意图分类仍可能把个别「建单」说法判偏;评估集只覆盖典型说法,不保证全覆盖。
- MCP 新工具的 schema 增长不受预算动态感知,靠安全余量兜底。
- `agent_tools` 在 interrupt 之前不执行任何工具,若同一轮模型同时发了建单和查询,查询会等确认后才执行;这是为了 resume 安全的刻意取舍。
- 前端预览卡的样式与交互按用户描述迭代,不进自动化测试。
