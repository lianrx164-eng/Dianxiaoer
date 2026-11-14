from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 这三项故意不给默认值,缺了在启动就报 Field required,直接指向 .env
    chat_model: str
    chat_base_url: str
    chat_api_key: str
    # 提取任务的上游思考开关。默认 None = 不发送该参数,保证换上游只改 .env;
    # DeepSeek 需设 disabled(思考模式不支持强制 tool_choice)
    extract_thinking: str | None = None
    database_url: str = "mysql+asyncmy://root:root@localhost:3306/mewhelp_ch02?charset=utf8mb4"
    test_database_url: str = "mysql+asyncmy://root:root@localhost:3306/mewhelp_test?charset=utf8mb4"

    # 嵌入:阿里云百炼系列,OpenAI 兼容网关;key 同聊天一样不给默认值
    embed_base_url: str = "https://maas.qianwenaiapi.com/compatible-mode/v1"
    embed_api_key: str
    embed_model: str = "text-embedding-v4"
    milvus_uri: str = "data/milvus_knowledge.db"
    retrieval_top_k: int = 3
    retrieval_min_score: float = 0.4

    # 重排:百炼原生 rerank(qwen3-rerank),与嵌入同一账号,key 缺省复用
    rerank_url: str = "https://maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank"
    rerank_api_key: str | None = None
    rerank_model: str = "qwen3-rerank"
    recall_top_k: int = 50
    rerank_top_k: int = 10
    rerank_min_score: float = 0.3

    # ch05 图编排
    max_agent_steps: int = 6              # ReAct 环最大步数(封顶,超则兜底)
    checkpointer_db_path: str = "data/ch05_checkpoints.sqlite"

    # ch06 意图识别模型配置(先求准:默认大模型;降级路仅种子,本章不实现运行时)
    intent_model: str = ""              # 意图识别模型名;空=回落 chat_model
    intent_small_model: str = ""        # 降级路小模型(种子,未接运行时)
    intent_mode: str = "accuracy"       # accuracy=只用大模型;cost=小模型判→低置信升级(未实现)
    intent_conf_threshold: float = 0.6  # cost 模式升级阈值(种子)

    # ch07 会话上下文预算(全部可从 env 覆盖;演示见 spec §7.2)
    model_context_window: int = 64000
    max_output_tokens: int = 2000
    max_user_input_tokens: int = 2000
    tool_result_max_tokens: int = 1200
    chars_per_token: float = 2.0
    system_prompt_tokens: int = 389       # AGENT_SYSTEM 777 字 @ 2.0
    tool_schema_tokens: int = 928         # 5 个内置工具 schema 1856 字 @ 2.0
    system_overhead_tokens: int = 30
    evidence_chunk_tokens: int = 400
    summary_budget_tokens: int = 750
    safety_margin_tokens: int = 500
    desired_history_turns: int = 30
    turn_token_estimate: int = 1000
    layer2_assistant_chars: int = 40
    layer2_tool_keep_chars: int = 120

    # ch08 工具系统
    mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"
    mcp_aftersales_url: str = "http://127.0.0.1:8102/mcp"
    tool_default_timeout: float = 5.0
    mcp_tool_timeout: float = 10.0
    mcp_discovery_timeout: float = 3.0
    tool_max_retries: int = 2
    demo_ticket_delay_seconds: float = 0.0


settings = Settings()
