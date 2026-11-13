from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 这三项故意不给默认值,缺了在启动就报 Field required,直接指向 .env
    chat_model: str
    chat_base_url: str
    chat_api_key: str
    token_budget: int = 2000
    # 提取任务的上游思考开关。默认 None = 不发送该参数,保证换上游只改 .env;
    # DeepSeek 需设 disabled(思考模式不支持强制 tool_choice)
    extract_thinking: str | None = None
    database_url: str = "mysql+asyncmy://root:root@localhost:3306/mewhelp_ch02?charset=utf8mb4"
    test_database_url: str = "mysql+asyncmy://root:root@localhost:3306/mewhelp_test?charset=utf8mb4"


settings = Settings()
