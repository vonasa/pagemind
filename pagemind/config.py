from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "secrets.local.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Backend axes: local | commercial | anthropic
    index_backend: str = "local"
    query_backend: str = "local"

    # Local OpenAI-compatible backend (Ollama, LM Studio, oMLX, etc.)
    local_base_url: str = "http://localhost:11434"
    local_model: str = "gemma3"
    local_api_key: str = "none"

    # Commercial OpenAI-compatible backend
    commercial_base_url: str = "https://api.openai.com"
    commercial_model: str = "gpt-4o"
    commercial_api_key: str = ""

    # Anthropic backend
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # Embedding server (Infinity)
    embedding_url: str = "http://localhost:7997"

    # Database
    database_url: str = "postgresql://pagemind:pagemind@localhost:5432/pagemind"

    # Runtime
    orchestrator_max_rounds: int = 10


settings = Settings()
