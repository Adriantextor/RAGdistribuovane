from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    node_api_token: SecretStr
    database_url: str = "sqlite:///./data/app.db"
    rag_url: str = "http://127.0.0.1:8001"
    inference_url: str = "http://127.0.0.1:8002"
    ollama_url: str = "http://127.0.0.1:11434"
    qdrant_url: str = "http://127.0.0.1:6333"
    llm_model: str = "qwen3:1.7b"
    embed_model: str = "embeddinggemma"
    max_attempts: int = Field(default=3, ge=1, le=10)
    lease_seconds: int = Field(default=60, ge=6)
    worker_concurrency: int = Field(default=2, ge=1, le=8)
    worker_poll_seconds: float = Field(default=1, ge=0.05)
    retry_base_seconds: float = Field(default=3, ge=0.05)
    job_timeout_seconds: float = Field(default=420, ge=1)
    request_timeout_seconds: float = Field(default=300, ge=1)
    cache_ttl_seconds: int = Field(default=3600, ge=0)
    rag_data_dir: str = "data/rag"
    import_concurrency: int = Field(default=2, ge=1, le=4)
    embedding_batch_size: int = Field(default=8, ge=1, le=16)
    retrieval_top_k: int = Field(default=4, ge=1, le=6)
    retrieval_min_score: float = Field(default=0.48, ge=0, le=1)
    llm_context_size: int = Field(default=4096, ge=2048, le=32768)
    llm_max_tokens: int = Field(default=512, ge=64, le=2048)
    model_keep_alive: str = "0s"
    inference_parallel: int = Field(default=1, ge=1, le=4)

    @field_validator("node_api_token")
    @classmethod
    def check_token(cls, value):
        token = value.get_secret_value()
        if len(token) < 32 or token.startswith("replace-"):
            raise ValueError("Set a random NODE_API_TOKEN with at least 32 characters")
        return value

    def headers(self):
        return {"Authorization": f"Bearer {self.node_api_token.get_secret_value()}"}


@lru_cache
def settings():
    return Settings()
