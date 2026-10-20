from functools import lru_cache

from pydantic import SecretStr, field_validator
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
    max_attempts: int = 3
    lease_seconds: int = 600

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
