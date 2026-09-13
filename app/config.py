from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
        gemini_api_key: str
        primary_model: str = "gemini-3.1-flash-lite"
        fallback_model: str = "gemini-3.1-flash-lite"

        langchain_tracing: bool = True
        langchain_api_key: str
        langchain_project: str

        app_env: str = "development"
        log_level: str = "debug"
        ratelimit: str = "20/minute"
        cache_ttl_seconds: int = 300
        max_retries: int = 3

        model_config: dict = {"env_file": ".env", "extra": "ignore"}

        @property
        def is_production(self) -> bool:
            return self.app_env == "production" 

@lru_cache()
def get_settings() -> Settings:
    return Settings()