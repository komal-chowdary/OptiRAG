
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "OptiRAG"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "nemotron-3-nano:4b"

    CHROMA_PATH: str = "storage/chroma_db"

    VECTOR_COLLECTION: str = "microsoft_docs_v2"
    SEMANTIC_COLLECTION: str = "microsoft_docs_semantic_v2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
