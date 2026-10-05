import logging
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def check_cuda():
    device = "cpu"
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Found GPU: {torch.cuda.get_device_name(0)}. Using CUDA.")
    else:
        logger.info("CUDA not found. Using CPU.")
    return device


class Settings(BaseSettings):
    fastapi_port: int
    model: str = "jinaai/jina-embeddings-v5-omni-nano-retrieval"
    modalities: str = "vision"
    hf_home: str = "D:/models/"
    device: str = check_cuda()

    surrealdb_host: str
    surrealdb_user: str
    surrealdb_password: str
    surrealdb_namespace: str
    surrealdb_name: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]