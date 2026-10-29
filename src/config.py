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
    # MegaLoc hub entry (torch.hub.load). Env: MEGALOC_REPO / MEGALOC_ENTRY
    megaloc_repo: str = "gmberton/MegaLoc"
    megaloc_entry: str = "get_trained_model"
    # Label written to embeddings.model on UPSERT.
    megaloc_model_label: str = "gmberton/MegaLoc"
    # VPR geometry. Default matches MegaLoc eval (322x322 squash).
    # Env: VPR_IMAGE_SIZE / VPR_TRANSFORM
    vpr_image_size: int = 322
    vpr_transform: str = "squash"  # squash | letterbox | center_crop
    hf_home: str = "D:/models/"
    device: str = check_cuda()
    # Seconds to wait for a query encode. 0 / negative = no timeout (debug).
    # Env: QUERY_TIMEOUT
    query_timeout: float = 0.0
    # Maximum records waiting in the dispatch queue (0 = unbounded).
    queue_capacity: int = 0
    # How often to pull embedding_queue status=pending rows (seconds).
    # 0 disables DB polling — only POST /generate feeds work.
    # Env: QUEUE_POLL_INTERVAL
    queue_poll_interval: float = 2.0

    surrealdb_host: str
    surrealdb_user: str
    surrealdb_password: str
    surrealdb_namespace: str
    surrealdb_name: str
    # Auth level for signin: root | namespace | database | record
    # Env: SURREALDB_AUTH_LEVEL  (see Settings.signin_vars)
    surrealdb_auth_level: str = "root"
    # Access method name for record-level auth (DEFINE ACCESS …).
    # Env: SURREALDB_ACCESS — required when auth_level=record
    surrealdb_access: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    def signin_vars(self) -> dict:
        """Build ``db.signin(...)`` payload for the configured auth level.

        Matches SurrealDB Python SDK auth levels
        (https://surrealdb.com/docs/reference/python/concepts/authentication):

        - ``root`` — username + password
        - ``namespace`` — namespace + username + password
        - ``database`` — namespace + database + username + password
        - ``record`` — namespace + database + access + variables
          (``username`` / ``password`` from settings; needs ``surrealdb_access``)
        """
        level = (self.surrealdb_auth_level or "root").strip().lower()
        # Common aliases
        if level in ("ns",):
            level = "namespace"
        elif level in ("db",):
            level = "database"
        elif level in ("scope",):  # pre-2.x name
            level = "record"

        user = self.surrealdb_user
        password = self.surrealdb_password
        ns = self.surrealdb_namespace
        database = self.surrealdb_name

        if level == "root":
            return {"username": user, "password": password}
        if level == "namespace":
            return {
                "namespace": ns,
                "username": user,
                "password": password,
            }
        if level == "database":
            return {
                "namespace": ns,
                "database": database,
                "username": user,
                "password": password,
            }
        if level == "record":
            access = (self.surrealdb_access or "").strip()
            if not access:
                raise ValueError(
                    "surrealdb_access is required when surrealdb_auth_level=record"
                )
            return {
                "namespace": ns,
                "database": database,
                "access": access,
                "variables": {
                    "username": user,
                    "password": password,
                },
            }
        raise ValueError(
            f"unknown surrealdb_auth_level={self.surrealdb_auth_level!r}; "
            f"expected root|namespace|database|record"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
