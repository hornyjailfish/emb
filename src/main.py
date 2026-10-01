import asyncio
import gc
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import TypeAlias

import torch

from src.config import Settings, get_settings
from src.handle_changes import close_live_queries, sub
from src.utils import check_deps, setup_logging, setup_process_prio

setup_logging(logging.INFO)
check_deps()

logger = logging.getLogger(__name__)

from fastapi import FastAPI
from surrealdb import (
    AsyncEmbeddedSurrealConnection,
    AsyncHttpSurrealConnection,
    AsyncSurreal,
    AsyncWsSurrealConnection,
    Table,
)

args = get_settings()

def init():
    settings = get_settings()
    os.environ["HF_HOME"] = settings.hf_home

    # Применяем оптимизации для CPU только в том случае, если CUDA недоступна
    if settings.device == "cpu":
        # Ограничиваем количество потоков для CPU
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"

        # Для Windows: выставляем процессу фоновый приоритет (IDLE), чтобы ПК не тормозил
        if sys.platform == "win32":
            setup_process_prio()

HEALTH_STATUS = {
    "status": "starting",          # starting, idle, processing, error
    "device_used": get_settings().device,
    "modalities": get_settings().modalities,
    "last_seen_db": None,
    "processed_count": 0,
    "errors_count": 0,
    "current_record_id": None,
    "uptime_start": time.time()
}

# logger.info(f"Загрузка модели jina-embeddings-v5-omni-nano на устройство [{get_settings().device.upper()}]...")
# logger.info(f"Выбранные модальности: {get_settings().modalities}")
# Инициализируем модель на выбранном устройстве
# model = SentenceTransformer("jinaai/jina-embeddings-v5-omni-nano-retrieval",
#     device=device,
#     trust_remote_code=True,
#     model_kwargs={"modality": args.modalities})
# logger.info(f"Модель успешно загружена на {get_settings().device.upper()}.")
HEALTH_STATUS["status"] = "idle"


DB: TypeAlias = AsyncEmbeddedSurrealConnection | AsyncWsSurrealConnection | AsyncHttpSurrealConnection

async def connect_db(db: DB, args: Settings):
    logger.info(f"Попытка подключения к SurrealDB по адресу {args.surrealdb_host}...")
    try:
        await db.connect()  # pyright: ignore[reportCallIssue]
        token = await db.signin({"username": args.surrealdb_user, "password": args.surrealdb_password})
        await db.use(namespace=args.surrealdb_namespace, database=args.surrealdb_name)
    except Exception as e:
        logger.error(f"Ошибка подключения к SurrealDB: {e}")
        raise
    else:
        logger.info(f" Успешное подключение к SurrealDB ({args.surrealdb_namespace}/{args.surrealdb_name})")
        app.state.db = db
        app.state.token = token


db = AsyncSurreal(args.surrealdb_host+"/rpc")
tasks: set[asyncio.Task] = set()
def done_callback(task: asyncio.Task):
    logger.info(f"Task {task.get_name()} is done")

# for now hardcoded

@asynccontextmanager
async def lifecycle(app: FastAPI):
    await connect_db(db, args)

    HEALTH_STATUS["last_seen_db"] = time.time()
    HEALTH_STATUS["status"] = "idle"


    uuid = await db.live(Table("embedding_queue"))
    stream = await sub(db, uuid)  # pyright: ignore[reportArgumentType]
    if stream is not None:
        tasks.add(stream)
        stream.add_done_callback(done_callback)
    logger.info(f"Создан поток: {stream}")

    yield
    if get_settings().device == "cuda":
        torch.cuda.empty_cache()
    await close_live_queries(db, tasks)
    gc.collect()


app = FastAPI(title="Jina Worker Health Monitor", lifespan=lifecycle)

@app.get("/health")
async def get_health():
    uptime = time.time() - HEALTH_STATUS["uptime_start"]
    db_ok = False
    if HEALTH_STATUS["last_seen_db"]:
        db_ok = (time.time() - HEALTH_STATUS["last_seen_db"] < 30)
    return {
        "worker_status": HEALTH_STATUS["status"],
        "device": HEALTH_STATUS["device_used"],
        "uptime_seconds": int(uptime),
        "tracked_table": get_settings().surrealdb_name,
        "metrics": {
            "total_processed": HEALTH_STATUS["processed_count"],
            "total_errors": HEALTH_STATUS["errors_count"]
        },
        "current_task": HEALTH_STATUS["current_record_id"],
        "db_connected": db_ok
    }
