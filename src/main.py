import asyncio
import json
import logging
import mimetypes
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from surrealdb import AsyncSurreal, RecordID
from surrealdb.errors import InvalidRecordIdError

from src.config import Settings, get_settings
from src.state import ServiceState
from src.utils import check_deps, setup_logging, setup_process_prio
from src.worker import ModelJob, ModelWorker

setup_logging(logging.INFO)
check_deps()

logger = logging.getLogger(__name__)

settings = get_settings()
state = ServiceState()
worker = ModelWorker(state)

job_queue: asyncio.Queue[RecordID] = asyncio.Queue()


def _setup_runtime(s: Settings) -> None:
    os.environ.setdefault("HF_HOME", s.hf_home)
    if s.device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
        if sys.platform == "win32":
            setup_process_prio()


def _build_proxy_url(raw: str | None) -> str | None:
    if not raw:
        return None
    if raw.startswith(("http://", "https://")):
        return raw
    # SurrealDB file pointer: "bucket:/path" -> our /files proxy.
    bucket, sep, path = raw.partition(":")
    if not sep:
        return raw
    path = path.lstrip("/")
    if not bucket or not path:
        return None
    return f"http://127.0.0.1:{settings.fastapi_port}/files/{bucket}/{path}"


# Keeping a file pointer embedded in SurrealQL requires only that neither the
# bucket nor path can break out of the `f"..."` literal. That means no `"`,
# no backslash, no whitespace/control chars. Everything else is inert inside a
# string and needs no further character whitelisting.
def _safe_literal_segment(seg: str) -> bool:
    if not seg:
        return False
    for ch in seg:
        if ch in ('"', "\\") or ch.isspace() or not ch.isprintable():
            return False
    return True


async def _connect_db(s: Settings) -> Any:
    db = AsyncSurreal(s.surrealdb_host)
    await db.signin({"username": s.surrealdb_user, "password": s.surrealdb_password})
    await db.use(namespace=s.surrealdb_namespace, database=s.surrealdb_name)
    logger.info("Connected to SurrealDB (%s/%s)", s.surrealdb_namespace, s.surrealdb_name)
    return db


async def _read_file_bytes(db: Any, bucket: str, path: str) -> bytes | None:
    if not _safe_literal_segment(bucket) or "/" in bucket:
        raise ValueError("invalid bucket name")
    if not _safe_literal_segment(path):
        raise ValueError("invalid file path")

    result = await db.query(f'RETURN f"{bucket}:/{path}".get();')
    if result is None:
        return None
    if isinstance(result, bytes):
        return result
    raise ValueError(f"unexpected file result: {type(result).__name__}")


class Broadcaster:
    def __init__(self) -> None:
        self._subs: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self, data: dict) -> None:
        for q in list(self._subs):
            q.put_nowait(data)


broadcaster = Broadcaster()


def _parse_record_id(raw: str) -> RecordID:
    try:
        return RecordID.parse(raw)
    except (InvalidRecordIdError, ValueError) as e:
        raise HTTPException(422, f"invalid record id: {raw}") from e


async def _dispatch(db: Any) -> None:
    while True:
        rid_obj = await job_queue.get()
        rid = str(rid_obj)
        state.set_task(rid, "processing")
        logger.info("Processing %s", rid)

        try:
            rec = await db.select(rid_obj)
            if isinstance(rec, list):
                rec = rec[0] if rec else None
        except Exception as e:
            logger.warning("Failed to select %s: %s", rid, e)
            state.set_task(rid, "error", str(e))
            state.bump(ok=False)
            broadcaster.publish({"record_id": rid, "status": "error", "error": str(e)})
            continue

        if not rec or not isinstance(rec, dict):
            state.set_task(rid, "error", "record not found")
            state.bump(ok=False)
            broadcaster.publish({"record_id": rid, "status": "error", "error": "record not found"})
            continue

        text = (rec.get("generated_description") or "").strip() or None
        image_proxy = _build_proxy_url(rec.get("image_url"))

        if text is None and image_proxy is None:
            state.set_task(rid, "error", "no text or image to embed")
            state.bump(ok=False)
            broadcaster.publish({"record_id": rid, "status": "error", "error": "no text or image to embed"})
            continue

        worker.jobs.put(ModelJob(record_id=rid, text=text, image_url=image_proxy))


async def _persist(db: Any) -> None:
    while True:
        result = await asyncio.to_thread(worker.results.get)
        rid = result.record_id

        if result.error:
            state.set_task(rid, "error", result.error)
            state.bump(ok=False)
            broadcaster.publish({"record_id": rid, "status": "error", "error": result.error})
            continue

        data: dict[str, Any] = {}
        if result.text_embedding is not None:
            data["text_embedding"] = result.text_embedding
        if result.image_embedding is not None:
            data["image_embedding"] = result.image_embedding

        try:
            await db.merge(RecordID.parse(rid), data)
        except Exception as e:
            logger.warning("Failed to persist %s: %s", rid, e)
            state.set_task(rid, "error", str(e))
            state.bump(ok=False)
            broadcaster.publish({"record_id": rid, "status": "error", "error": str(e)})
            continue

        state.set_task(rid, "done")
        state.bump(ok=True)
        broadcaster.publish({"record_id": rid, "status": "done"})
        logger.info("Embedded and saved %s", rid)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_runtime(settings)
    db = await _connect_db(settings)
    app.state.db = db

    worker.start()
    dispatch_task = asyncio.create_task(_dispatch(db))
    persist_task = asyncio.create_task(_persist(db))
    app.state.dispatch_task = dispatch_task
    app.state.persist_task = persist_task
    logger.info("Service ready (model loading in background)")
    yield

    for t in (dispatch_task, persist_task):
        t.cancel()
    try:
        await db.close()
    except Exception as e:
        logger.debug("Error closing DB connection: %s", e)


app = FastAPI(title="Embedding Service", lifespan=lifespan)


class GenerateRequest(BaseModel):
    items: list[str]


@app.post("/generate", status_code=202)
async def generate(payload: GenerateRequest):
    if not payload.items:
        raise HTTPException(400, "items must not be empty")

    accepted: list[str] = []
    for raw in payload.items:
        rid_obj = _parse_record_id(raw.strip())
        rid = str(rid_obj)
        state.set_task(rid, "queued")
        await job_queue.put(rid_obj)
        broadcaster.publish({"record_id": rid, "status": "queued"})
        accepted.append(rid)

    return {"accepted": accepted, "count": len(accepted)}


@app.get("/health")
async def health():
    snap = state.snapshot()
    return {
        "worker_status": snap["model"],
        "model_error": snap["model_error"],
        "device": settings.device,
        "modalities": settings.modalities,
        "uptime_seconds": snap["uptime_seconds"],
        "metrics": {
            "total_processed": snap["processed"],
            "total_errors": snap["errors"],
        },
        "pending": snap["pending"],
    }


@app.get("/tasks")
async def tasks():
    return state.all_tasks()


@app.get("/tasks/{record_id}")
async def task_status(record_id: str):
    task = state.get_task(record_id)
    if task is None:
        raise HTTPException(404, "unknown task")
    return {"record_id": record_id, **task}


@app.get("/files/{bucket}/{path:path}")
async def get_file(bucket: str, path: str):
    try:
        data = await _read_file_bytes(app.state.db, bucket, path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("Failed to read file %s:%s: %s", bucket, path, e)
        raise HTTPException(500, "failed to read file")

    if data is None:
        raise HTTPException(404, "file not found")

    media = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return Response(content=data, media_type=media)


@app.get("/queue/stream")
async def queue_stream():
    async def gen():
        q = broadcaster.subscribe()
        try:
            yield _sse({"type": "snapshot", "data": state.snapshot()})
            while True:
                msg = await q.get()
                yield _sse({"type": "task", "data": msg})
        finally:
            broadcaster.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"