import asyncio
import json
import logging
import mimetypes
import os
import sys
from contextlib import asynccontextmanager
from queue import Empty
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel
from surrealdb import AsyncSurreal, AsyncSurrealConnection, File, RecordID
from surrealdb.errors import InvalidRecordIdError

from src.config import Settings, get_settings
from src.search import (
    TARGETS,
    aggregate_by_parent,
    bind_filters,
    build_candidate_query,
    candidate_limit,
    get_target,
    hydrate_parents,
)
from src.state import ServiceState
from src.utils import check_deps, setup_logging, setup_process_prio
from src.worker import (
    ModelJob,
    ModelNotReadyError,
    ModelWorker,
    QueryError,
    prepare_image,
)

setup_logging(logging.INFO)
check_deps()

logger = logging.getLogger(__name__)

settings = get_settings()
state = ServiceState()
worker = ModelWorker(state)

job_queue: asyncio.Queue[RecordID] = asyncio.Queue(maxsize=settings.queue_capacity)

# Counts for the current busy wave; reset when the queue returns to idle.
_batch_ok = 0
_batch_err = 0


def _setup_runtime(s: Settings) -> None:
    os.environ.setdefault("HF_HOME", s.hf_home)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "warning")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    if s.device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
        if sys.platform == "win32":
            setup_process_prio()


def _as_file(raw: str | File | None) -> File | None:
    """Coerce a record field into a SurrealDB ``File``, or ``None`` if it isn't one.

    Accepts a native ``File`` (decoded by the SDK from CBOR tag 55) or a string
    form ``bucket:/key`` / ``bucket://key`` (and the optional ``f\"...\"`` wrapper).
    HTTP(S) URLs and anything else return ``None``.
    """
    if raw is None:
        return None
    if isinstance(raw, File):
        return raw
    if not isinstance(raw, str):
        return None
    if raw.startswith(("http://", "https://")):
        return None

    s = raw.strip()
    if len(s) >= 3 and s[0] == "f" and s[1] in "'\"" and s[-1] == s[1]:
        s = s[2:-1]

    bucket, sep, path = s.partition(":")
    if not sep or not bucket or not path:
        return None
    # "bucket://key" and "bucket:/key" both collapse to key "/key".
    while path.startswith("//"):
        path = path[1:]
    if not path.startswith("/"):
        path = f"/{path}"
    if path == "/":
        return None
    return File(bucket, path)


def _build_proxy_url(raw: str | File | None) -> str | None:
    if not raw:
        return None
    if isinstance(raw, str) and raw.startswith(("http://", "https://")):
        return raw
    ref = _as_file(raw)
    if ref is None:
        # Bare non-pointer strings keep their value; a broken ``bucket:…`` form
        # is treated as missing rather than passed through.
        if isinstance(raw, str) and ":" not in raw:
            return raw
        return None
    path = ref.key.lstrip("/")
    if not path:
        return None
    return f"http://127.0.0.1:{settings.fastapi_port}/files/{ref.bucket}/{path}"


def _safe_path_segment(seg: str) -> bool:
    """Reject path/bucket pieces that shouldn't reach the files API."""
    if not seg:
        return False
    for ch in seg:
        if ch in ('"', "\\") or ch.isspace() or not ch.isprintable():
            return False
    return True


# Back-compat alias used by unit tests.
_safe_literal_segment = _safe_path_segment


async def _connect_db(s: Settings) -> AsyncSurrealConnection:
    db = AsyncSurreal(s.surrealdb_host)
    try:
        await db.signin({"username": s.surrealdb_user, "password": s.surrealdb_password})
        await db.use(s.surrealdb_namespace, s.surrealdb_name)
    except Exception:
        await _close_quietly(db)
        raise
    return db


async def _close_quietly(db: Any) -> None:
    try:
        await db.close()
    except Exception as e:
        logger.debug("Error closing partial DB connection: %s", e)


async def _read_file_bytes(db: Any, bucket: str, path: str) -> bytes | None:
    """Read file bytes from a SurrealDB bucket via the SDK ``files`` helper.

    ``path`` may be with or without a leading ``/``. Uses ``db.files.get(File)``
    so the pointer is bound as CBOR tag 55 — no SurrealQL string interpolation.
    """
    if not _safe_path_segment(bucket) or "/" in bucket:
        raise ValueError("invalid bucket name")
    key_body = path.lstrip("/")
    if not _safe_path_segment(key_body):
        raise ValueError("invalid file path")

    data = await db.files.get(File(bucket, key_body))
    # Empty blob is treated as missing so the model never sees a zero-byte file.
    if data is None:
        return None
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(f"unexpected file result: {type(data).__name__}")
    return bytes(data) or None


class Connection:
    """DB connection holder that tolerates an unavailable SurrealDB.

    Never crashes the app on startup: it connects best-effort, reconnects in
    the background, and gates dispatch/persist/proxy behind a ready event.
    """

    def __init__(self, s: Settings, state: ServiceState):
        self.settings = s
        self.state = state
        self.db: AsyncSurrealConnection | None = None
        self.ready = asyncio.Event()
        self._announced_down = False

    async def connect(self) -> bool:
        self.ready.clear()
        try:
            self.db = await _connect_db(self.settings)
        except Exception as e:
            self.db = None
            self.state.set_db_status(False, str(e))
            if not self._announced_down:
                logger.error("SurrealDB unavailable: %s (retrying every 5s)", e)
                self._announced_down = True
            return False

        self.state.set_db_status(True, None)
        self.ready.set()
        self._announced_down = False
        logger.info(
            "Connected to SurrealDB (%s/%s)",
            self.settings.surrealdb_namespace,
            self.settings.surrealdb_name,
        )
        return True

    async def reconnect_loop(self) -> None:
        while True:
            if not self.ready.is_set():
                # Not connected (startup failure or a detected drop): retry.
                await self.connect()
                await asyncio.sleep(5)
                continue

            # Connected: light liveness probe so a mid-run drop self-heals.
            try:
                await self.db.version()
            except Exception:
                logger.warning("SurrealDB connection lost — reconnecting")
                await self.close()
                self.state.set_db_status(False, "connection lost")
            else:
                await asyncio.sleep(30)

    async def close(self) -> None:
        db = self.db
        self.db = None
        self.ready.clear()
        if db is not None:
            try:
                await db.close()
            except Exception as e:
                logger.debug("Error closing DB connection: %s", e)


class Broadcaster:
    """Fan-out for SSE clients. `close()` wakes every subscriber so streams end."""

    _SHUTDOWN = object()

    def __init__(self) -> None:
        self._subs: set[asyncio.Queue] = set()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def subscribe(self) -> asyncio.Queue:
        if self._closed:
            raise RuntimeError("broadcaster closed")
        q: asyncio.Queue = asyncio.Queue()
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self, data: dict) -> None:
        if self._closed:
            return
        for q in list(self._subs):
            q.put_nowait(data)

    def close(self) -> None:
        """Force-disconnect all SSE clients (used on app shutdown / reload)."""
        self._closed = True
        for q in list(self._subs):
            try:
                q.put_nowait(self._SHUTDOWN)
            except Exception:
                pass
        self._subs.clear()

    def reset(self) -> None:
        """Allow new subscribers after a previous close (new lifespan)."""
        self._closed = False
        self._subs.clear()


broadcaster = Broadcaster()


def _parse_record_id(raw: str) -> RecordID:
    try:
        return RecordID.parse(raw)
    except (InvalidRecordIdError, ValueError) as e:
        raise HTTPException(422, f"invalid record id: {raw}") from e


def _queue_info() -> dict:
    """Snapshot of the dispatch queue for /health and the SSE feed."""
    return {
        "queue_capacity": settings.queue_capacity,
        "queue_size": job_queue.qsize(),
    }


# The embedding_queue row for a marker_views record shares its record key: the
# add_embedding_task event does `UPSERT type::record("embedding_queue",
# record::id($after.id))`, so marker_views:abc maps to embedding_queue:abc. Its
# `status` column is the enum 'pending' | 'processing' | 'failed' | 'done' and
# `error_message` holds the failure reason. We mirror the in-memory task state
# onto that row so the queue reflects progress durably.
_QUEUE_STATUS = {
    "queued": "pending",
    "processing": "processing",
    "done": "done",
    "error": "failed",
}


def _queue_rid(rid: str) -> RecordID:
    """Map a marker_views record id to its embedding_queue record id."""
    return RecordID("embedding_queue", RecordID.parse(rid).id)


async def _set_queue_status(
    conn: Connection,
    rid: str,
    status: str,
    error: str | None = None,
    *,
    version: Any = None,
) -> None:
    """Reflect a task's status onto its embedding_queue row (best-effort).

    Never raises: a failed status write must not stall the dispatch/persist loop.
    On success, ``version`` (marker_views.version sampled at dispatch) is stored
    as ``version_snapshot`` so the app can detect out-of-sync embeddings.
    """
    try:
        queue_rid = _queue_rid(rid)
    except Exception as e:
        logger.warning("Cannot derive embedding_queue id for %s: %s", rid, e)
        return
    payload: dict[str, Any] = {
        "status": _QUEUE_STATUS.get(status, status),
        "error_message": error,
    }
    if version is not None:
        payload["version_snapshot"] = version
    try:
        await conn.ready.wait()
        # SDK 3.x: merge is a clause on update(), not a top-level method.
        await conn.db.update(queue_rid).merge(payload)
    except Exception as e:
        logger.warning("Failed to update embedding_queue status for %s: %s", rid, e)


async def _image_bytes(conn: Connection, raw: str | File) -> bytes | None:
    """Load raw image bytes from a record's image_url.

    Handles bucket file pointers (``File`` or ``\"bucket:/path\"``, read from the
    DB) and http(s) URLs.
    """
    if isinstance(raw, str) and raw.startswith(("http://", "https://")):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(raw)
                resp.raise_for_status()
                return resp.content
        except Exception as e:
            logger.warning("Failed to fetch image %s: %s", raw, e)
            return None

    ref = _as_file(raw)
    if ref is None:
        return None
    try:
        return await _read_file_bytes(conn.db, ref.bucket, ref.key)
    except Exception as e:
        logger.warning("Failed to read image %s: %s", raw, e)
        return None


async def _prepare_image(
    conn: Connection, raw: str | File | None
) -> Image.Image | None:
    """Load a record's image and downscale it (see prepare_image) for encoding."""
    if not raw:
        return None
    data = await _image_bytes(conn, raw)
    if not data:
        return None
    try:
        return await asyncio.to_thread(prepare_image, data, settings.image_max_pixels)
    except Exception as e:
        logger.warning("Failed to prepare image %s: %s", raw, e)
        return None


def _queue_idle() -> bool:
    """True when nothing is waiting, in-flight, or still marked pending."""
    if not job_queue.empty() or worker.jobs.qsize() > 0:
        return False
    return not any(
        t["status"] in ("queued", "processing") for t in state.all_tasks().values()
    )


def _note_finished(ok: bool) -> None:
    """Count a terminal task; log once when the whole queue drains."""
    global _batch_ok, _batch_err
    if ok:
        _batch_ok += 1
    else:
        _batch_err += 1
    if not _queue_idle():
        return
    logger.info("Queue done: %d ok, %d failed", _batch_ok, _batch_err)
    broadcaster.publish(
        {
            "status": "queue_done",
            "ok": _batch_ok,
            "failed": _batch_err,
        }
    )
    _batch_ok = 0
    _batch_err = 0


async def _dispatch(conn: Connection) -> None:
    while True:
        await conn.ready.wait()
        rid_obj = await job_queue.get()
        rid = str(rid_obj)
        state.set_task(rid, "processing")
        await _set_queue_status(conn, rid, "processing")
        broadcaster.publish({"record_id": rid, "status": "processing"})
        logger.info("Processing %s", rid)

        try:
            # SDK 3.x: select(RecordID) returns the record dict or None.
            rec = await conn.db.select(rid_obj)
        except Exception as e:
            logger.warning("Failed to select %s: %s", rid, e)
            state.set_task(rid, "error", str(e))
            await _set_queue_status(conn, rid, "error", str(e))
            state.bump(ok=False)
            broadcaster.publish({"record_id": rid, "status": "error", "error": str(e)})
            _note_finished(ok=False)
            continue

        if not rec or not isinstance(rec, dict):
            state.set_task(rid, "error", "record not found")
            await _set_queue_status(conn, rid, "error", "record not found")
            state.bump(ok=False)
            broadcaster.publish({"record_id": rid, "status": "error", "error": "record not found"})
            _note_finished(ok=False)
            continue

        text = (rec.get("generated_description") or "").strip() or None
        image = await _prepare_image(conn, rec.get("image_url"))

        if text is None and image is None:
            state.set_task(rid, "error", "no text or image to embed")
            await _set_queue_status(conn, rid, "error", "no text or image to embed")
            state.bump(ok=False)
            broadcaster.publish({"record_id": rid, "status": "error", "error": "no text or image to embed"})
            _note_finished(ok=False)
            continue

        worker.jobs.put(
            ModelJob(
                record_id=rid,
                text=text,
                image=image,
                version=rec.get("version"),
            )
        )


async def _persist(conn: Connection) -> None:
    while True:
        try:
            result = await asyncio.to_thread(worker.results.get, timeout=0.5)
        except Empty:
            continue
        rid = result.record_id

        if result.error:
            state.set_task(rid, "error", result.error)
            await _set_queue_status(conn, rid, "error", result.error)
            state.bump(ok=False)
            broadcaster.publish({"record_id": rid, "status": "error", "error": result.error})
            _note_finished(ok=False)
            continue

        try:
            await conn.ready.wait()
            # Only the vector goes on marker_views. Version snapshot lives on
            # embedding_queue (SCHEMAFULL has no embedding_version field).
            await conn.db.update(RecordID.parse(rid)).merge(
                {"embedding": result.embedding}
            )
        except Exception as e:
            logger.warning("Failed to persist %s: %s", rid, e)
            state.set_task(rid, "error", str(e))
            await _set_queue_status(conn, rid, "error", str(e))
            state.bump(ok=False)
            broadcaster.publish({"record_id": rid, "status": "error", "error": str(e)})
            _note_finished(ok=False)
            continue

        state.set_task(rid, "done")
        await _set_queue_status(conn, rid, "done", version=result.version)
        state.bump(ok=True)
        broadcaster.publish({"record_id": rid, "status": "done"})
        logger.info("Embedded and saved %s", rid)
        _note_finished(ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_runtime(settings)
    broadcaster.reset()
    conn = Connection(settings, state)
    app.state.conn = conn
    await conn.connect()

    worker.start()
    dispatch_task = asyncio.create_task(_dispatch(conn))
    persist_task = asyncio.create_task(_persist(conn))
    reconnect_task = asyncio.create_task(conn.reconnect_loop())
    logger.info("Service ready (model loading in background)")
    yield

    # Wake SSE generators first so uvicorn reload is not blocked waiting for
    # clients to drop /queue/stream. Generators poll closed + SHUTDOWN.
    broadcaster.close()
    await asyncio.sleep(0)  # let generators observe shutdown and exit
    worker.stop()
    for t in (dispatch_task, persist_task, reconnect_task):
        t.cancel()
    await asyncio.gather(dispatch_task, persist_task, reconnect_task, return_exceptions=True)
    await conn.close()


app = FastAPI(title="Embedding Service", lifespan=lifespan)


class GenerateRequest(BaseModel):
    items: list[str]


class QueryRequest(BaseModel):
    text: str | None = None
    image_url: str | None = None
    normalize: bool = False


class SearchRequest(BaseModel):
    """Cosine search over a registered target (default: markers via marker_views).

    Provide ``text``, ``image_url``, or both. ``top_k`` is the number of parent
    entities (markers) returned after view-level hits are summed per parent.
    ``filters`` keys depend on the target (markers: ``level``).
    """

    target: str = "markers"
    text: str | None = None
    image_url: str | None = None
    top_k: int = 10
    filters: dict[str, Any] | None = None


async def _encode_inputs(
    *,
    text: str | None,
    image_url: str | None,
    normalize: bool,
) -> list[float]:
    """Shared path for /encode and /search: load image, run query encode."""
    text = (text or "").strip() or None
    image = None
    if image_url:
        conn: Connection = app.state.conn
        image = await _prepare_image(conn, image_url)

    if text is None and image is None:
        raise HTTPException(400, "provide text and/or image_url")

    try:
        embedding = await asyncio.to_thread(
            worker.encode_query,
            text,
            image,
            settings.query_timeout,
            normalize,
        )
    except (TimeoutError, ModelNotReadyError) as e:
        raise HTTPException(503, str(e))
    except QueryError as e:
        raise HTTPException(502, str(e))

    if not embedding:
        raise HTTPException(502, "model returned an empty embedding")
    return embedding


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


@app.post("/encode")
async def encode(payload: QueryRequest):
    embedding = await _encode_inputs(
        text=payload.text,
        image_url=payload.image_url,
        normalize=payload.normalize,
    )
    return {"embedding": embedding}


@app.post("/search")
async def search(payload: SearchRequest):
    if payload.top_k < 1:
        raise HTTPException(400, "top_k must be >= 1")
    if payload.top_k > 100:
        raise HTTPException(400, "top_k must be <= 100")

    try:
        target = get_target(payload.target)
    except KeyError as e:
        raise HTTPException(400, str(e)) from e

    try:
        filter_clauses, filter_params = bind_filters(target, payload.filters)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    # Cosine search needs unit-length vectors; always normalize the query.
    query_vec = await _encode_inputs(
        text=payload.text,
        image_url=payload.image_url,
        normalize=True,
    )

    conn: Connection = app.state.conn
    if not conn.ready.is_set() or conn.db is None:
        raise HTTPException(503, "database unavailable")

    limit = candidate_limit(target, payload.top_k)
    sql = build_candidate_query(
        target, filter_clauses=filter_clauses, limit=limit
    )
    params: dict[str, Any] = {"query": query_vec, **filter_params}

    try:
        await conn.ready.wait()
        rows = await conn.db.query(sql, params)
    except Exception as e:
        logger.warning("Search query failed: %s", e)
        raise HTTPException(502, f"search query failed: {e}") from e

    if rows is None:
        rows = []
    if not isinstance(rows, list):
        raise HTTPException(502, f"unexpected search result: {type(rows).__name__}")

    # Keep only well-formed scored dicts.
    scored = [r for r in rows if isinstance(r, dict) and "score" in r]

    if target.parent_field:
        hits = aggregate_by_parent(
            scored, parent_field=target.parent_field, top_k=payload.top_k
        )
        hits = await hydrate_parents(conn.db, target, hits)
    else:
        # No parent collapse — return source rows directly.
        hits = [
            {
                "id": str(r.get("id")),
                "score": float(r["score"]),
            }
            for r in scored[: payload.top_k]
        ]

    return {
        "target": target.name,
        "top_k": payload.top_k,
        "count": len(hits),
        "results": hits,
    }


@app.get("/search/targets")
async def search_targets():
    """List registered search targets and their filter keys (for clients)."""
    return {
        name: {
            "source_table": t.source_table,
            "parent_table": t.parent_table,
            "parent_field": t.parent_field,
            "filters": [f.key for f in t.filters],
            "parent_fields": list(t.parent_fields),
        }
        for name, t in TARGETS.items()
    }


@app.get("/health")
async def health():
    snap = state.snapshot()
    return {
        "worker_status": snap["model"],
        "model_error": snap["model_error"],
        "device": settings.device,
        "modalities": settings.modalities,
        "uptime_seconds": snap["uptime_seconds"],
        "db_connected": snap["db_connected"],
        "db_error": snap["db_error"],
        "metrics": {
            "total_processed": snap["processed"],
            "total_errors": snap["errors"],
        },
        "pending": snap["pending"],
        **_queue_info(),
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
    conn: Connection = app.state.conn
    await conn.ready.wait()
    try:
        data = await _read_file_bytes(conn.db, bucket, path)
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
async def queue_stream(request: Request):
    async def gen():
        try:
            q = broadcaster.subscribe()
        except RuntimeError:
            yield _sse({"type": "shutdown"})
            return

        try:
            yield _sse({"type": "snapshot", "data": {**state.snapshot(), **_queue_info()}})
            while True:
                # Timed wait so we can notice app shutdown / client disconnect
                # without depending on the peer closing first (uvicorn reload).
                if broadcaster.closed or await request.is_disconnected():
                    yield _sse({"type": "shutdown"})
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # SSE comment heartbeat — keeps proxies from buffering and
                    # gives the loop another chance to see closed/disconnect.
                    yield ":\n\n"
                    continue
                if msg is Broadcaster._SHUTDOWN:
                    yield _sse({"type": "shutdown"})
                    break
                yield _sse({"type": "task", "data": msg})
        finally:
            broadcaster.unsubscribe(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
