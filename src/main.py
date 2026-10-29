import asyncio
import base64
import json
import logging
import mimetypes
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from queue import Empty
from typing import Any
from urllib.parse import unquote_to_bytes

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from surrealdb import AsyncSurreal, AsyncSurrealConnection, File, RecordID
from surrealdb.errors import (
    ConnectionUnavailableError,
    InvalidRecordIdError,
    NotAllowedError,
    TransportError,
)

from src.config import Settings, get_settings
from src.ids import parse_record_id, record_key
from src.preprocess import TransformMode
from src.search import (
    TARGETS,
    aggregate_by_parent,
    bind_filters,
    build_candidate_query,
    candidate_limit,
    filter_top_views,
    get_target,
    hydrate_parents,
    unwrap_query_result,
)
from src.state import ServiceState
from src.utils import check_deps, setup_logging, setup_process_prio
from src.worker import ModelJob, ModelNotReadyError, ModelWorker, QueryError

setup_logging(logging.INFO)
check_deps()

logger = logging.getLogger(__name__)

settings = get_settings()
state = ServiceState()
worker = ModelWorker(state)


# VPR async jobs. Free-form client ids stay in-memory only; when ``task_id``
# is a marker_views record the result is upserted onto ``embeddings`` and the
# matching ``embedding_queue`` row is mirrored.
@dataclass
class VprQueueItem:
    task_id: str
    # None → load image_url from the marker_views row (DB-backed jobs only).
    image_url: str | File | None = None
    transform: str | None = None


vpr_queue: asyncio.Queue[VprQueueItem] = asyncio.Queue(
    maxsize=settings.queue_capacity
)

# Counts for the current busy wave; reset when the queue returns to idle.
_batch_ok = 0
_batch_err = 0


def _setup_runtime(s: Settings) -> None:
    os.environ.setdefault("HF_HOME", s.hf_home)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "warning")
    if s.device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
        if sys.platform == "win32":
            setup_process_prio()


def _image_ref_log(raw: Any) -> str:
    """Short, safe label for logs — never dump data-URLs / base64 bodies."""
    if raw is None:
        return "none"
    if isinstance(raw, File):
        return f"file:{raw.bucket}:{raw.key}"
    if not isinstance(raw, str):
        return type(raw).__name__
    s = raw.strip()
    if not s:
        return "empty"
    # data:[<mime>][;base64],<payload>
    if s.startswith("data:"):
        header, sep, payload = s.partition(",")
        return f"data_url({header}; len={len(payload) if sep else 0})"
    if s.startswith(("http://", "https://")):
        return s if len(s) <= 120 else s[:117] + "..."
    ref = _as_file(s)
    if ref is not None:
        return f"file:{ref.bucket}:{ref.key}"
    # Long opaque bodies (bare base64) — length only, never the payload.
    if len(s) > 64:
        return f"base64(len={len(s)})"
    return s if len(s) <= 80 else s[:77] + "..."


def _as_file(raw: str | File | None) -> File | None:
    """Coerce a record field into a SurrealDB ``File``, or ``None`` if it isn't one.

    Accepts a native ``File`` (decoded by the SDK from CBOR tag 55) or a string
    form ``bucket:/key`` / ``bucket://key`` (and the optional ``f"..."`` wrapper).

    Rejects http(s), ``data:`` URLs, and raw base64 — those are not bucket refs.
    Path after the bucket must start with ``/`` or ``//`` so ``data:image/...``
    is never mistaken for bucket ``data``.
    """
    if raw is None:
        return None
    if isinstance(raw, File):
        return raw
    if not isinstance(raw, str):
        return None

    s = raw.strip()
    if not s:
        return None
    low = s.lower()
    if low.startswith(("http://", "https://", "data:")):
        return None

    if len(s) >= 3 and s[0] == "f" and s[1] in "'\"" and s[-1] == s[1]:
        s = s[2:-1].strip()

    bucket, sep, path = s.partition(":")
    if not sep or not bucket or not path:
        return None
    # Canonical Surreal file form is bucket:/key or bucket://key — require slash.
    if not path.startswith("/"):
        return None
    # "bucket://key" and "bucket:/key" both collapse to key "/key".
    while path.startswith("//"):
        path = path[1:]
    if path == "/":
        return None
    if not _safe_path_segment(bucket) or "/" in bucket:
        return None
    key_body = path.lstrip("/")
    if not key_body or not _safe_path_segment(key_body):
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


async def _signin(db: AsyncSurrealConnection, s: Settings) -> None:
    """Authenticate ``db`` at the configured auth level, then select NS/DB."""
    await db.signin(s.signin_vars())
    await db.use(s.surrealdb_namespace, s.surrealdb_name)


async def _connect_db(s: Settings) -> AsyncSurrealConnection:
    db = AsyncSurreal(s.surrealdb_host)
    try:
        await _signin(db, s)
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


def _is_token_expired(exc: BaseException) -> bool:
    """True when Surreal rejected the session token — re-signin fixes it."""
    if isinstance(exc, NotAllowedError) and (
        exc.is_token_expired or exc.is_invalid_auth
    ):
        return True
    msg = str(exc).lower()
    return "token expired" in msg or "invalid auth" in msg


def _is_connection_dead(exc: BaseException) -> bool:
    """True when the transport is gone and a full reconnect is needed."""
    return isinstance(exc, (ConnectionUnavailableError, TransportError))


class Connection:
    """DB connection holder that tolerates an unavailable SurrealDB.

    Never crashes the app on startup: it connects best-effort, reconnects in
    the background, and gates dispatch/persist/proxy behind a ready event.

    Token expiry (e.g. after a long CPU embed) is handled by ``run()``:
    re-signin on the same socket, then retry. Transport death still does a
    full reconnect.
    """

    def __init__(self, s: Settings, state: ServiceState):
        self.settings = s
        self.state = state
        self.db: AsyncSurrealConnection | None = None
        self.ready = asyncio.Event()
        self._announced_down = False
        self._auth_lock = asyncio.Lock()

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
            "Connected to SurrealDB (%s/%s, auth=%s)",
            self.settings.surrealdb_namespace,
            self.settings.surrealdb_name,
            self.settings.surrealdb_auth_level,
        )
        return True

    async def reauth(self) -> bool:
        """Fresh sign-in on the existing connection (token expired)."""
        async with self._auth_lock:
            db = self.db
            if db is None:
                logger.warning("SurrealDB re-auth: no connection — reconnecting")
                return await self.connect()
            s = self.settings
            try:
                logger.warning(
                    "SurrealDB token expired — re-signing in (level=%s)",
                    s.surrealdb_auth_level,
                )
                await _signin(db, s)
                self.state.set_db_status(True, None)
                self.ready.set()
                logger.info(
                    "SurrealDB re-authenticated (level=%s)", s.surrealdb_auth_level
                )
                return True
            except Exception as e:
                logger.error("SurrealDB re-auth failed: %s — full reconnect", e)
                await self.close()
                self.state.set_db_status(False, str(e))
                return await self.connect()

    async def force_reconnect(self, reason: str = "reconnect") -> bool:
        """Drop the socket and open a new session."""
        async with self._auth_lock:
            logger.warning("SurrealDB %s — reconnecting", reason)
            await self.close()
            self.state.set_db_status(False, reason)
            return await self.connect()

    async def run(self, op, *, retries: int = 2):
        """Run ``await op(db)``; re-auth on token expiry, reconnect if socket dies.

        ``op`` is an async callable that receives the live connection.
        """
        last: BaseException | None = None
        attempts = max(1, retries + 1)
        for i in range(attempts):
            await self.ready.wait()
            db = self.db
            if db is None:
                ok = await self.force_reconnect("connection missing")
                if not ok:
                    last = RuntimeError("SurrealDB unavailable")
                    continue
                db = self.db
                if db is None:
                    last = RuntimeError("SurrealDB unavailable")
                    continue
            try:
                return await op(db)
            except Exception as e:
                last = e
                if i + 1 >= attempts:
                    raise
                if _is_token_expired(e):
                    ok = await self.reauth()
                    if ok:
                        continue
                    raise
                if _is_connection_dead(e):
                    ok = await self.force_reconnect(f"connection error: {e}")
                    if ok:
                        continue
                    raise
                raise
        assert last is not None
        raise last

    async def reconnect_loop(self) -> None:
        while True:
            if not self.ready.is_set():
                # Not connected (startup failure or a detected drop): retry.
                await self.connect()
                await asyncio.sleep(5)
                continue

            # Connected: light liveness probe so a mid-run drop self-heals.
            try:
                db = self.db
                if db is None:
                    raise ConnectionUnavailableError("no db")
                await db.version()
            except Exception as e:
                if _is_token_expired(e):
                    await self.reauth()
                else:
                    await self.force_reconnect("connection lost")
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
    """Parse a client record id, stripping SurrealQL ``⟨⟩`` / backtick escapes."""
    try:
        return parse_record_id(raw)
    except (InvalidRecordIdError, ValueError, TypeError) as e:
        raise HTTPException(422, f"invalid record id: {raw}") from e


def _queue_info() -> dict:
    """Snapshot of the dispatch queue for /health and the SSE feed."""
    return {
        "queue_capacity": settings.queue_capacity,
        "queue_size": vpr_queue.qsize(),
    }


# The embedding_queue / embeddings rows for a marker_views record share its
# record key: the app's add_embedding_task event does
# `UPSERT type::record("embedding_queue", record::id($after.id))`, so
# marker_views:abc maps to embedding_queue:abc and embeddings:abc.
# Queue `status` is 'pending' | 'processing' | 'failed' | 'done'; we mirror
# in-memory task state onto that row. Vectors land on ``embeddings.embedding``
# — never on marker_views. Out-of-sync:
#   marker_views.image_hash != embedding_queue.hash_snapshot
_QUEUE_STATUS = {
    "queued": "pending",
    "processing": "processing",
    "done": "done",
    "error": "failed",
}


def _queue_rid(rid: str) -> RecordID:
    """Map a marker_views record id to its embedding_queue record id."""
    return RecordID("embedding_queue", parse_record_id(rid).id)


def _embeddings_rid(rid: str) -> RecordID:
    """Map a marker_views record id to its embeddings record id."""
    return RecordID("embeddings", parse_record_id(rid).id)


def _is_marker_view_id(rid: str) -> bool:
    """True when ``rid`` parses as a ``marker_views`` record."""
    try:
        return parse_record_id(rid).table_name == "marker_views"
    except Exception:
        return False


def _marker_views_key_from_queue_row(row: dict[str, Any]) -> str | None:
    """Resolve marker_views:… key from an embedding_queue SELECT row."""
    data = row.get("data")
    if data is not None:
        try:
            if isinstance(data, RecordID):
                return record_key(data)
            return record_key(str(data))
        except Exception:
            pass
    # Fall back to shared key: embedding_queue:X → marker_views:X
    qid = row.get("id")
    if qid is None:
        return None
    try:
        if isinstance(qid, RecordID):
            return record_key(RecordID("marker_views", qid.id))
        parsed = parse_record_id(str(qid))
        return record_key(RecordID("marker_views", parsed.id))
    except Exception:
        return None


async def _set_queue_status(
    conn: Connection,
    rid: str,
    status: str,
    error: str | None = None,
    *,
    image_hash: str | None = None,
) -> None:
    """Reflect a task's status onto its embedding_queue row.

    Never raises: a failed status write must not stall the dispatch loop.
    Terminal writes (done/failed) retry harder, including fresh sign-in after
    token expiry — otherwise the row stays stuck on ``processing`` while the
    UI already shows failed/done from the in-memory SSE feed.
    On success, ``image_hash`` (marker_views.image_hash sampled at dispatch) is
    stored as ``hash_snapshot`` so the app can detect out-of-sync embeddings.
    """
    try:
        queue_rid = _queue_rid(rid)
    except Exception as e:
        logger.warning("Cannot derive embedding_queue id for %s: %s", rid, e)
        return
    mapped = _QUEUE_STATUS.get(status, status)
    payload: dict[str, Any] = {
        "status": mapped,
        "error_message": error,
    }
    if image_hash is not None:
        payload["hash_snapshot"] = image_hash

    # processing: one reconnect cycle is enough
    # done/failed: more attempts — this is what unsticks the queue UI/DB
    retries = 4 if mapped in ("done", "failed") else 2

    async def _write(db: AsyncSurrealConnection) -> None:
        # SDK 3.x: merge is a clause on update(), not a top-level method.
        await db.update(queue_rid).merge(payload)

    try:
        await conn.run(_write, retries=retries)
    except Exception as e:
        logger.error(
            "Failed to update embedding_queue %s → %s after retries: %s",
            rid,
            mapped,
            e,
        )


async def _write_embedding(
    conn: Connection,
    marker_views_rid: str,
    embedding: list[float],
    *,
    model: str,
) -> None:
    """UPSERT the vector onto ``embeddings:{same_id}`` for a marker_views row.

    Writes ``embedding``, ``model``, and ``data`` (REFERENCE → marker_views).
    Raises on failure so the caller can mark the task failed.
    """
    mv = parse_record_id(marker_views_rid)
    emb_rid = _embeddings_rid(marker_views_rid)
    payload = {
        "embedding": embedding,
        "model": model,
        "data": mv,
    }

    async def _upsert(db: AsyncSurrealConnection) -> None:
        # upsert+merge: create row if missing; only touch the named fields.
        await db.upsert(emb_rid).merge(payload)

    await conn.run(_upsert, retries=3)


def _decode_data_url(raw: str) -> bytes | None:
    """Decode ``data:[mime][;base64],payload`` — split on first comma only."""
    if not raw.startswith("data:") or "," not in raw:
        return None
    header, payload = raw.split(",", 1)
    try:
        if ";base64" in header:
            return base64.b64decode(payload, validate=False)
        return unquote_to_bytes(payload)
    except Exception as e:
        logger.warning("Failed to decode data URL %s: %s", _image_ref_log(raw), e)
        return None


def _decode_raw_base64(raw: str) -> bytes | None:
    """Decode a bare base64 body (no ``data:`` header)."""
    try:
        data = base64.b64decode("".join(raw.split()), validate=False)
    except Exception as e:
        logger.warning(
            "Failed to decode base64 image %s: %s", _image_ref_log(raw), e
        )
        return None
    if not data or len(data) < 24:
        return None
    return data


def _decode_base64_image(raw: str) -> bytes | None:
    """App ``image_url``: ``data:`` URL or bare base64 → bytes."""
    s = raw.strip()
    if not s:
        return None
    if s.startswith("data:"):
        return _decode_data_url(s)
    return _decode_raw_base64(s)


async def _image_bytes(conn: Connection, raw: str | File) -> bytes | None:
    """Load image bytes for VPR encode.

    Order is fixed and boring (prefix checks only, no regex):
      1. ``data:…`` or bare base64
      2. ``http://`` / ``https://``
      3. Surreal ``File`` / ``bucket:/key``
    """
    if isinstance(raw, File):
        try:
            return await _read_file_bytes(conn.db, raw.bucket, raw.key)
        except Exception as e:
            logger.warning("Failed to read image %s: %s", _image_ref_log(raw), e)
            return None

    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None

    # 1) app base64 — MUST be before _as_file (data: ≠ bucket "data")
    if s.startswith("data:"):
        return await asyncio.to_thread(_decode_data_url, s)

    if s.startswith("http://") or s.startswith("https://"):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(s)
                resp.raise_for_status()
                return resp.content
        except Exception as e:
            logger.warning("Failed to fetch image %s: %s", _image_ref_log(s), e)
            return None

    ref = _as_file(s)
    if ref is not None:
        try:
            return await _read_file_bytes(conn.db, ref.bucket, ref.key)
        except Exception as e:
            logger.warning("Failed to read image %s: %s", _image_ref_log(s), e)
            return None

    # bare base64 (no data: prefix) — last, only if not a file pointer
    return await asyncio.to_thread(_decode_raw_base64, s)


def _parse_transform(raw: str | None) -> str | None:
    """Validate optional VPR transform override; None → settings default."""
    if raw is None or not str(raw).strip():
        return None
    try:
        return TransformMode(str(raw).strip().lower()).value
    except ValueError as e:
        raise HTTPException(
            422,
            f"invalid transform {raw!r}; expected "
            f"{', '.join(m.value for m in TransformMode)}",
        ) from e


def _queue_idle() -> bool:
    """True when nothing is waiting, in-flight, or still marked pending."""
    if not vpr_queue.empty() or worker.jobs.qsize() > 0:
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


def _task_in_flight(rid: str) -> bool:
    task = state.get_task(rid)
    return bool(task and task.get("status") in ("queued", "processing"))


async def _load_vpr_image(
    conn: Connection, raw: str | File | None
) -> bytes | None:
    """Load image bytes for VPR. Geometry + ImageNet norm run on the worker."""
    if not raw:
        return None
    ref = _image_ref_log(raw)
    t0 = time.perf_counter()
    data = await _image_bytes(conn, raw)
    if not data:
        logger.info(
            "VPR image skip %s: no bytes (%.0fms)",
            ref,
            (time.perf_counter() - t0) * 1000,
        )
        return None
    logger.info(
        "VPR image loaded %s bytes=%d (%.0fms)",
        ref,
        len(data),
        (time.perf_counter() - t0) * 1000,
    )
    return data


async def _enqueue_vpr(
    tid: str,
    *,
    image_url: str | File | None = None,
    transform: str | None = None,
) -> bool:
    """Put a VPR task on the in-memory queue if not already in flight.

    Returns False when skipped (already queued/processing).
    """
    if _task_in_flight(tid):
        return False
    state.set_task(tid, "queued", kind="vpr", clear_embedding=True)
    await vpr_queue.put(
        VprQueueItem(task_id=tid, image_url=image_url, transform=transform)
    )
    broadcaster.publish({"record_id": tid, "status": "queued", "kind": "vpr"})
    return True


async def _vpr_dispatch(conn: Connection) -> None:
    """Drain vpr_queue → worker.jobs.

    marker_views-backed jobs also mirror ``embedding_queue`` and sample
    ``image_hash`` for the post-embed snapshot. Free-form client ids skip DB.
    """
    while True:
        item = await vpr_queue.get()
        rid = item.task_id
        persist = _is_marker_view_id(rid)
        state.set_task(rid, "processing", kind="vpr")
        broadcaster.publish(
            {"record_id": rid, "status": "processing", "kind": "vpr"}
        )
        logger.info("VPR processing %s persist=%s", rid, persist)

        if persist:
            await _set_queue_status(conn, rid, "processing")

        image_hash: str | None = None
        image_src: str | File | None = item.image_url

        try:
            # marker_views row: always touch DB (image_hash + optional image load).
            if persist:
                await conn.ready.wait()
                rid_obj = parse_record_id(rid)

                async def _select(db: AsyncSurrealConnection):
                    return await db.select(rid_obj)

                rec = await conn.run(_select, retries=2)
                if isinstance(rec, list):
                    rec = rec[0] if rec else None
                if not isinstance(rec, dict):
                    raise ValueError("record not found")
                raw_hash = rec.get("image_hash")
                image_hash = (
                    str(raw_hash).strip()
                    if raw_hash is not None and str(raw_hash).strip()
                    else None
                )
                if image_src is None:
                    image_src = rec.get("image_url")
            elif isinstance(image_src, File) or _as_file(
                image_src if isinstance(image_src, str) else None
            ):
                # File pointers need DB; http/data URLs do not.
                await conn.ready.wait()

            image = await _load_vpr_image(conn, image_src)
        except Exception as e:
            logger.warning("VPR load failed %s: %s", rid, e)
            state.set_task(rid, "error", str(e), kind="vpr")
            if persist:
                await _set_queue_status(conn, rid, "error", str(e))
            state.bump(ok=False)
            broadcaster.publish(
                {
                    "record_id": rid,
                    "status": "error",
                    "error": str(e),
                    "kind": "vpr",
                }
            )
            _note_finished(ok=False)
            continue

        if image is None:
            err = "no image to encode"
            state.set_task(rid, "error", err, kind="vpr")
            if persist:
                await _set_queue_status(conn, rid, "error", err)
            state.bump(ok=False)
            broadcaster.publish(
                {
                    "record_id": rid,
                    "status": "error",
                    "error": err,
                    "kind": "vpr",
                }
            )
            _note_finished(ok=False)
            continue

        worker.jobs.put(
            ModelJob(
                record_id=rid,
                image=image,
                image_hash=image_hash,
                transform=item.transform,
            )
        )


async def _vpr_collect(conn: Connection) -> None:
    """Receive VPR ModelResults → SSE + optional embeddings/queue writes.

    Free-form ids: descriptor stays in ServiceState only.
    marker_views ids: UPSERT ``embeddings.embedding`` and mark the queue row
    done (with hash_snapshot). Descriptor still kept in-memory so
    ``GET /tasks/{id}`` works either way.
    """
    while True:
        try:
            result = await asyncio.to_thread(worker.results.get, timeout=0.5)
        except Empty:
            continue
        rid = result.record_id
        persist = _is_marker_view_id(rid)

        if result.error:
            state.set_task(rid, "error", result.error, kind="vpr")
            if persist:
                await _set_queue_status(conn, rid, "error", result.error)
            state.bump(ok=False)
            broadcaster.publish(
                {
                    "record_id": rid,
                    "status": "error",
                    "error": result.error,
                    "kind": "vpr",
                }
            )
            _note_finished(ok=False)
            continue

        emb = result.embedding or []

        if persist:
            try:
                await _write_embedding(
                    conn,
                    rid,
                    emb,
                    model=settings.megaloc_model_label,
                )
            except Exception as e:
                logger.warning("Failed to persist VPR %s: %s", rid, e)
                state.set_task(rid, "error", str(e), kind="vpr")
                await _set_queue_status(conn, rid, "error", str(e))
                state.bump(ok=False)
                broadcaster.publish(
                    {
                        "record_id": rid,
                        "status": "error",
                        "error": str(e),
                        "kind": "vpr",
                    }
                )
                _note_finished(ok=False)
                continue
            await _set_queue_status(
                conn, rid, "done", image_hash=result.image_hash
            )

        state.set_task(rid, "done", kind="vpr", embedding=emb)
        state.bump(ok=True)
        broadcaster.publish(
            {
                "record_id": rid,
                "status": "done",
                "kind": "vpr",
                "dim": len(emb),
            }
        )
        logger.info(
            "VPR done %s dim=%d persist=%s hash=%s",
            rid,
            len(emb),
            persist,
            (result.image_hash or "")[:12] or "-",
        )
        _note_finished(ok=True)


async def _poll_pending_queue(conn: Connection) -> None:
    """Pull embedding_queue status=pending rows into the local VPR queue.

    The app's ``add_embedding_task`` event writes pending rows; this loop drains
    them without requiring an external POST /generate. Interval 0 disables.
    """
    interval = float(settings.queue_poll_interval or 0.0)
    if interval <= 0:
        logger.info("Queue poll disabled (QUEUE_POLL_INTERVAL<=0)")
        return

    logger.info("Queue poll every %.1fs for embedding_queue status=pending", interval)
    sql = (
        "SELECT id, data, hash_snapshot FROM embedding_queue "
        "WHERE status = 'pending' LIMIT 50"
    )
    while True:
        try:
            await conn.ready.wait()

            async def _fetch(db: AsyncSurrealConnection):
                return await db.query(sql)

            raw = await conn.run(_fetch, retries=1)
            rows = unwrap_query_result(raw)
            accepted = 0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                tid = _marker_views_key_from_queue_row(row)
                if not tid:
                    continue
                if await _enqueue_vpr(tid):
                    accepted += 1
            if accepted:
                logger.info("Queue poll enqueued %d pending job(s)", accepted)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Queue poll failed: %s", e)

        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_runtime(settings)
    broadcaster.reset()
    conn = Connection(settings, state)
    app.state.conn = conn
    await conn.connect()

    worker.start()
    tasks = [
        asyncio.create_task(conn.reconnect_loop()),
        asyncio.create_task(_vpr_dispatch(conn)),
        asyncio.create_task(_vpr_collect(conn)),
        asyncio.create_task(_poll_pending_queue(conn)),
    ]
    logger.info(
        "Service ready backend=megaloc device=%s ns=%s db=%s "
        "(model loading in background)",
        settings.device,
        settings.surrealdb_namespace,
        settings.surrealdb_name,
    )
    yield

    # Wake SSE generators first so uvicorn reload is not blocked waiting for
    # clients to drop /queue/stream. Generators poll closed + SHUTDOWN.
    broadcaster.close()
    await asyncio.sleep(0)  # let generators observe shutdown and exit
    worker.stop()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await conn.close()


app = FastAPI(title="VPR Embedding Service", lifespan=lifespan)


class VprEncodeRequest(BaseModel):
    """Sync MegaLoc descriptor for one image."""

    image_url: str
    # squash | letterbox | center_crop — omit to use settings.vpr_transform
    transform: str | None = None


class EncodeRequest(BaseModel):
    """Root ``/encode`` body — image required."""

    image_url: str | None = None
    transform: str | None = None


class GenerateItem(BaseModel):
    """One async generate task (object form).

    ``id`` is the SSE /tasks key. Free-form ids need ``image_url`` and stay
    in-memory only. ``marker_views:...`` ids UPSERT ``embeddings.embedding``
    and mirror ``embedding_queue``; ``image_url`` is optional then (loaded from
    the record when omitted).
    """

    id: str
    image_url: str | None = None


class GenerateRequest(BaseModel):
    """Queue VPR embeds. Item form may be a bare record-id string or an object."""

    items: list[str | GenerateItem]
    transform: str | None = None


class SearchRequest(BaseModel):
    """Cosine VPR search over a registered target.

    Image-only query vs ``embeddings.embedding``. ``top_k`` is the number of
    **views** kept after cosine ranking (then collapsed onto parents).
    Parent score is ``max(sim) + α·log(1+n_views)`` (α = ``VIEW_COUNT_ALPHA``).
    ``filters`` keys depend on the target (markers*: ``level``).

    ``min_score`` is **soft-deprecated**: optional server-side view noise floor
    only (default ``0`` = off for typical scores). Prefer wide ``top_k`` +
    level filters here and threshold scores on the client. Do not treat it as
    relevance/confidence.
    """

    target: str = "markers"
    image_url: str | None = None
    top_k: int = 10
    # Soft-deprecated via description + OpenAPI `deprecated` (not
    # Field(deprecated=True) — that emits DeprecationWarning on every request).
    min_score: float = Field(
        default=0.0,
        description=(
            "SOFT-DEPRECATED optional view cosine noise floor before top_k. "
            "Default 0 (off for typical VPR scores). Prefer top_k + client-side "
            "score filtering; not a relevance/confidence gate. Retained for a "
            "possible server noise floor when the gallery is large."
        ),
        json_schema_extra={"deprecated": True},
    )
    filters: dict[str, Any] | None = None


async def _encode_vpr(
    *,
    image_url: str,
    transform: str | None,
) -> list[float]:
    """Load one image and run MegaLoc on the worker thread."""
    tfm = _parse_transform(transform)
    conn: Connection = app.state.conn
    image = await _load_vpr_image(conn, image_url)
    if image is None:
        raise HTTPException(400, "provide a valid image_url")

    timeout = settings.query_timeout
    timeout_label = "none" if timeout is None or timeout <= 0 else f"{timeout}s"
    logger.info(
        "VPR encode start image=%s transform=%s size=%d timeout=%s",
        _image_ref_log(image_url),
        tfm or settings.vpr_transform,
        settings.vpr_image_size,
        timeout_label,
    )
    t0 = time.perf_counter()
    try:
        embedding = await asyncio.to_thread(
            worker.encode_vpr,
            image,
            timeout,
            transform=tfm,
        )
    except (TimeoutError, ModelNotReadyError) as e:
        logger.warning(
            "VPR encode failed after %.0fms: %s",
            (time.perf_counter() - t0) * 1000,
            e,
        )
        raise HTTPException(503, str(e))
    except QueryError as e:
        logger.warning(
            "VPR encode error after %.0fms: %s",
            (time.perf_counter() - t0) * 1000,
            e,
        )
        raise HTTPException(502, str(e))

    if not embedding:
        raise HTTPException(502, "model returned an empty descriptor")
    logger.info(
        "VPR encode done dim=%d (%.0fms)",
        len(embedding),
        (time.perf_counter() - t0) * 1000,
    )
    return embedding


def _normalize_generate_items(
    items: list[str | GenerateItem],
) -> list[tuple[str, str | None]]:
    """Flatten bare strings / objects into ``(id, image_url | None)``."""
    out: list[tuple[str, str | None]] = []
    for raw in items:
        if isinstance(raw, str):
            tid = raw.strip()
            url = None
        else:
            tid = (raw.id or "").strip()
            url = (raw.image_url or "").strip() or None
        if not tid:
            raise HTTPException(422, "each item needs a non-empty id")
        out.append((tid, url))
    return out


async def _queue_vpr_generate(
    items: list[str | GenerateItem],
    transform: str | None,
) -> dict:
    """Accept VPR tasks. marker_views ids also write embeddings + queue."""
    pairs = _normalize_generate_items(items)
    tfm = _parse_transform(transform)
    accepted: list[str] = []
    for tid, url in pairs:
        if not url and not _is_marker_view_id(tid):
            raise HTTPException(
                422,
                f"item {tid!r} needs image_url "
                f"(or a marker_views:… id to load the record image)",
            )
        # Free-form ids must stay unique keys; marker_views go through parse.
        if _is_marker_view_id(tid):
            tid = record_key(_parse_record_id(tid))
        if await _enqueue_vpr(tid, image_url=url, transform=tfm):
            accepted.append(tid)
        else:
            accepted.append(tid)  # already in flight — still report accepted
    return {
        "accepted": accepted,
        "count": len(accepted),
        "backend": "megaloc",
        "transform": tfm or settings.vpr_transform,
        "image_size": settings.vpr_image_size,
    }


async def _vpr_encode_response(payload: VprEncodeRequest) -> dict:
    if not (payload.image_url or "").strip():
        raise HTTPException(400, "image_url is required")
    embedding = await _encode_vpr(
        image_url=payload.image_url.strip(),
        transform=payload.transform,
    )
    return {
        "embedding": embedding,
        "dim": len(embedding),
        "backend": "megaloc",
        "transform": _parse_transform(payload.transform) or settings.vpr_transform,
        "image_size": settings.vpr_image_size,
    }


@app.post("/generate", status_code=202)
async def generate(payload: GenerateRequest):
    """Queue MegaLoc embeds."""
    if not payload.items:
        raise HTTPException(400, "items must not be empty")
    return await _queue_vpr_generate(payload.items, payload.transform)


@app.post("/vpr/generate", status_code=202)
async def vpr_generate(payload: GenerateRequest):
    """Alias of ``POST /generate``."""
    if not payload.items:
        raise HTTPException(400, "items must not be empty")
    return await _queue_vpr_generate(payload.items, payload.transform)


@app.post("/encode")
async def encode(payload: EncodeRequest):
    """Sync MegaLoc descriptor."""
    return await _vpr_encode_response(
        VprEncodeRequest(
            image_url=payload.image_url or "",
            transform=payload.transform,
        )
    )


@app.post("/vpr/encode")
async def vpr_encode(payload: VprEncodeRequest):
    """Alias of ``POST /encode``."""
    return await _vpr_encode_response(payload)


@app.post("/search")
async def search(payload: SearchRequest):
    has_image = bool(payload.image_url and str(payload.image_url).strip())
    image_ref = _image_ref_log(payload.image_url) if has_image else "none"

    logger.info(
        "Search request target=%s filters=%s image=%s top_k=%s min_score=%s",
        payload.target,
        payload.filters or {},
        image_ref,
        payload.top_k,
        payload.min_score,
    )

    if payload.top_k < 1:
        raise HTTPException(400, "top_k must be >= 1")
    if payload.top_k > 100:
        raise HTTPException(400, "top_k must be <= 100")
    if not -1.0 <= payload.min_score <= 1.0:
        raise HTTPException(400, "min_score must be between -1 and 1")
    if payload.min_score > 0.0:
        logger.info(
            "Search min_score=%.4f is soft-deprecated; prefer client-side score "
            "filtering (noise floor only)",
            payload.min_score,
        )
    if not has_image:
        raise HTTPException(400, "image_url required for VPR search")

    try:
        target = get_target(payload.target)
    except KeyError as e:
        raise HTTPException(400, str(e)) from e

    try:
        filter_clauses, filter_params = bind_filters(target, payload.filters)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    query_vec = await _encode_vpr(
        image_url=payload.image_url,  # type: ignore[arg-type]
        transform=None,
    )
    logger.info(
        "Search query encoded dim=%d target=%s field=%s",
        len(query_vec),
        target.name,
        target.embedding_field,
    )

    conn: Connection = app.state.conn
    if not conn.ready.is_set() or conn.db is None:
        raise HTTPException(503, "database unavailable")

    limit = candidate_limit(target, payload.top_k)
    sql = build_candidate_query(
        target,
        filter_clauses=filter_clauses,
        limit=limit,
        min_score=payload.min_score,
    )
    params: dict[str, Any] = {
        "query": query_vec,
        "min_score": payload.min_score,
        **filter_params,
    }

    try:
        await conn.ready.wait()
        raw = await conn.db.query(sql, params)
    except Exception as e:
        logger.warning(
            "Search query failed target=%s filters=%s image=%s: %s",
            target.name,
            payload.filters or {},
            image_ref,
            e,
        )
        raise HTTPException(502, f"search query failed: {e}") from e

    # SDK 3 returns [[rows...]] per statement — unwrap before scoring.
    rows = unwrap_query_result(raw)
    # View-level gate: cutoff + top_k apply to views, not aggregated parents.
    scored = filter_top_views(
        rows, top_k=payload.top_k, min_score=payload.min_score
    )

    if target.parent_field:
        hits = aggregate_by_parent(scored, parent_field=target.parent_field)
        hits = await hydrate_parents(conn.db, target, hits)
    else:
        hits = [
            {
                "id": str(r.get("id")),
                "score": float(r["score"]),
            }
            for r in scored
        ]

    top = ", ".join(
        f"{h.get('id')}={h.get('score'):.4f}"
        if isinstance(h.get("score"), (int, float))
        else str(h.get("id"))
        for h in hits[:5]
    ) or "(none)"
    logger.info(
        "Search done target=%s filters=%s min_score=%s views=%d hits=%d top=[%s]",
        target.name,
        payload.filters or {},
        payload.min_score,
        len(scored),
        len(hits),
        top,
    )

    body: dict[str, Any] = {
        "target": target.name,
        "top_k": payload.top_k,
        "min_score": payload.min_score,
        "count": len(hits),
        "results": hits,
    }
    # Soft-deprecation signal for clients still tuning server-side cutoffs.
    if payload.min_score > 0.0:
        body["warnings"] = [
            "min_score is soft-deprecated: optional noise floor only; "
            "prefer top_k + client-side score filtering"
        ]
    return body


@app.get("/search/targets")
async def search_targets():
    """List registered search targets and their filter keys (for clients)."""
    return {
        name: {
            "source_table": t.source_table,
            "embedding_field": t.embedding_field,
            "query_backend": "megaloc",
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
        "backend": "megaloc",
        "device": settings.device,
        "uptime_seconds": snap["uptime_seconds"],
        "db_connected": snap["db_connected"],
        "db_error": snap["db_error"],
        "db_namespace": settings.surrealdb_namespace,
        "db_database": settings.surrealdb_name,
        "metrics": {
            "total_processed": snap["processed"],
            "total_errors": snap["errors"],
        },
        "pending": snap["pending"],
        "queue_capacity": settings.queue_capacity,
        "queue_size": vpr_queue.qsize(),
        "queue_poll_interval": settings.queue_poll_interval,
        "megaloc_repo": settings.megaloc_repo,
        "vpr_image_size": settings.vpr_image_size,
        "vpr_transform": settings.vpr_transform,
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
            yield _sse(
                {
                    "type": "snapshot",
                    "data": {**state.snapshot(), **_queue_info()},
                }
            )
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
