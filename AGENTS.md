# Embeddings Service

Internal service that generates text and image embeddings for SurrealDB
`marker_views` records using `jinaai/jina-embeddings-v5-omni-nano-retrieval`.

The app calls an HTTP API; the service reads a record, writes back
`text_embedding` and `image_embedding` via `MERGE`, and exposes a bucket-file
proxy so the model can fetch images by URL.

## Architecture

Two OS threads; no live queries, no batching.

- **Model worker thread** (`src/worker.py`) — loads the model lazily in the
  background, then processes jobs one at a time. `encode_document` for record
  embeddings, `encode_query` for on-the-fly search queries.
- **Main async loop** (`src/main.py`) — FastAPI + SurrealDB (async SDK) + SSE.

Bridge between them: two thread-safe `queue.Queue`s.
- `_dispatch` (loop): read record → push `ModelJob` to worker.
- `_persist` (loop): receive `ModelResult` → `MERGE` embeddings back to the DB.

Queries bypass the DB: `POST /encode` pushes a `QueryJob` and blocks on a
`threading.Condition` until the worker returns the vectors.

## Files

- `src/config.py` — pydantic-settings (`Settings`, `get_settings`).
- `src/state.py` — `ServiceState`, small thread-safe status registry.
- `src/worker.py` — model worker thread, `ModelJob` / `ModelResult`.
- `src/main.py` — app, endpoints, dispatch/persist tasks, SSE, file proxy.
- `src/utils.py` — logging, dependency check, Windows process priority.
- `run.py` — uvicorn entrypoint.

## Endpoints

- `POST /generate` `{"items": ["marker_views:..."]}` → `202`
- `POST /encode` `{"text": "..."}` and/or `{"image_url": "..."}` → embeddings
- `GET /health`
- `GET /tasks`, `GET /tasks/{record_id}`
- `GET /queue/stream` — SSE task feed
- `GET /files/{bucket}/{path}` — bucket file proxy

## Conventions

- **Record IDs**: parse with `RecordID.parse()` and pass the object to the SDK
  (parameterized, injection-safe). Never interpolate raw IDs into SurrealQL.
- **No regex** for validation. For the file-pointer literal only check that a
  segment can't break out of `f"..."` (no `"`, `\`, whitespace, control chars).
- **File read**: `RETURN f"{bucket}:/{path}".get()` — returns bytes.
- **Write back** with `db.merge(...)`, not `update`, so `version` and the
  computed `generated_description` field are left untouched.
- **DB**: use the async SDK; no manual `connect()` or timeouts (it self-connects
  and has its own client timeout). Connection is wrapped in `main.Connection`,
  which retries on startup and pings liveness so an unavailable DB never crashes
  the app.
- DB access stays on the async loop; model inference stays on the worker thread.